#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import settings
from app.models.schemas import (
    Choice,
    Message,
    Model,
    ModelsResponse,
    OpenAIRequest,
    OpenAIResponse,
    Usage,
)
from app.core.upstream import UpstreamClient
from app.utils.logger import get_logger
from app.utils.request_logging import (
    extract_openai_usage,
    wrap_openai_stream_with_logging,
    write_request_log,
)
from app.utils.request_source import detect_request_source, format_request_source

logger = get_logger()
router = APIRouter()

_upstream_client: Optional[UpstreamClient] = None


def get_upstream_client() -> UpstreamClient:
    """获取懒加载的上游适配器单例。"""
    global _upstream_client
    if _upstream_client is None:
        _upstream_client = UpstreamClient()
    return _upstream_client


def get_upstream_client_if_ready() -> Optional[UpstreamClient]:
    """Return upstream client if already initialized."""
    return _upstream_client


async def handle_non_stream_response(stream_response, request: OpenAIRequest) -> JSONResponse:
    """处理非流式响应。"""
    logger.info("[stream] start handle non stream response")

    full_content = []
    reasoning_content = []
    tool_calls_dict = {}
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async for chunk_data in stream_response():
        if chunk_data.startswith("data: "):
            chunk_str = chunk_data[6:].strip()
            if chunk_str and chunk_str != "[DONE]":
                try:
                    chunk = json.loads(chunk_str)
                    
                    if "usage" in chunk and chunk["usage"]:
                        usage = chunk["usage"]
                        usage_info.update(usage)

                    if "choices" in chunk and chunk["choices"]:
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        
                        if "content" in delta and delta["content"]:
                            full_content.append(delta["content"])
                            
                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            reasoning_content.append(delta["reasoning_content"])
                            
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                index = tc.get("index", 0)
                                if index not in tool_calls_dict:
                                    tool_calls_dict[index] = {
                                        "id": tc.get("id") or f"call_{int(time.time())}_{index}",
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("function", {}).get("name", ""),
                                            "arguments": ""
                                        }
                                    }
                                if "function" in tc and isinstance(tc["function"].get("arguments"), str):
                                    tool_calls_dict[index]["function"]["arguments"] += tc["function"]["arguments"]
                except json.JSONDecodeError:
                    continue

    message_kwargs = {
        "role": "assistant",
        "content": "".join(full_content) or None,
    }
    
    if reasoning_content:
        message_kwargs["reasoning_content"] = "".join(reasoning_content)
        
    if tool_calls_dict:
        message_kwargs["tool_calls"] = [tool_calls_dict[k] for k in sorted(tool_calls_dict.keys())]
    else:
        message_kwargs["tool_calls"] = None

    response_data = OpenAIResponse(
        id=f"chatcmpl-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(**message_kwargs),
                finish_reason="tool_calls" if tool_calls_dict else "stop",
            )
        ],
        usage=Usage(
            prompt_tokens=usage_info.get("prompt_tokens", 0),
            completion_tokens=usage_info.get("completion_tokens", 0),
            total_tokens=usage_info.get("total_tokens", 0),
        ),
    )

    logger.info("[stream] end handle non stream response")
    return JSONResponse(content=response_data.model_dump(exclude_none=True))


@router.get("/v1/models")
async def list_models():
    """返回当前服务支持的模型列表（含能力声明）。"""
    try:
        client = get_upstream_client()
        current_time = int(time.time())
        model_manager = client._model_manager
        response = ModelsResponse(
            data=[
                Model(
                    id=model_id,
                    created=current_time,
                    owned_by=settings.SERVICE_NAME,
                    capabilities=model_manager.get_model_capabilities(model_id),
                )
                for model_id in client.get_supported_models()
            ]
        )
        return JSONResponse(content=response.model_dump(exclude_none=True))
    except Exception as exc:
        logger.error(f"❌ 获取模型列表失败: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {exc}")


@router.post("/v1/chat/completions")
async def chat_completions(
    body: OpenAIRequest,
    http_request: Request,
    authorization: Optional[str] = Header(None),
):
    """直接调用上游适配器处理请求。"""
    source_info = detect_request_source(
        http_request,
        protocol_hint="openai",
        model_hint=body.model,
    )
    source_prefix = format_request_source(source_info)
    started_at = time.perf_counter()
    body.started_at = started_at

    role = body.messages[0].role if body.messages else "unknown"
    logger.info(
        f"{source_prefix} OpenAI req - model: {body.model}, stream: {body.stream}, messages: {len(body.messages)}, role: {role}, tools: {len(body.tools) if body.tools else 0}"
    )
    logger.debug(f"{source_prefix} 客户端请求原样数据: {body}")

    try:
        if not settings.SKIP_AUTH_TOKEN:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

            api_key = authorization[7:]
            if api_key != settings.AUTH_TOKEN:
                raise HTTPException(status_code=401, detail="Invalid API key")

        client = get_upstream_client()
        result = await client.chat_completion(body, http_request=http_request)

        if isinstance(result, dict) and "error" in result:
            error_info = result["error"]
            error_message = error_info.get("message", "Unknown upstream error")
            error_code = error_info.get("code")
            status_code = 404 if error_code == "model_not_found" else 500
            raise HTTPException(status_code=status_code, detail=error_message)

        if body.stream:
            if hasattr(result, "__aiter__"):
                return StreamingResponse(
                    wrap_openai_stream_with_logging(
                        result,
                        provider="zai",
                        model=body.model,
                        source_info=source_info,
                        started_at=started_at,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            raise HTTPException(
                status_code=500,
                detail="Expected streaming response but got non-streaming result",
            )

        if isinstance(result, dict):
            usage = extract_openai_usage(result)
            await write_request_log(
                provider="zai",
                model=body.model,
                source_info=source_info,
                success="error" not in result,
                started_at=started_at,
                status_code=200 if "error" not in result else 500,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cache_creation_tokens=usage["cache_creation_tokens"],
                cache_read_tokens=usage["cache_read_tokens"],
                total_tokens=usage["total_tokens"],
                error_message=(result.get("error") or {}).get("message") if isinstance(result, dict) else None,
            )
            return JSONResponse(content=result)

        response = await handle_non_stream_response(result, body)
        response_body = json.loads(response.body)
        usage = extract_openai_usage(response_body)
        await write_request_log(
            provider="zai",
            model=body.model,
            source_info=source_info,
            success=True,
            started_at=started_at,
            status_code=200,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_creation_tokens=usage["cache_creation_tokens"],
            cache_read_tokens=usage["cache_read_tokens"],
            total_tokens=usage["total_tokens"],
        )
        return response

    except HTTPException as exc:
        await write_request_log(
            provider="zai",
            model=body.model,
            source_info=source_info,
            success=False,
            started_at=started_at,
            status_code=exc.status_code,
            error_message=str(exc.detail),
        )
        raise
    except Exception as exc:
        logger.error(f"{source_prefix} ❌ 请求处理失败: {exc}")
        await write_request_log(
            provider="zai",
            model=body.model,
            source_info=source_info,
            success=False,
            started_at=started_at,
            status_code=500,
            error_message=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")
