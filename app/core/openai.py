#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenAI 兼容 API 端点。

提供 /v1/models 和 /v1/chat/completions 端点。
"""

import time
from typing import Optional

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import settings
from app.exceptions import (
    AppError,
    AuthenticationError,
    UpstreamError,
)
from app.models.schemas import (
    Model,
    ModelsResponse,
    OpenAIRequest,
)
from app.core.openai_compat import resolve_http_error_status
from app.core.upstream import UpstreamClient
from app.utils.logger import logger
from app.utils.request_logging import (
    extract_openai_usage,
    wrap_openai_stream_with_logging,
    write_request_log,
)
from app.utils.request_source import detect_request_source, format_request_source

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


async def reset_upstream_client() -> None:
    """Close and clear the cached upstream client singleton."""
    global _upstream_client
    if _upstream_client is not None:
        await _upstream_client.close()
    _upstream_client = None


@router.get("/v1/models")
async def list_models():
    """返回当前服务支持的模型列表（含能力声明）。"""
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
    )
    source_prefix = format_request_source(source_info)
    started_at = time.perf_counter()
    body.started_at = started_at
    bearer_token = (
        authorization[7:]
        if authorization and authorization.startswith("Bearer ")
        else None
    )

    logger.debug(
        "%s OpenAI req - model: %s, stream: %s, messages: %s",
        source_prefix, body.model, body.stream, len(body.messages),
    )
    logger.debug("%s 客户端请求原样数据: %s", source_prefix, body)
    upstream_auth_token: Optional[str] = None

    # 认证检查
    if not settings.SKIP_AUTH_TOKEN:
        if not bearer_token:
            raise AuthenticationError("Missing or invalid Authorization header")

        if bearer_token != settings.AUTH_TOKEN:
            raise AuthenticationError("Invalid API key")

    # 调用上游
    client = get_upstream_client()
    result, upstream_auth_token = await client.chat_completion(
        body,
        http_request=http_request,
    )

    # 处理上游错误
    if isinstance(result, dict) and "error" in result:
        error_info = result["error"]
        error_message = error_info.get("message", "Unknown upstream error")
        error_code = error_info.get("code")
        error_type = error_info.get("type")
        status_code = resolve_http_error_status(error_code, error_type)

        # 记录错误日志
        await write_request_log(
            provider="zai",
            model=body.model,
            source_info=source_info,
            auth_token=bearer_token,
            upstream_auth_token=upstream_auth_token,
            success=False,
            started_at=started_at,
            status_code=status_code,
            error_message=error_message,
        )

        raise UpstreamError(
            message=error_message,
            upstream_code=error_code if isinstance(error_code, int) else None,
            upstream_status=status_code,
            upstream_type=error_type,
        )

    # 流式响应
    if body.stream:
        if hasattr(result, "__aiter__"):
            return StreamingResponse(
                wrap_openai_stream_with_logging(
                    result,
                    provider="zai",
                    model=body.model,
                    source_info=source_info,
                    auth_token=bearer_token,
                    upstream_auth_token=upstream_auth_token,
                    started_at=started_at,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        raise UpstreamError(
            message="Expected streaming response but got non-streaming result",
        )

    # 非流式响应
    if isinstance(result, dict):
        usage = extract_openai_usage(result)
        await write_request_log(
            provider="zai",
            model=body.model,
            source_info=source_info,
            auth_token=bearer_token,
            upstream_auth_token=upstream_auth_token,
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

    # 不应该到达这里
    raise UpstreamError(
        message="Unexpected non-stream result type from upstream",
    )
