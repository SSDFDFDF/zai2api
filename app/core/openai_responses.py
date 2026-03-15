#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import settings
from app.core.openai import get_upstream_client
from app.core.openai_responses_request_adapter import (
    responses_request_to_openai_request,
)
from app.core.openai_responses_serializer import (
    extract_openai_responses_usage,
    openai_chat_response_to_openai_response,
    openai_chat_stream_to_openai_responses_stream,
)
from app.models.openai_responses import OpenAIResponsesRequest
from app.utils.logger import get_logger
from app.utils.request_logging import (
    wrap_openai_responses_stream_with_logging,
    write_request_log,
)
from app.utils.request_source import detect_request_source, format_request_source

logger = get_logger()
router = APIRouter()


@router.post("/v1/responses")
async def create_response(
    body: OpenAIResponsesRequest,
    http_request: Request,
    authorization: Optional[str] = Header(None),
):
    source_info = detect_request_source(
        http_request,
        protocol_hint="openai",
        model_hint=body.model,
    )
    source_prefix = format_request_source(source_info)
    started_at = time.perf_counter()
    bearer_token = (
        authorization[7:]
        if authorization and authorization.startswith("Bearer ")
        else None
    )
    upstream_auth_token: Optional[str] = None

    logger.info(
        "{} Responses req - model: {}, stream: {}, tools: {}",
        source_prefix,
        body.model,
        body.stream,
        len(body.tools) if body.tools else 0,
    )
    logger.debug("{} 客户端 Responses 原样数据: {}", source_prefix, body)

    try:
        if not settings.SKIP_AUTH_TOKEN:
            if not bearer_token:
                raise HTTPException(
                    status_code=401,
                    detail="Missing or invalid Authorization header",
                )
            if bearer_token != settings.AUTH_TOKEN:
                raise HTTPException(status_code=401, detail="Invalid API key")

        try:
            openai_request = responses_request_to_openai_request(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        openai_request.started_at = started_at
        client = get_upstream_client()
        result, upstream_auth_token = await client.chat_completion(
            openai_request,
            http_request=http_request,
        )

        if isinstance(result, dict) and "error" in result:
            error_info = result["error"]
            error_message = error_info.get("message", "Unknown upstream error")
            error_code = error_info.get("code")
            status_code = 404 if error_code == "model_not_found" else 500
            raise HTTPException(status_code=status_code, detail=error_message)

        if body.stream:
            if not hasattr(result, "__aiter__"):
                raise HTTPException(
                    status_code=500,
                    detail="Expected streaming response but got non-streaming result",
                )

            responses_stream = openai_chat_stream_to_openai_responses_stream(
                result,
                model=body.model,
                has_tools=bool(body.tools),
                metadata=body.metadata,
                parallel_tool_calls=body.parallel_tool_calls,
            )
            return StreamingResponse(
                wrap_openai_responses_stream_with_logging(
                    responses_stream,
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

        if not isinstance(result, dict):
            raise HTTPException(
                status_code=500,
                detail="Unexpected non-stream result type from upstream",
            )

        response_payload = openai_chat_response_to_openai_response(
            result,
            metadata=body.metadata,
            parallel_tool_calls=body.parallel_tool_calls,
        )
        usage = extract_openai_responses_usage(response_payload)
        await write_request_log(
            provider="zai",
            model=body.model,
            source_info=source_info,
            auth_token=bearer_token,
            upstream_auth_token=upstream_auth_token,
            success=True,
            started_at=started_at,
            status_code=200,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_creation_tokens=usage["cache_creation_tokens"],
            cache_read_tokens=usage["cache_read_tokens"],
            total_tokens=usage["total_tokens"],
        )
        return JSONResponse(content=response_payload)

    except HTTPException as exc:
        await write_request_log(
            provider="zai",
            model=body.model,
            source_info=source_info,
            auth_token=bearer_token,
            upstream_auth_token=upstream_auth_token,
            success=False,
            started_at=started_at,
            status_code=exc.status_code,
            error_message=str(exc.detail),
        )
        raise
    except Exception as exc:
        logger.error("{} ❌ Responses 请求处理失败: {}", source_prefix, exc)
        await write_request_log(
            provider="zai",
            model=body.model,
            source_info=source_info,
            auth_token=bearer_token,
            upstream_auth_token=upstream_auth_token,
            success=False,
            started_at=started_at,
            status_code=500,
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(exc)}",
        )
