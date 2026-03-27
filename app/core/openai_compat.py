#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenAI 兼容响应辅助函数。

提供：
- OpenAI 格式的响应构建
- 错误消息提取
- 错误响应构建
- HTTP 状态码映射
"""

import json
import time
from typing import Any, Dict, Optional

from app.core.retry_policy import summarize_upstream_error_text
from app.exceptions import AppError, UpstreamError, ValidationError
from app.utils.logger import logger

SYSTEM_FINGERPRINT = "fp_api_proxy_001"


def create_openai_chunk(
    chat_id: str,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
    created: Optional[int] = None,
) -> Dict[str, Any]:
    """创建 OpenAI 格式的流式响应块。"""
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
        "system_fingerprint": SYSTEM_FINGERPRINT,
    }


def create_openai_response_with_reasoning(
    chat_id: str,
    model: str,
    content: str,
    reasoning_content: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """创建包含 reasoning 的 OpenAI 响应。"""
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }

    if reasoning_content and reasoning_content.strip():
        message["reasoning_content"] = reasoning_content

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "system_fingerprint": SYSTEM_FINGERPRINT,
    }


def format_sse_chunk(chunk: Dict[str, Any]) -> str:
    """格式化 SSE 响应块。"""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def format_sse_done() -> str:
    """格式化 SSE 结束标记。"""
    return "data: [DONE]\n\n"


def get_error_message(error: Exception) -> str:
    """从异常中提取更友好的错误消息。"""
    error_str = str(error).strip()
    if not error_str:
        error_str = repr(error)

    summarized = summarize_upstream_error_text(None, error_str)
    if summarized != error_str:
        return summarized
    
    # 特殊处理 SSL 错误
    if "SSL" in error_str or "ssl" in error_str:
        return f"SSL/TLS 连接握手失败，请检查证书或网络环境: {error_str}"
    
    # 特殊处理代理错误
    if "proxy" in error_str.lower() or "Proxy" in error_str:
        return f"代理连接失败，请检查代理设置或服务器联通性: {error_str}"
    
    # 处理超时
    if "timeout" in error_str.lower():
        return f"连接上游响应超时: {error_str}"
        
    return error_str


def resolve_http_error_status(error_code: Any, error_type: Any) -> int:
    """根据标准 error.code / error.type 推导 HTTP 状态码。"""
    if error_code == "model_not_found":
        return 404
    if isinstance(error_code, int) and 400 <= error_code <= 599:
        return error_code
    if (
        error_type == "invalid_request_error"
        or error_code == "invalid_request_error"
    ):
        return 400
    return 500


def handle_error(error: Exception, context: str = "") -> Dict[str, Any]:
    """统一错误处理。

    将异常转换为 OpenAI 兼容的错误响应格式。
    注意：此函数主要用于向后兼容，新代码应直接抛出 AppError 子类。
    """
    from app.exceptions import ModelNotFoundError

    # 如果已经是 AppError，直接返回其字典形式
    if isinstance(error, AppError):
        return error.to_dict()

    # 未知模型 → ModelNotFoundError (404)
    error_str = str(error)
    if isinstance(error, ValueError) and ("Unsupported model" in error_str or "不支持的模型" in error_str):
        # 从错误消息中提取模型名称
        import re
        # 匹配 "Unsupported model: xxx" 或 "不支持的模型: xxx"
        match = re.search(r"(?:Unsupported model|不支持的模型)[:\s]*(\S+)", error_str)
        model = match.group(1) if match else "unknown"
        exc = ModelNotFoundError(model)
        logger.warning(error_str)
        return exc.to_dict()

    if isinstance(error, ValueError):
        exc = ValidationError(str(error))
        logger.warning("invalid request: %s", error)
        return exc.to_dict()

    # 其他异常 → UpstreamError
    friendly_msg = get_error_message(error)
    error_msg = f"上游{context}错误: {friendly_msg}" if context else f"上游错误: {friendly_msg}"
    logger.error("%s (raw: %s)", error_msg, error)

    exc = UpstreamError(message=error_msg)
    return exc.to_dict()
