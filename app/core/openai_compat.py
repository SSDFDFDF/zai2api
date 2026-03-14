#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenAI 兼容响应辅助函数。"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger()
SYSTEM_FINGERPRINT = "fp_api_proxy_001"


def create_chat_id() -> str:
    """生成聊天 ID。"""
    return f"chatcmpl-{uuid.uuid4().hex}"


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


def create_openai_response(
    chat_id: str,
    model: str,
    content: str,
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """创建 OpenAI 格式的非流式响应。"""
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
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


def create_openai_response_with_reasoning(
    chat_id: str,
    model: str,
    content: str,
    reasoning_content: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """创建包含 reasoning/tool_calls 的 OpenAI 响应。"""
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }

    if reasoning_content and reasoning_content.strip():
        message["reasoning_content"] = reasoning_content

    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
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


def handle_error(error: Exception, context: str = "") -> Dict[str, Any]:
    """统一错误处理。"""
    # 未知模型 → model_not_found（路由层据此返回 404）
    if isinstance(error, ValueError) and "不支持的模型" in str(error):
        logger.warning(str(error))
        return {
            "error": {
                "message": str(error),
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }

    friendly_msg = get_error_message(error)
    error_msg = f"上游{context}错误: {friendly_msg}" if context else f"上游错误: {friendly_msg}"
    logger.error("{} (raw: {})", error_msg, error)
    
    return {
        "error": {
            "message": error_msg,
            "type": "upstream_error",
            "code": "internal_error",
        }
    }
