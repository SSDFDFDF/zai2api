#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""双池重试策略模块。

将原 UpstreamClient 中的双池（认证池 + 匿名池）重试决策逻辑提取为
独立的 RetryPolicy 类，以及错误解析工具函数。
所有方法签名与原实现保持一致。
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings
from app.utils.logger import logger
from app.utils.token_pool import get_token_pool
from app.utils.guest_session_pool import get_guest_session_pool

_HTML_ERROR_PAGE_RE = re.compile(
    r"<(?:!doctype|html|head|body|title|h1|div|p|code)\b",
    re.IGNORECASE,
)
_HTML_CONTENT_TYPE_RE = re.compile(
    r"(?:^|[;\s])(text/html|application/xhtml\+xml)(?:$|[;\s])",
    re.IGNORECASE,
)


def is_upstream_page_response(
    content_type: Optional[str],
    error_text: str,
) -> bool:
    """判断上游响应是否为网页/拦截页，而非标准 API 错误。"""
    if content_type and _HTML_CONTENT_TYPE_RE.search(content_type):
        return True

    text = (error_text or "").lstrip()
    return bool(text) and bool(_HTML_ERROR_PAGE_RE.search(text[:2048]))


def get_upstream_page_error_message(status_code: Optional[int]) -> str:
    """为网页类型错误生成统一消息。"""
    if not isinstance(status_code, int):
        return "上游返回网页错误"

    normalized_status = status_code if status_code >= 400 else 502

    if normalized_status in (401, 403, 405):
        base_message = "上游网页访问被拦截"
    elif normalized_status == 429:
        base_message = "上游网页请求过于频繁"
    elif normalized_status in (502, 504):
        base_message = "上游网页网关异常"
    elif normalized_status == 503:
        base_message = "上游网页服务不可用"
    elif normalized_status >= 500:
        base_message = "上游网页服务异常"
    else:
        base_message = "上游返回网页错误"

    return f"{base_message} (HTTP {normalized_status})"


def summarize_upstream_error_text(
    status_code: Optional[int],
    error_text: str,
    *,
    content_type: Optional[str] = None,
) -> str:
    """将冗长的上游错误文本压缩为适合返回给客户端的短消息。"""
    text = (error_text or "").strip()
    if not text:
        return ""

    if is_upstream_page_response(content_type, text):
        return get_upstream_page_error_message(status_code)

    return text


# ---------------------------------------------------------------------------
# 错误解析工具
# ---------------------------------------------------------------------------


def extract_upstream_error_details(
    status_code: int,
    error_text: str,
    content_type: Optional[str] = None,
) -> Tuple[Optional[int], str]:
    """解析上游错误响应中的 code/message。

    Args:
        status_code: HTTP 响应状态码。
        error_text: 响应 body 文本。

    Returns:
        ``(error_code, error_message)`` 二元组，解析失败时 code 为 None。
    """
    parsed_code: Optional[int] = None
    fallback_message = summarize_upstream_error_text(
        status_code,
        error_text,
        content_type=content_type,
    )
    parsed_message = ""

    if is_upstream_page_response(content_type, error_text):
        return status_code, fallback_message

    try:
        payload = json.loads(error_text)
    except Exception:
        return parsed_code, fallback_message

    if not isinstance(payload, dict):
        return parsed_code, fallback_message

    candidates = [
        payload,
        payload.get("error") if isinstance(payload.get("error"), dict) else None,
        payload.get("detail") if isinstance(payload.get("detail"), dict) else None,
        payload.get("data") if isinstance(payload.get("data"), dict) else None,
    ]

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        candidate_message = ""
        code = candidate.get("code")
        if isinstance(code, int):
            parsed_code = code
        elif isinstance(code, str) and code.isdigit():
            parsed_code = int(code)

        for key in ("message", "msg", "detail", "error"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                candidate_message = summarize_upstream_error_text(
                    status_code,
                    value.strip(),
                )
                parsed_message = candidate_message
                break

        if parsed_code is not None or candidate_message:
            break

    if not parsed_message:
        parsed_message = fallback_message

    return parsed_code, parsed_message


def is_concurrency_limited(
    status_code: int,
    error_code: Optional[int],
    error_message: str,
) -> bool:
    """判断是否为上游并发限制/429 场景。

    Args:
        status_code: HTTP 状态码。
        error_code: 解析出的错误 code（可能为 None）。
        error_message: 解析出的错误消息文本。

    Returns:
        True 表示命中并发限制，需要重试。
    """
    message = (error_message or "").casefold()
    return (
        status_code == 429
        or error_code == 429
        or "concurrency" in message
        or "too many requests" in message
    )


# ---------------------------------------------------------------------------
# 重试策略
# ---------------------------------------------------------------------------


class RetryPolicy:
    """双池重试策略。

    封装认证号池与匿名号池的重试预算计算和切换决策。
    """

    def __init__(self) -> None:
        self.logger = logger

    async def get_guest_retry_limit(self) -> int:
        """匿名号池可提供的最大重试预算。"""
        if not settings.ANONYMOUS_MODE:
            return 0

        guest_pool = get_guest_session_pool()
        if not guest_pool:
            return max(2, settings.GUEST_POOL_SIZE + 1)

        pool_status = guest_pool.get_pool_status()
        available_sessions = int(
            pool_status.get("valid_sessions")
            or pool_status.get("available_sessions")
            or 0
        )
        return max(2, available_sessions + 1)

    async def get_authenticated_retry_limit(self) -> int:
        """认证号池与静态 Token 可提供的最大重试预算。"""
        available_tokens = 0
        token_pool = get_token_pool()
        if token_pool:
            status = await token_pool.get_pool_status()
            available_tokens = int(status.get("available_tokens", 0) or 0)
        return max(0, available_tokens)

    async def get_total_retry_limit(self) -> int:
        """综合认证号池与匿名号池的最大尝试次数。"""
        auth_limit = await self.get_authenticated_retry_limit()
        guest_limit = await self.get_guest_retry_limit()
        return max(1, auth_limit + guest_limit)

    def is_guest_auth(self, transformed: Dict[str, Any]) -> bool:
        """判断当前请求是否使用匿名会话。"""
        return str(transformed.get("auth_mode") or "") == "guest"

    def should_retry_guest_session(
        self,
        status_code: int,
        is_concurrency_limited_flag: bool,
        attempt: int,
        max_attempts: int,
        transformed: Dict[str, Any],
    ) -> bool:
        """判断匿名号池是否需要刷新会话后重试。"""
        return (
            settings.ANONYMOUS_MODE
            and self.is_guest_auth(transformed)
            and (status_code == 401 or is_concurrency_limited_flag)
            and attempt + 1 < max_attempts
        )

    def should_retry_authenticated_session(
        self,
        status_code: int,
        is_concurrency_limited_flag: bool,
        attempt: int,
        max_attempts: int,
        transformed: Dict[str, Any],
    ) -> bool:
        """判断认证号池是否需要切号重试。"""
        current_token = str(transformed.get("token") or "")
        return (
            not self.is_guest_auth(transformed)
            and bool(current_token)
            and (status_code == 401 or is_concurrency_limited_flag)
            and attempt + 1 < max_attempts
        )

    async def release_guest_session(self, transformed: Dict[str, Any]) -> None:
        """释放当前匿名会话占用。"""
        if not self.is_guest_auth(transformed) or not settings.ANONYMOUS_MODE:
            return

        guest_pool = get_guest_session_pool()
        guest_user_id = str(
            transformed.get("guest_user_id") or transformed.get("user_id") or ""
        )
        if guest_pool and guest_user_id:
            guest_pool.release(guest_user_id)

    async def report_guest_session_failure(
        self,
        transformed: Dict[str, Any],
        *,
        is_concurrency_limited_flag: bool = False,
    ) -> None:
        """上报匿名会话失败并补齐新会话。"""
        if not self.is_guest_auth(transformed) or not settings.ANONYMOUS_MODE:
            return

        guest_pool = get_guest_session_pool()
        guest_user_id = str(
            transformed.get("guest_user_id") or transformed.get("user_id") or ""
        )
        if not guest_pool or not guest_user_id:
            return

        if is_concurrency_limited_flag:
            await guest_pool.cleanup_idle_chats()

        await guest_pool.report_failure(guest_user_id)
