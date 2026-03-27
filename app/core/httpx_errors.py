#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helpers for normalized httpx error logging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.core.openai_compat import get_error_message
    from app.core.retry_policy import summarize_upstream_error_text


_DEFAULT_MAX_BODY = 240


def _format_context(context: str) -> str:
    if not context:
        return ""
    return f"[{context}] "


def _sanitize_text(text: str, max_length: int = _DEFAULT_MAX_BODY) -> str:
    if not text:
        return ""
    sanitized = " ".join(text.split())
    if len(sanitized) > max_length:
        return sanitized[:max_length].rstrip() + "..."
    return sanitized


def _format_attempt(attempt: Optional[int]) -> str:
    if attempt is None:
        return ""
    return f" attempt={attempt}"


def normalize_httpx_exception(
    exc: Exception,
    *,
    method: str,
    url: str,
    context: str = "",
    attempt: Optional[int] = None,
) -> str:
    """Normalize httpx exception into a short log message."""
    from app.core.openai_compat import get_error_message

    friendly = get_error_message(exc)
    if not friendly:
        friendly = repr(exc)
    return (
        f"{_format_context(context)}HTTP {method.upper()} {url}"
        f" failed{_format_attempt(attempt)}: {friendly}"
    )


def normalize_httpx_response(
    status_code: int,
    body_text: str,
    *,
    content_type: Optional[str] = None,
    method: str,
    url: str,
    context: str = "",
) -> str:
    """Normalize non-200 response into a short log message."""
    from app.core.retry_policy import summarize_upstream_error_text

    summarized = summarize_upstream_error_text(
        status_code,
        body_text or "",
        content_type=content_type,
    )
    summarized = _sanitize_text(summarized, max_length=_DEFAULT_MAX_BODY)
    message = f"{_format_context(context)}HTTP {method.upper()} {url} returned {status_code}"
    if summarized:
        return f"{message}: {summarized}"
    return message
