#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Request log persistence and OpenAI stream logging helpers."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from app.services.request_log_dao import get_request_log_dao
from app.utils.format import format_compact_number
from app.utils.logger import logger, log_exception
from app.utils.request_source import RequestSourceInfo
from app.utils.utlis import mask_token

CACHE_CREATION_FLOOR = 1024
CACHE_STRIDE = 128


def _empty_usage() -> Dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "total_tokens": 0,
    }


def _coerce_int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any, default: int = 0) -> int:
    coerced = _coerce_int_or_none(value)
    return default if coerced is None else coerced


def _get_first_present(
    mapping: Dict[str, Any],
    *keys: str,
) -> Tuple[Optional[Any], Optional[str]]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key], key
    return None, None


def _estimate_cache_creation_tokens(
    input_tokens: int,
    cache_read_tokens: int,
) -> int:
    if input_tokens < CACHE_CREATION_FLOOR:
        return 0
    cacheable = (input_tokens // CACHE_STRIDE) * CACHE_STRIDE
    return max(0, cacheable - cache_read_tokens)


def _merge_openai_usage(
    current: Dict[str, int],
    update: Dict[str, int],
) -> Dict[str, int]:
    merged = dict(current)

    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
    ):
        value = _coerce_int(update.get(key))
        if value > 0:
            merged[key] = value

    total_tokens = _coerce_int(update.get("total_tokens"))
    if total_tokens > 0:
        merged["total_tokens"] = total_tokens
    else:
        merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]

    return merged


def extract_openai_usage(response: Dict[str, Any]) -> Dict[str, int]:
    """Extract OpenAI-compatible usage, with cache fallback estimation."""
    usage = response.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    input_details = usage.get("input_token_details") or {}

    raw, _ = _get_first_present(usage, "prompt_tokens", "input_tokens")
    input_tokens = _coerce_int(raw)

    raw, _ = _get_first_present(usage, "completion_tokens", "output_tokens")
    output_tokens = _coerce_int(raw)

    raw, _ = _get_first_present(usage, "total_tokens")
    total_tokens = _coerce_int(raw)
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    raw, source = _get_first_present(
        usage,
        "cache_read_input_tokens",
        "cache_read_tokens",
        "cached_tokens",
    )
    if not source:
        raw, source = _get_first_present(
            prompt_details,
            "cached_tokens",
            "cache_read_tokens",
            "cache_read_input_tokens",
        )
    if not source:
        raw, source = _get_first_present(
            input_details,
            "cached_tokens",
            "cache_read_input_tokens",
            "cache_read_tokens",
        )
    cache_read_tokens = max(0, _coerce_int(raw))

    raw, source = _get_first_present(
        usage,
        "cache_creation_input_tokens",
        "cache_creation_tokens",
    )
    if not source:
        raw, source = _get_first_present(
            prompt_details,
            "cache_creation_tokens",
            "cache_creation_input_tokens",
        )
    if not source:
        raw, source = _get_first_present(
            input_details,
            "cache_creation_input_tokens",
            "cache_creation_tokens",
        )
    reported_cache_creation = _coerce_int_or_none(raw)

    if reported_cache_creation is not None:
        cache_creation_tokens = max(0, reported_cache_creation)
    else:
        cache_creation_tokens = _estimate_cache_creation_tokens(
            input_tokens,
            cache_read_tokens,
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "total_tokens": total_tokens,
    }


async def write_request_log(
    *,
    provider: str,
    model: str,
    source_info: RequestSourceInfo,
    auth_token: Optional[str] = None,
    upstream_auth_token: Optional[str] = None,
    success: bool,
    started_at: float,
    status_code: int = 200,
    first_token_time: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    total_tokens: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """Persist a request log entry without breaking request handling."""
    duration = max(0.0, time.perf_counter() - started_at)

    status_tag = "OK" if success else "ERR"
    masked_auth_token = mask_token(auth_token)
    masked_upstream_auth_token = mask_token(upstream_auth_token)
    auth_segment = f" | Auth: {masked_auth_token}" if masked_auth_token else ""
    upstream_auth_segment = (
        f" | Upstream Auth: {masked_upstream_auth_token}"
        if masked_upstream_auth_token
        else ""
    )
    logger.info(
        "[%s] [%s] %s%s%s | In: %s | Out: %s | %.2fs",
        status_tag,
        provider,
        model,
        auth_segment,
        upstream_auth_segment,
        format_compact_number(input_tokens),
        format_compact_number(output_tokens),
        duration,
    )

    try:
        dao = get_request_log_dao()
        dao.add_log_nowait(
            provider=provider,
            endpoint=source_info.endpoint,
            source=source_info.source,
            protocol=source_info.protocol,
            client_name=source_info.client_name,
            auth_token=auth_token,
            upstream_auth_token=upstream_auth_token,
            model=model,
            status_code=status_code,
            success=success,
            duration=duration,
            first_token_time=first_token_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            total_tokens=total_tokens,
            error_message=error_message,
        )
    except Exception as exc:
        log_exception(logger, "Failed to write request log")


def _openai_response_has_output(response: Dict[str, Any]) -> bool:
    """Check whether a complete (non-stream) OpenAI response has visible output."""
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        return False
    message = choices[0].get("message") or {}
    if message.get("content") or message.get("reasoning_content"):
        return True
    if message.get("tool_calls"):
        return True
    return False


def _openai_payload_has_output(payload: Dict[str, Any]) -> bool:
    """Treat only visible content/tool deltas as first-token output."""
    choice = ((payload.get("choices") or [{}])[0]) if isinstance(payload, dict) else {}
    delta = choice.get("delta") or {}

    if delta.get("content") or delta.get("reasoning_content"):
        return True
    if delta.get("tool_calls"):
        return True
    return False


def _iter_sse_payloads(chunk: str):
    for line in chunk.splitlines():
        if not line.startswith("data: "):
            continue
        payload_text = line[6:].strip()
        if payload_text:
            yield payload_text


async def wrap_openai_stream_with_logging(
    stream: AsyncGenerator[str, None],
    *,
    provider: str,
    model: str,
    source_info: RequestSourceInfo,
    auth_token: Optional[str] = None,
    upstream_auth_token: Optional[str] = None,
    started_at: float,
) -> AsyncGenerator[str, None]:
    """Wrap an OpenAI-compatible SSE stream and persist completion metadata."""
    success = True
    status_code = 200
    error_message: Optional[str] = None
    first_token_time = 0.0
    has_output = False
    usage = _empty_usage()

    try:
        async for chunk in stream:
            for payload_text in _iter_sse_payloads(chunk):
                if payload_text == "[DONE]":
                    continue

                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    payload = None

                if not isinstance(payload, dict):
                    continue

                if "error" in payload:
                    success = False
                    error = payload.get("error") or {}
                    error_message = error.get("message") or "Unknown stream error"
                    status_code = _coerce_int(error.get("code"), 500)
                    continue

                if _openai_payload_has_output(payload):
                    has_output = True
                    if not first_token_time:
                        first_token_time = max(0.0, time.perf_counter() - started_at)

                if payload.get("usage"):
                    usage = _merge_openai_usage(
                        usage,
                        extract_openai_usage(payload),
                    )

            yield chunk
    except Exception as exc:
        success = False
        status_code = 500
        error_message = str(exc)
        raise
    finally:
        if success and not has_output:
            success = False
            error_message = error_message or "Empty response: no content in stream"
        await write_request_log(
            provider=provider,
            model=model,
            source_info=source_info,
            auth_token=auth_token,
            upstream_auth_token=upstream_auth_token,
            success=success,
            started_at=started_at,
            status_code=status_code,
            first_token_time=first_token_time,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_creation_tokens=usage["cache_creation_tokens"],
            cache_read_tokens=usage["cache_read_tokens"],
            total_tokens=usage["total_tokens"],
            error_message=error_message,
        )
