#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Request source detection helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from fastapi import Request


@dataclass(frozen=True)
class RequestSourceInfo:
    """Normalized request-source metadata for logging."""

    source: str
    protocol: str
    client_name: str
    endpoint: str
    user_agent: str


def _normalize_source_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower())
    return normalized.strip("_") or "unknown"


def detect_request_source(
    request: Request,
    protocol_hint: Optional[str] = None,
) -> RequestSourceInfo:
    """Detect the request source from headers and path."""
    headers = request.headers
    endpoint = request.url.path
    user_agent = (headers.get("user-agent") or "").strip()
    user_agent_normalized = user_agent.casefold()

    protocol = (protocol_hint or "").strip().lower()
    if not protocol:
        protocol = "openai" if endpoint.startswith("/v1/") else "unknown"

    explicit_source = headers.get("x-request-source") or headers.get("x-client-source")
    if explicit_source:
        return RequestSourceInfo(
            source=_normalize_source_name(explicit_source),
            protocol=protocol,
            client_name=explicit_source.strip(),
            endpoint=endpoint,
            user_agent=user_agent,
        )

    if "openai" in user_agent_normalized:
        source = "openai_sdk"
        client_name = "OpenAI SDK"
    elif "curl/" in user_agent_normalized:
        source = "curl"
        client_name = "curl"
    elif any(
        token in user_agent_normalized
        for token in ("python-httpx", "httpx/", "python-requests", "requests/")
    ):
        source = "custom_http_client"
        client_name = "HTTP Client"
    elif "mozilla/" in user_agent_normalized:
        source = "browser"
        client_name = "Browser"
    elif protocol == "openai":
        source = "openai_compatible"
        client_name = "OpenAI Compatible"
    else:
        source = "unknown"
        client_name = "Unknown"

    return RequestSourceInfo(
        source=source,
        protocol=protocol,
        client_name=client_name,
        endpoint=endpoint,
        user_agent=user_agent,
    )


def format_request_source(info: RequestSourceInfo) -> str:
    """Render request-source metadata into a compact log prefix."""
    return (
        f"[source={info.source}]"
        f"[protocol={info.protocol}]"
        f"[client={info.client_name}]"
        f"[endpoint={info.endpoint}]"
    )
