#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helpers for deriving upstream URLs from the configured chat endpoint."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from app.core.config import settings

DEFAULT_CHAT_COMPLETION_ENDPOINT = "https://chat.z.ai/api/v2/chat/completions"
_KNOWN_CHAT_COMPLETION_SUFFIXES = (
    "/api/v2/chat/completions",
    "/api/v1/chat/completions",
    "/v1/chat/completions",
    "/chat/completions",
)


def get_api_endpoint() -> str:
    """Return the configured chat-completion endpoint with a safe fallback."""
    endpoint = str(settings.API_ENDPOINT or "").strip()
    return endpoint or DEFAULT_CHAT_COMPLETION_ENDPOINT


def derive_upstream_base_url(api_endpoint: str) -> str:
    """Derive a reusable upstream base URL from a chat-completion endpoint.

    Examples:
        https://chat.z.ai/api/v2/chat/completions
            -> https://chat.z.ai
        http://host/token/platform/https/chat.z.ai/api/v2/chat/completions
            -> http://host/token/platform/https/chat.z.ai
    """
    endpoint = str(api_endpoint or "").strip()
    if not endpoint:
        endpoint = DEFAULT_CHAT_COMPLETION_ENDPOINT

    parts = urlsplit(endpoint)
    if not parts.scheme or not parts.netloc:
        return endpoint.rstrip("/")

    path = parts.path.rstrip("/")
    for suffix in _KNOWN_CHAT_COMPLETION_SUFFIXES:
        if path.endswith(suffix):
            base_path = path[: -len(suffix)].rstrip("/")
            return urlunsplit((parts.scheme, parts.netloc, base_path, "", ""))

    if not path:
        return urlunsplit((parts.scheme, parts.netloc, "", "", ""))

    base_path = path.rsplit("/", 1)[0]
    return urlunsplit((parts.scheme, parts.netloc, base_path, "", ""))


def get_upstream_base_url() -> str:
    """Return the configured base URL for upstream auxiliary endpoints."""
    return derive_upstream_base_url(get_api_endpoint())


def build_upstream_url(path: str) -> str:
    """Join a relative upstream path onto the configured upstream base URL."""
    base_url = get_upstream_base_url().rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{base_url}{normalized_path}"
