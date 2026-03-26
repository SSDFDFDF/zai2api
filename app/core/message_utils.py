#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simplified OpenAI message preprocessing utilities."""

import json
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Content extraction helper
# ---------------------------------------------------------------------------


def _extract_text_from_content(
    content: Any,
    *,
    include_structured: bool = False,
) -> str:
    """Extract text parts from OpenAI-compatible content payloads."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif include_structured and isinstance(item, dict):
                try:
                    parts.append(json.dumps(item, ensure_ascii=False))
                except Exception:
                    parts.append(str(item))
        return " ".join(part for part in parts if part).strip()

    if content is None:
        return ""

    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


# ---------------------------------------------------------------------------
# Last user text extraction
# ---------------------------------------------------------------------------


def extract_last_user_text(messages: List[Dict[str, Any]]) -> str:
    """Extract the last user text from the original OpenAI message history."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = _extract_text_from_content(message.get("content"))
        if content:
            return content
    return ""


# ---------------------------------------------------------------------------
# Simplified message preprocessing
# ---------------------------------------------------------------------------


def preprocess_openai_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize OpenAI history into shapes accepted by the upstream service.

    Simplified conversions:
    - ``developer`` role -> ``system``
    - Everything else passes through unchanged

    Args:
        messages: OpenAI-format message list (already model_dump'd).

    Returns:
        Normalized message list suitable for upstream consumption.
    """
    normalized: List[Dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")

        # developer -> system
        if role == "developer":
            converted = dict(message)
            converted["role"] = "system"
            normalized.append(converted)
            continue

        # Everything else passes through unchanged
        normalized.append(dict(message))

    return normalized
