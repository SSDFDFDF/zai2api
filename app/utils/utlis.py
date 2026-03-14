#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared utility helpers."""

from __future__ import annotations

from typing import Optional


def mask_token(
    token: Optional[str], *, keep_prefix: int = 6, keep_suffix: int = 4
) -> str:
    """Return a short recognizable token fragment for logs and UI."""
    value = str(token or "").strip()
    if not value:
        return ""
    if len(value) <= keep_prefix + keep_suffix + 3:
        return value[: min(len(value), keep_prefix)] + "..."
    return f"{value[:keep_prefix]}...{value[-keep_suffix:]}"
