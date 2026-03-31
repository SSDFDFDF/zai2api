#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helpers for optional Resin-specific upstream request compatibility."""

from __future__ import annotations

import hashlib
from typing import MutableMapping

from app.core.config import settings

RESIN_ACCOUNT_HEADER = "X-Resin-Account"
RESIN_ACCOUNT_PREFIX = "zai-token"


def build_resin_account(token: str) -> str:
    """Build a stable, opaque Resin account key for one upstream token."""
    raw = str(token or "").strip()
    if not raw:
        return ""

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{RESIN_ACCOUNT_PREFIX}:{digest}"


def apply_resin_account_header(
    headers: MutableMapping[str, str],
    token: str,
) -> MutableMapping[str, str]:
    """Attach the Resin account header when compatibility mode is enabled."""
    if not settings.RESIN_COMPAT_ENABLED:
        headers.pop(RESIN_ACCOUNT_HEADER, None)
        return headers

    account = build_resin_account(token)
    if account:
        headers[RESIN_ACCOUNT_HEADER] = account
    else:
        headers.pop(RESIN_ACCOUNT_HEADER, None)
    return headers
