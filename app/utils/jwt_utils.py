#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""JWT 解析工具函数。

从 message_preprocessing.py 拆分而来，仅包含与 JWT 相关的解析逻辑。
"""

import base64
import json
from typing import Any, Dict


def _urlsafe_b64decode(data: str) -> bytes:
    """Decode a URL-safe base64 string with proper padding."""
    if isinstance(data, str):
        data_bytes = data.encode("utf-8")
    else:
        data_bytes = data
    padding = b"=" * (-len(data_bytes) % 4)
    return base64.urlsafe_b64decode(data_bytes + padding)


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    """Decode JWT payload without verification to extract metadata."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_raw = _urlsafe_b64decode(parts[1])
        return json.loads(payload_raw.decode("utf-8", errors="ignore"))
    except Exception:
        return {}


def extract_user_id_from_token(token: str) -> str:
    """Extract user_id from a JWT's payload. Fallback to 'guest'."""
    payload = _decode_jwt_payload(token) if token else {}
    for key in ("id", "user_id", "uid", "sub"):
        val = payload.get(key)
        if isinstance(val, (str, int)) and str(val):
            return str(val)
    return "guest"
