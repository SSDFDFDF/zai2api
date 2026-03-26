#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI 签名工具模块
"""

import hmac
import hashlib
import base64
from typing import Dict


def generate_signature(e: str, t: str, s: int) -> dict:
    """Generate signature matching JavaScript zs function.

    Args:
        e: canonical metadata string, e.g. "requestId,<uuid>,timestamp,<ms>,user_id,<id>"
        t: latest user message text that feeds into the signature prompt (may be empty)
        s: timestamp in milliseconds
    
    Returns:
        Dictionary with signature and timestamp
    """
    # r = Number(s) - convert to number (already a number in Python)
    r = s
    # i = s - timestamp as string
    i = str(s)
    
    # n = new TextEncoder
    # a = n.encode(t)
    a = t.encode('utf-8')
    
    # w = btoa(String.fromCharCode(...a))
    # This is equivalent to base64 encoding the UTF-8 bytes
    w = base64.b64encode(a).decode('ascii')
    
    # c = `${e}|${w}|${i}`
    c = f"{e}|{w}|{i}"
    
    # E = Math.floor(r / (5 * 60 * 1e3))
    E = r // (5 * 60 * 1000)
    
    # A = CryptoJS.HmacSHA256(`${E}`, "key-@@@@)))()((9))-xxxx&&&%%%%%")
    secret = "key-@@@@)))()((9))-xxxx&&&%%%%%"
    A = hmac.new(secret.encode('utf-8'), str(E).encode('utf-8'), hashlib.sha256).hexdigest()
    
    # k = CryptoJS.HmacSHA256(c, A).toString()
    k = hmac.new(A.encode('utf-8'), c.encode('utf-8'), hashlib.sha256).hexdigest()
    
    # return n.encode(c), { signature: k, timestamp: i }
    # Note: n.encode(c) is not used in the return value, so we ignore it
    return {
        "signature": k,
        "timestamp": i
    }
