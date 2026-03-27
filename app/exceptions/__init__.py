#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""统一异常包。

提供标准化的异常类层次结构，支持：
- 统一的错误码和 HTTP 状态码映射
- 结构化的错误详情
- OpenAI 兼容的错误响应格式
"""

from app.exceptions.base import AppError
from app.exceptions.api import (
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    ModelNotFoundError,
    RateLimitError,
)
from app.exceptions.upstream import (
    UpstreamError,
    UpstreamRateLimited,
    UpstreamTimeout,
    UpstreamConnectionError,
)
from app.exceptions.http import (
    HttpError,
    HttpConnectionError,
    SSLError,
    ProxyError,
    HttpTimeoutError,
)

__all__ = [
    # Base
    "AppError",
    # API errors
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "ModelNotFoundError",
    "RateLimitError",
    # Upstream errors
    "UpstreamError",
    "UpstreamRateLimited",
    "UpstreamTimeout",
    "UpstreamConnectionError",
    # HTTP errors
    "HttpError",
    "HttpConnectionError",
    "SSLError",
    "ProxyError",
    "HttpTimeoutError",
]