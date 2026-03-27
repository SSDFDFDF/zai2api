#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""上游服务异常类。

用于处理上游 API（如 z.ai）返回的错误。
"""

from typing import Any, Dict, Optional

from app.exceptions.base import AppError


class UpstreamError(AppError):
    """上游服务错误基类。

    当上游 API 返回错误时抛出。

    Attributes:
        upstream_code: 上游返回的错误码
        upstream_status: 上游返回的 HTTP 状态码

    Example:
        >>> raise UpstreamError("Upstream API error", upstream_code=50001, upstream_status=500)
    """

    def __init__(
        self,
        message: str,
        upstream_code: Optional[int] = None,
        upstream_status: Optional[int] = None,
        upstream_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        final_details = details or {}
        if upstream_code is not None:
            final_details["upstream_code"] = upstream_code
        if upstream_status is not None:
            final_details["upstream_status"] = upstream_status
        if upstream_type is not None:
            final_details["upstream_type"] = upstream_type

        super().__init__(
            message=message,
            code="upstream_error",
            status_code=502,  # Bad Gateway
            details=final_details,
        )


class UpstreamRateLimited(UpstreamError):
    """上游限流异常。

    当上游 API 返回 429 状态码时抛出。

    Example:
        >>> raise UpstreamRateLimited(retry_after=30)
    """

    def __init__(
        self,
        message: str = "Upstream rate limited",
        retry_after: Optional[int] = None,
    ) -> None:
        details = {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(
            message=message,
            upstream_status=429,
            details=details,
        )
        self.code = "rate_limited"
        self.status_code = 429


class UpstreamTimeout(UpstreamError):
    """上游超时异常。

    当上游 API 响应超时时抛出。

    Example:
        >>> raise UpstreamTimeout(timeout=30.0)
    """

    def __init__(
        self,
        message: str = "Upstream request timeout",
        timeout: Optional[float] = None,
    ) -> None:
        details = {}
        if timeout is not None:
            details["timeout_seconds"] = timeout
        super().__init__(message=message, details=details)
        self.code = "upstream_timeout"
        self.status_code = 504  # Gateway Timeout


class UpstreamConnectionError(UpstreamError):
    """上游连接失败异常。

    当无法连接到上游 API 时抛出。

    Example:
        >>> raise UpstreamConnectionError("Connection refused")
    """

    def __init__(
        self,
        message: str = "Failed to connect to upstream",
        reason: Optional[str] = None,
    ) -> None:
        details = {}
        if reason:
            details["reason"] = reason
        super().__init__(message=message, details=details)
        self.code = "upstream_connection_error"
        self.status_code = 503  # Service Unavailable


class UpstreamAuthenticationError(UpstreamError):
    """上游认证失败异常。

    当上游 Token 失效或无效时抛出。

    Example:
        >>> raise UpstreamAuthenticationError("Token expired")
    """

    def __init__(
        self,
        message: str = "Upstream authentication failed",
        token_hint: Optional[str] = None,
    ) -> None:
        details = {}
        if token_hint:
            details["token_hint"] = token_hint
        super().__init__(
            message=message,
            upstream_status=401,
            details=details,
        )
        self.code = "upstream_auth_error"
        self.status_code = 401