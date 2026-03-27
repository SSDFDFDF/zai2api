#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HTTP 客户端异常类。

用于处理 HTTP 请求过程中的错误，如网络问题、SSL、代理等。
"""

from typing import Any, Dict, Optional

from app.exceptions.base import AppError


class HttpError(AppError):
    """HTTP 客户端错误基类。

    当 HTTP 请求失败时抛出。

    Attributes:
        url: 请求的 URL
        method: HTTP 方法

    Example:
        >>> raise HttpError("Request failed", url="https://api.example.com", method="POST")
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        method: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        final_details = details or {}
        if url:
            final_details["url"] = url
        if method:
            final_details["method"] = method
        super().__init__(
            message=message,
            code="http_error",
            status_code=502,
            details=final_details,
        )


class HttpConnectionError(HttpError):
    """连接失败异常。

    当无法建立 TCP 连接时抛出。

    Example:
        >>> raise HttpConnectionError("https://api.example.com", "Connection refused")
    """

    def __init__(
        self,
        url: str,
        reason: str,
    ) -> None:
        super().__init__(
            message=f"Connection failed: {reason}",
            url=url,
        )
        self.code = "connection_error"
        self.status_code = 503


class SSLError(HttpError):
    """SSL/TLS 错误异常。

    当 SSL 握手失败或证书验证失败时抛出。

    Example:
        >>> raise SSLError("https://api.example.com", "Certificate verify failed")
    """

    def __init__(
        self,
        url: str,
        reason: str,
    ) -> None:
        super().__init__(
            message=f"SSL/TLS error: {reason}",
            url=url,
        )
        self.code = "ssl_error"


class ProxyError(HttpError):
    """代理错误异常。

    当代理连接失败或代理返回错误时抛出。

    Example:
        >>> raise ProxyError("https://api.example.com", "Proxy connection refused")
    """

    def __init__(
        self,
        url: str,
        reason: str,
    ) -> None:
        super().__init__(
            message=f"Proxy error: {reason}",
            url=url,
        )
        self.code = "proxy_error"


class HttpTimeoutError(HttpError):
    """请求超时异常。

    当请求超时时抛出。

    Example:
        >>> raise HttpTimeoutError("https://api.example.com", "POST", timeout=30.0)
    """

    def __init__(
        self,
        url: str,
        method: str,
        timeout: Optional[float] = None,
    ) -> None:
        message = f"Request timeout"
        if timeout is not None:
            message += f" after {timeout}s"
        details = {}
        if timeout is not None:
            details["timeout_seconds"] = timeout
        super().__init__(
            message=message,
            url=url,
            method=method,
            details=details,
        )
        self.code = "timeout_error"
        self.status_code = 504