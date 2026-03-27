#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""API 层异常类。

用于 API 端点层的错误，如认证、授权、验证等。
"""

from typing import Any, Dict, Optional

from app.exceptions.base import AppError


class AuthenticationError(AppError):
    """认证失败异常。

    当用户未提供认证信息或认证信息无效时抛出。

    Example:
        >>> raise AuthenticationError("Missing Authorization header")
        >>> raise AuthenticationError("Invalid API key")
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code="authentication_error",
            status_code=401,
            details=details,
        )


class AuthorizationError(AppError):
    """授权失败异常。

    当用户已认证但权限不足时抛出。

    Example:
        >>> raise AuthorizationError("Access denied to this resource")
    """

    def __init__(
        self,
        message: str = "Permission denied",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code="authorization_error",
            status_code=403,
            details=details,
        )


class ValidationError(AppError):
    """请求验证失败异常。

    当请求数据不符合预期格式或约束时抛出。

    Example:
        >>> raise ValidationError("Invalid email format", field="email")
        >>> raise ValidationError("messages cannot be empty")
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        final_details = details or {}
        if field:
            final_details["field"] = field
        super().__init__(
            message=message,
            code="validation_error",
            status_code=400,
            details=final_details,
        )


class ModelNotFoundError(AppError):
    """模型不存在异常。

    当请求的模型不在支持列表中时抛出。

    Example:
        >>> raise ModelNotFoundError("gpt-5")
    """

    def __init__(self, model: str) -> None:
        super().__init__(
            message=f"Model '{model}' not found",
            code="model_not_found",
            status_code=404,
            details={"model": model},
        )


class RateLimitError(AppError):
    """请求频率限制异常。

    当客户端请求过于频繁时抛出。

    Example:
        >>> raise RateLimitError(retry_after=60)
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
    ) -> None:
        details = {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(
            message=message,
            code="rate_limit_exceeded",
            status_code=429,
            details=details,
        )