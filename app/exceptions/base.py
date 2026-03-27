#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""基础异常类。

所有应用异常的基类，提供：
- 标准化的错误消息
- 错误码 (code)
- HTTP 状态码 (status_code)
- 结构化详情 (details)
- OpenAI 兼容的 to_dict() 方法
"""

from typing import Any, Dict, Optional


class AppError(Exception):
    """应用基础异常类。

    所有业务异常应继承此类，确保统一的错误处理。

    Attributes:
        message: 人类可读的错误消息
        code: 错误码，用于程序识别
        status_code: HTTP 状态码
        details: 额外的错误详情

    Example:
        >>> raise AppError("Something went wrong", code="internal_error", status_code=500)
        >>> raise ValidationError("Invalid email", field="email")
    """

    def __init__(
        self,
        message: str,
        code: str = "internal_error",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """转换为 OpenAI 兼容的错误响应格式。

        Returns:
            符合 OpenAI API 规范的错误字典
        """
        error: Dict[str, Any] = {
            "message": self.message,
            "type": self.__class__.__name__,
            "code": self.code,
        }
        if self.details:
            error["details"] = self.details
        return {"error": error}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code!r}, status_code={self.status_code})"


class ErrorContext:
    """错误上下文管理器，用于自动捕获和转换异常。

    Example:
        >>> async with ErrorContext("Processing request", request_id="123"):
        ...     await process_request()
    """

    def __init__(
        self,
        operation: str,
        **context: Any,
    ) -> None:
        self.operation = operation
        self.context = context

    def __enter__(self) -> "ErrorContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_val is not None and not isinstance(exc_val, AppError):
            # 将非 AppError 异常包装为 AppError
            raise AppError(
                message=f"{self.operation} failed: {exc_val}",
                code="internal_error",
                status_code=500,
                details=self.context,
            ) from exc_val
        return False