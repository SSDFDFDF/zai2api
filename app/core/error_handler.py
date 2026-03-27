#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""全局错误处理器。

提供统一的异常处理机制：
- AppError 及其子类 → 结构化错误响应
- RequestValidationError → 验证错误响应
- HTTPException → 保持原有行为
- Exception → 兜底处理

所有错误响应遵循 OpenAI API 格式。
"""

import time
from typing import Any, Callable, Dict

from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions import AppError
from app.utils.logger import logger


# -----------------------------------------------------------------------
# OpenAI 兼容错误响应格式
# -----------------------------------------------------------------------

def create_error_response(
    message: str,
    error_type: str,
    code: str,
    status_code: int = 500,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """创建 OpenAI 兼容的错误响应体。"""
    error: Dict[str, Any] = {
        "message": message,
        "type": error_type,
        "code": code,
    }
    if details:
        error["details"] = details
    return {"error": error}


# -----------------------------------------------------------------------
# 异常处理器
# -----------------------------------------------------------------------

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """处理 AppError 及其子类。

    记录错误日志并返回结构化的错误响应。
    """
    # 根据状态码选择日志级别
    if exc.status_code >= 500:
        log_func = logger.error
    elif exc.status_code >= 400:
        log_func = logger.warning
    else:
        log_func = logger.info

    log_func(
        "API error: %s (code=%s, status=%d, path=%s)",
        exc.message,
        exc.code,
        exc.status_code,
        request.url.path,
        extra={
            "error_code": exc.code,
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
            "details": exc.details,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """处理 FastAPI 请求验证错误。"""
    errors = exc.errors()

    # 提取第一个错误作为主要消息
    if errors:
        first_error = errors[0]
        field = ".".join(str(loc) for loc in first_error.get("loc", []))
        message = first_error.get("msg", "Validation failed")
        if field:
            message = f"{field}: {message}"
    else:
        message = "Request validation failed"

    logger.warning(
        "Validation error: %s (path=%s)",
        message,
        request.url.path,
        extra={
            "path": request.url.path,
            "method": request.method,
            "validation_errors": errors,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            message=message,
            error_type="ValidationError",
            code="validation_error",
            status_code=422,
            details={"errors": errors},
        ),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """处理 FastAPI HTTPException。

    将 HTTPException 转换为 OpenAI 兼容格式。
    """
    # 特殊处理重定向
    if exc.status_code in (301, 302, 303, 307, 308):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )

    logger.warning(
        "HTTP exception: %s (status=%d, path=%s)",
        exc.detail,
        exc.status_code,
        request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            message=str(exc.detail),
            error_type="HTTPException",
            code=f"http_{exc.status_code}",
            status_code=exc.status_code,
        ),
        headers=exc.headers,
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底处理未知异常。

    记录完整堆栈，返回通用错误响应。
    """
    logger.exception(
        "Unhandled exception: %s (path=%s, method=%s)",
        type(exc).__name__,
        request.url.path,
        request.method,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            message="Internal server error",
            error_type="InternalServerError",
            code="internal_error",
        ),
    )


# -----------------------------------------------------------------------
# 流式响应错误处理
# -----------------------------------------------------------------------

def format_sse_error(
    message: str,
    code: str = "internal_error",
    error_type: str = "InternalError",
) -> str:
    """格式化 SSE 错误响应。

    用于流式响应中的错误传递。

    Args:
        message: 错误消息
        code: 错误码
        error_type: 错误类型

    Returns:
        SSE 格式的错误字符串
    """
    import json
    error_data = {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }
    return f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


def format_sse_done() -> str:
    """返回 SSE 结束标记。"""
    return "data: [DONE]\n\n"


# -----------------------------------------------------------------------
# 注册函数
# -----------------------------------------------------------------------

def register_exception_handlers(app) -> None:
    """注册所有异常处理器到 FastAPI 应用。

    Args:
        app: FastAPI 应用实例
    """
    from fastapi.exceptions import HTTPException, RequestValidationError

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    logger.info("✅ Global exception handlers registered")