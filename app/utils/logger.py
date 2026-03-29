#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from logging.handlers import RotatingFileHandler

from pathlib import Path
from typing import Any

LOGGER_NAME = "zai2api"
LEGACY_STDERR_HANDLER_NAME = "zai2api.stderr"
CONSOLE_HANDLER_NAME = "zai2api.console"
FILE_HANDLER_NAME = "zai2api.file"
MANAGED_HANDLER_NAMES = {
    LEGACY_STDERR_HANDLER_NAME,
    CONSOLE_HANDLER_NAME,
    FILE_HANDLER_NAME,
}
DEFAULT_LOG_DIR = "logs"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOISY_LOGGER_LEVELS = {
    "aiosqlite": logging.WARNING,
    "httpcore": logging.WARNING,
    "httpcore.connection": logging.WARNING,
    "httpcore.http11": logging.WARNING,
    "httpcore.http2": logging.WARNING,
    "httpcore.proxy": logging.WARNING,
    "httpx": logging.WARNING,
}

logger = logging.getLogger(LOGGER_NAME)


def _render_log_message(message: str, args: tuple[Any, ...]) -> str:
    if not args:
        return message

    formatter_args: Any = args
    if len(args) == 1 and isinstance(args[0], dict):
        formatter_args = args[0]

    try:
        return message % formatter_args
    except Exception:
        rendered_args = " ".join(str(arg) for arg in args)
        return f"{message} {rendered_args}".strip()


def log_exception(
    target_logger: logging.Logger,
    message: str,
    *args: Any,
    error: BaseException | None = None,
    append_error: bool = True,
    **kwargs: Any,
) -> None:
    """Log exceptions with traceback only in debug mode."""
    from app.core.config import settings

    current_error = error if error is not None else sys.exc_info()[1]
    if settings.DEBUG_LOGGING:
        if current_error is not None:
            kwargs["exc_info"] = (
                type(current_error),
                current_error,
                current_error.__traceback__,
            )
        else:
            kwargs["exc_info"] = True
        target_logger.error(message, *args, **kwargs)
        return

    if current_error is None or not append_error:
        target_logger.error(message, *args, **kwargs)
        return

    target_logger.error(
        "%s: %s",
        _render_log_message(message, args),
        str(current_error).strip() or repr(current_error),
        **kwargs,
    )


def resolve_log_file_path(
    configured_path: str | None,
    *,
    log_dir: str | None = DEFAULT_LOG_DIR,
) -> Path:
    """Resolve the configured log file path against the project root."""
    raw_path = (configured_path or "").strip() or "logs/app.log"
    log_path = Path(raw_path).expanduser()
    if log_path.is_absolute():
        return log_path.resolve()
    if log_dir and len(log_path.parts) == 1:
        return (PROJECT_ROOT / log_dir / log_path).resolve()
    return (PROJECT_ROOT / log_path).resolve()


def _build_formatter(debug_mode: bool) -> logging.Formatter:
    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        if debug_mode
        else "%(asctime)s | %(levelname)-8s | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S" if debug_mode else "%H:%M:%S"
    return logging.Formatter(fmt, datefmt=datefmt)


def _close_managed_handlers(target_logger: logging.Logger) -> None:
    for handler in list(target_logger.handlers):
        if handler.get_name() not in MANAGED_HANDLER_NAMES:
            continue
        target_logger.removeHandler(handler)
        handler.close()


def _configure_noisy_loggers() -> None:
    for name, level in NOISY_LOGGER_LEVELS.items():
        logging.getLogger(name).setLevel(level)


def setup_logger(
    log_dir: str | None = DEFAULT_LOG_DIR,
    debug_mode: bool = False,
) -> logging.Logger:
    """Configure a single root logging pipeline for the whole application."""
    from app.core.config import settings

    log_path = resolve_log_file_path(settings.LOG_FILE_PATH, log_dir=log_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    app_level = logging.DEBUG if debug_mode else logging.INFO
    formatter = _build_formatter(debug_mode)
    root_level = logging.WARNING

    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)
    _close_managed_handlers(root_logger)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.set_name(CONSOLE_HANDLER_NAME)
    console_handler.setLevel(app_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max(1024, settings.LOG_FILE_MAX_BYTES),
        backupCount=max(1, settings.LOG_FILE_BACKUP_COUNT),
        encoding="utf-8",
    )
    file_handler.set_name(FILE_HANDLER_NAME)
    file_handler.setLevel(app_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _configure_noisy_loggers()

    logger.handlers.clear()
    logger.setLevel(app_level)
    logger.propagate = True

    return logger
