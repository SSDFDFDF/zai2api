import logging
from pathlib import Path

from app.utils.logger import (
    CONSOLE_HANDLER_NAME,
    DEFAULT_LOG_DIR,
    FILE_HANDLER_NAME,
    LEGACY_STDERR_HANDLER_NAME,
    NOISY_LOGGER_LEVELS,
    PROJECT_ROOT,
    _configure_noisy_loggers,
    logger as app_logger,
    resolve_log_file_path,
    setup_logger,
)


def test_resolve_log_file_path_puts_bare_filename_under_default_log_dir():
    resolved = resolve_log_file_path("app.log")

    assert resolved == (PROJECT_ROOT / DEFAULT_LOG_DIR / "app.log").resolve()


def test_resolve_log_file_path_keeps_nested_relative_path():
    resolved = resolve_log_file_path("custom/runtime.log")

    assert resolved == (PROJECT_ROOT / "custom" / "runtime.log").resolve()


def test_resolve_log_file_path_keeps_absolute_path(tmp_path: Path):
    target = tmp_path / "runtime.log"

    assert resolve_log_file_path(str(target)) == target.resolve()


def test_configure_noisy_loggers_raises_http_noise_threshold():
    logger = logging.getLogger("httpcore.http11")
    previous_level = logger.level

    try:
        logger.setLevel(logging.DEBUG)
        _configure_noisy_loggers()

        assert logger.level == NOISY_LOGGER_LEVELS["httpcore.http11"]
    finally:
        logger.setLevel(previous_level)


def test_configure_noisy_loggers_raises_aiosqlite_noise_threshold():
    logger = logging.getLogger("aiosqlite")
    previous_level = logger.level

    try:
        logger.setLevel(logging.DEBUG)
        _configure_noisy_loggers()

        assert logger.level == NOISY_LOGGER_LEVELS["aiosqlite"]
    finally:
        logger.setLevel(previous_level)


def test_setup_logger_keeps_root_at_warning_and_adds_console_and_file_handlers(
    monkeypatch,
    tmp_path: Path,
):
    from app.core.config import settings

    original_log_path = settings.LOG_FILE_PATH
    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    previous_app_level = app_logger.level

    try:
        settings.LOG_FILE_PATH = str(tmp_path / "app.log")
        setup_logger(debug_mode=True)

        assert root_logger.level == logging.WARNING
        assert app_logger.level == logging.DEBUG
        handlers = {handler.get_name(): handler for handler in root_logger.handlers}
        handler_names = set(handlers)
        assert CONSOLE_HANDLER_NAME in handler_names
        assert FILE_HANDLER_NAME in handler_names
        assert LEGACY_STDERR_HANDLER_NAME not in handler_names
        assert handlers[CONSOLE_HANDLER_NAME].level == logging.DEBUG
        assert handlers[FILE_HANDLER_NAME].level == logging.DEBUG
    finally:
        settings.LOG_FILE_PATH = original_log_path
        root_logger.setLevel(previous_root_level)
        app_logger.setLevel(previous_app_level)


def test_setup_logger_keeps_console_and_file_outputs_consistent(capsys, tmp_path: Path):
    from app.core.config import settings

    original_log_path = settings.LOG_FILE_PATH

    try:
        settings.LOG_FILE_PATH = str(tmp_path / "app.log")
        setup_logger(debug_mode=True)

        app_logger.debug("same-debug")
        app_logger.warning("same-warning")

        for handler in logging.getLogger().handlers:
            handler.flush()

        output = capsys.readouterr().out
        log_text = (tmp_path / "app.log").read_text(encoding="utf-8")

        assert "same-debug" in output
        assert "same-warning" in output
        assert "same-debug" in log_text
        assert "same-warning" in log_text
    finally:
        settings.LOG_FILE_PATH = original_log_path
