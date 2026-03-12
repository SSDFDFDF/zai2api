#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from loguru import logger

# Global logger instance
app_logger = None


def setup_logger(log_dir, log_retention_days=7, log_rotation="1 day", debug_mode=False):
    """
    Create a logger instance

    Parameters:
        log_dir (str): 日志目录
        log_retention_days (int): 日志保留天数
        log_rotation (str): 日志轮转间隔
        debug_mode (bool): 是否开启调试模式
    """
    global app_logger

    # 移除所有现有的日志处理器（支持热重载）
    logger.remove()

    log_level = "DEBUG" if debug_mode else "INFO"

    console_format = (
        "{time:HH:mm:ss} | {level: <8} | {message}"
        if not debug_mode
        else "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} | {message}"
    )

    # 添加控制台输出（根据 debug_mode 设置级别）
    logger.add(sys.stderr, level=log_level, format=console_format, colorize=False)

    # 添加文件输出
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        log_file = log_path / "{time:YYYY-MM-DD}.log"
        file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"

        logger.add(
            str(log_file),
            level=log_level,
            format=file_format,
            rotation=log_rotation,
            retention=f"{log_retention_days} days",
            encoding="utf-8",
            compression="zip",
            enqueue=True,
            catch=True,
        )
    except (PermissionError, OSError) as e:
        # 如果无法创建日志目录或文件，降级为仅控制台输出
        logger.warning(f"无法创建日志文件 ({e})，将仅使用控制台输出")

    app_logger = logger

    return logger


def get_logger():
    """Get the logger instance"""
    global app_logger
    if app_logger is None:
        # 如果没有设置过logger，使用默认配置
        logger.remove()  # 移除所有现有处理器
        logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}", colorize=False)
        app_logger = logger
    return app_logger


if __name__ == "__main__":
    """Test the logger"""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            setup_logger(temp_dir, debug_mode=True)

            logger.debug("这是一条调试日志")
            logger.info("这是一条信息日志")
            logger.warning("这是一条警告日志")
            logger.error("这是一条错误日志")
            logger.critical("这是一条严重日志")

            try:
                1 / 0
            except ZeroDivisionError:
                logger.exception("发生了除零异常")

            print("✅ 日志测试完成")

            logger.remove()

        except Exception as e:
            print(f"❌ 日志测试失败: {e}")
            logger.remove()
            raise
