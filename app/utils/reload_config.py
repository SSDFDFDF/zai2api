#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
热重载配置模块
定义 Uvicorn 服务器热重载时需要监视和忽略的目录
"""

import os

# 监视的路径（只监视应用相关代码）
RELOAD_DIRS = [
    "app",
    "main.py",
]

# 忽略的目录/文件模式（glob 格式，uvicorn 使用 watchfiles）
RELOAD_EXCLUDES = [
    "logs",
    "storage",
    "__pycache__",
    ".git",
    ".github",
    ".vscode",
    ".idea",
    "deploy",
    "node_modules",
    "migrations",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "htmlcov",
    "tests",
    "app/templates",
    "*.log",
    "*.sqlite3*",
    "*.db",
    "*.pyc",
    "*.pid",
]


def get_uvicorn_reload_config() -> dict:
    """根据环境变量 RELOAD 决定是否启用热重载，返回 uvicorn 配置字典。"""
    enable_reload = os.getenv("RELOAD", "").lower() in ("1", "true", "yes")
    if not enable_reload:
        return {"reload": False}

    return {
        "reload": True,
        "reload_dirs": RELOAD_DIRS,
        "reload_excludes": RELOAD_EXCLUDES,
    }
