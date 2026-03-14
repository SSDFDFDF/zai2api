#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    API_ENDPOINT: str = "https://chat.z.ai/api/v2/chat/completions"
    
    # Authentication
    AUTH_TOKEN: Optional[str] = os.getenv("AUTH_TOKEN")

    # Token池配置
    TOKEN_FAILURE_THRESHOLD: int = int(os.getenv("TOKEN_FAILURE_THRESHOLD", "3"))  # 失败3次后标记为不可用
    TOKEN_RECOVERY_TIMEOUT: int = int(os.getenv("TOKEN_RECOVERY_TIMEOUT", "1800"))  # 30分钟后重试失败的token
    TOKEN_AUTO_IMPORT_ENABLED: bool = (
        os.getenv("TOKEN_AUTO_IMPORT_ENABLED", "false").lower() == "true"
    )
    TOKEN_AUTO_IMPORT_SOURCE_DIR: str = os.getenv("TOKEN_AUTO_IMPORT_SOURCE_DIR", "")
    TOKEN_AUTO_IMPORT_INTERVAL: int = int(
        os.getenv("TOKEN_AUTO_IMPORT_INTERVAL", "300")
    )
    TOKEN_AUTO_MAINTENANCE_ENABLED: bool = (
        os.getenv("TOKEN_AUTO_MAINTENANCE_ENABLED", "false").lower() == "true"
    )
    TOKEN_AUTO_MAINTENANCE_INTERVAL: int = int(
        os.getenv("TOKEN_AUTO_MAINTENANCE_INTERVAL", "1800")
    )
    TOKEN_AUTO_REMOVE_DUPLICATES: bool = (
        os.getenv("TOKEN_AUTO_REMOVE_DUPLICATES", "true").lower() == "true"
    )
    TOKEN_AUTO_HEALTH_CHECK: bool = (
        os.getenv("TOKEN_AUTO_HEALTH_CHECK", "true").lower() == "true"
    )
    TOKEN_AUTO_DELETE_INVALID: bool = (
        os.getenv("TOKEN_AUTO_DELETE_INVALID", "false").lower() == "true"
    )

    # Chat Cleanup
    CHAT_CLEANUP_ENABLED: bool = (
        os.getenv("CHAT_CLEANUP_ENABLED", "true").lower() == "true"
    )
    CHAT_CLEANUP_INTERVAL_DAYS: int = int(os.getenv("CHAT_CLEANUP_INTERVAL_DAYS", "7"))

    # Request Log Cleanup
    LOG_CLEANUP_INTERVAL_DAYS: int = int(os.getenv("LOG_CLEANUP_INTERVAL_DAYS", "1"))
    LOG_RETENTION_DAYS: int = int(os.getenv("LOG_RETENTION_DAYS", "30"))

    # Model Configuration
    GLM45_MODEL: str = os.getenv("GLM45_MODEL", "GLM-4.5")
    GLM45_THINKING_MODEL: str = os.getenv("GLM45_THINKING_MODEL", "GLM-4.5-Thinking")
    GLM45_SEARCH_MODEL: str = os.getenv("GLM45_SEARCH_MODEL", "GLM-4.5-Search")
    GLM45_AIR_MODEL: str = os.getenv("GLM45_AIR_MODEL", "GLM-4.5-Air")
    GLM46V_MODEL: str = os.getenv("GLM46V_MODEL", "GLM-4.6V")
    GLM46V_ADVANCED_SEARCH_MODEL: str = os.getenv("GLM46V_ADVANCED_SEARCH_MODEL", "GLM-4.6V-advanced-search")
    GLM5_MODEL: str = os.getenv("GLM5_MODEL", "GLM-5")
    GLM5_THINKING_MODEL: str = os.getenv("GLM5_THINKING_MODEL", "GLM-5-Thinking")
    GLM5_AGENT_MODEL: str = os.getenv("GLM5_AGENT_MODEL", "GLM-5-Agent")
    GLM5_ADVANCED_SEARCH_MODEL: str = os.getenv("GLM5_ADVANCED_SEARCH_MODEL", "GLM-5-advanced-search")
    GLM47_MODEL: str = os.getenv("GLM47_MODEL", "GLM-4.7")
    GLM47_THINKING_MODEL: str = os.getenv("GLM47_THINKING_MODEL", "GLM-4.7-Thinking")
    GLM47_SEARCH_MODEL: str = os.getenv("GLM47_SEARCH_MODEL", "GLM-4.7-Search")
    GLM47_ADVANCED_SEARCH_MODEL: str = os.getenv("GLM47_ADVANCED_SEARCH_MODEL", "GLM-4.7-advanced-search")
    MODEL_BLACKLIST: str = os.getenv("MODEL_BLACKLIST", "glm-4-flash")
    # 模型别名映射，格式：别名=目标模型名，逗号分隔。例如：gpt-4o=GLM-5,gpt-4o-mini=GLM-4.7
    MODEL_ALIASES: str = os.getenv("MODEL_ALIASES", "")
    # 在线模型自动刷新周期（小时），0 表示禁用自动刷新
    MODEL_AUTO_REFRESH_HOURS: int = int(os.getenv("MODEL_AUTO_REFRESH_HOURS", "0"))

    # Server Configuration
    LISTEN_PORT: int = int(os.getenv("LISTEN_PORT", "8080"))
    DEBUG_LOGGING: bool = os.getenv("DEBUG_LOGGING", "false").lower() == "true"
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "api-proxy-server")
    ROOT_PATH: str = os.getenv("ROOT_PATH", "")  # For Nginx reverse proxy path prefix, e.g., "/api" or "/path-prefix"

    ANONYMOUS_MODE: bool = os.getenv("ANONYMOUS_MODE", "true").lower() == "true"
    GUEST_POOL_SIZE: int = int(os.getenv("GUEST_POOL_SIZE", "3"))
    GUEST_SESSION_MAX_AGE: int = int(os.getenv("GUEST_SESSION_MAX_AGE", "480"))
    GUEST_POOL_MAINTENANCE_INTERVAL: int = int(
        os.getenv("GUEST_POOL_MAINTENANCE_INTERVAL", "30")
    )
    GUEST_CLEANUP_PARALLELISM: int = int(
        os.getenv("GUEST_CLEANUP_PARALLELISM", "4")
    )
    GUEST_HTTP_MAX_KEEPALIVE_CONNECTIONS: int = int(
        os.getenv("GUEST_HTTP_MAX_KEEPALIVE_CONNECTIONS", "20")
    )
    GUEST_HTTP_MAX_CONNECTIONS: int = int(
        os.getenv("GUEST_HTTP_MAX_CONNECTIONS", "50")
    )
    TOOL_SUPPORT: bool = os.getenv("TOOL_SUPPORT", "true").lower() == "true"
    TOOL_RESULT_MAX_LENGTH: int = int(os.getenv("TOOL_RESULT_MAX_LENGTH", "0"))  # 0 = 不截断
    SCAN_LIMIT: int = int(os.getenv("SCAN_LIMIT", "200000"))
    GLM_INTERNAL_TOOL_HINT_ENABLED: bool = (
        os.getenv("GLM_INTERNAL_TOOL_HINT_ENABLED", "false").lower() == "true"
    )

    # File Upload
    # 上传文件大小限制（字节），默认 10MB，0 表示不限制
    MAX_UPLOAD_FILE_SIZE: int = int(os.getenv("MAX_UPLOAD_FILE_SIZE", str(10 * 1024 * 1024)))

    # Upstream User Variables
    UPSTREAM_USER_NAME: str = os.getenv("UPSTREAM_USER_NAME", "User")
    UPSTREAM_USER_LOCATION: str = os.getenv("UPSTREAM_USER_LOCATION", "Unknown")
    UPSTREAM_USER_TIMEZONE: str = os.getenv("UPSTREAM_USER_TIMEZONE", "Asia/Shanghai")
    UPSTREAM_USER_LANGUAGE: str = os.getenv("UPSTREAM_USER_LANGUAGE", "zh-CN")

    # Upstream Background Tasks (title/tag generation)
    # 代理场景下每次请求都是新 chat_id, 无需上游生成标题/标签
    UPSTREAM_BACKGROUND_TASKS: bool = (
        os.getenv("UPSTREAM_BACKGROUND_TASKS", "false").lower() == "true"
    )

    # Session / Continuous Conversation Configuration
    SESSION_ENABLED: bool = os.getenv("SESSION_ENABLED", "false").lower() == "true"
    SESSION_TTL: int = int(os.getenv("SESSION_TTL", "3600"))           # 会话 TTL（秒）
    SESSION_MAX_PER_CLIENT: int = int(os.getenv("SESSION_MAX_PER_CLIENT", "50"))  # 每客户端最大会话数
    SESSION_CLEANUP_INTERVAL: int = int(os.getenv("SESSION_CLEANUP_INTERVAL", "300"))  # 清理间隔（秒）
    SESSION_SYSTEM_INJECT: bool = os.getenv("SESSION_SYSTEM_INJECT", "true").lower() == "true"
    SKIP_AUTH_TOKEN: bool = os.getenv("SKIP_AUTH_TOKEN", "false").lower() == "true"

    # HTTP Timeout Configuration (单位: 秒)
    HTTP_CONNECT_TIMEOUT: float = float(os.getenv("HTTP_CONNECT_TIMEOUT", "5.0"))
    HTTP_WRITE_TIMEOUT: float = float(os.getenv("HTTP_WRITE_TIMEOUT", "10.0"))
    HTTP_POOL_TIMEOUT: float = float(os.getenv("HTTP_POOL_TIMEOUT", "5.0"))
    # 普通短请求 (鉴权/文件上传/模型列表) 的读取超时
    HTTP_DEFAULT_READ_TIMEOUT: float = float(os.getenv("HTTP_DEFAULT_READ_TIMEOUT", "60.0"))
    # 流式聊天的读取超时 (即相邻两个 chunk 之间允许的最大空闲时间)
    HTTP_STREAM_READ_TIMEOUT: float = float(os.getenv("HTTP_STREAM_READ_TIMEOUT", "120.0"))
    # 在线模型列表拉取超时
    HTTP_MODEL_FETCH_TIMEOUT: float = float(os.getenv("HTTP_MODEL_FETCH_TIMEOUT", "10.0"))
    # 非流式请求端到端总超时（含所有重试）
    CHAT_TOTAL_TIMEOUT: float = float(os.getenv("CHAT_TOTAL_TIMEOUT", "300.0"))
    # 流式请求端到端总超时（含所有重试）
    HTTP_STREAM_TOTAL_TIMEOUT: float = float(os.getenv("HTTP_STREAM_TOTAL_TIMEOUT", "600.0"))

    # Proxy Configuration (统一网络代理, 支持 HTTP/HTTPS/SOCKS5)
    HTTP_PROXY: Optional[str] = os.getenv("HTTP_PROXY")  # 统一代理, 默认使用本地 10808 端口

    # Admin Panel Authentication
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")  # 管理后台密码
    SESSION_SECRET_KEY: str = os.getenv("SESSION_SECRET_KEY", "your-secret-key-change-in-production")  # Session 密钥
    DB_PATH: str = os.getenv("DB_PATH", "tokens.db")
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"  # 忽略额外字段，防止环境变量中的未知字段导致验证错误


settings = Settings()
