#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.admin import api as admin_api
from app.admin import routes as admin_routes
from app.core import openai
from app.core.config import settings
from app.utils.logger import DEFAULT_LOG_DIR, setup_logger
from app.utils.reload_config import get_uvicorn_reload_config
from app.core.error_handler import register_exception_handlers

# Setup logger
logger = setup_logger(log_dir=DEFAULT_LOG_DIR, debug_mode=settings.DEBUG_LOGGING)


def _stringify_bool_env(value: str | None) -> str:
    if value is None:
        return "<unset>"
    return value


async def warmup_upstream_client():
    """可选预热上游适配器，提前初始化动态依赖。"""
    try:
        from app.utils.fe_version import get_latest_fe_version
        from app.core.openai import get_upstream_client
        await get_latest_fe_version()
        client = get_upstream_client()
        # 优先从数据库缓存加载在线模型，缓存为空时从上游拉取一次
        loaded = await client.load_cached_models()
        if not loaded:
            logger.info("数据库中无在线模型缓存，首次从上游拉取...")
            try:
                await client.get_online_models()
            except Exception as exc:
                logger.warning(
                    "首次拉取在线模型失败，使用硬编码默认值",
                    exc_info=True,
                )
        logger.info(
            f"✅ 上游适配器已就绪，支持 {len(client.get_supported_models())} 个模型"
        )
    except Exception as exc:
        logger.warning("⚠️ 上游适配器预热失败", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化数据库表结构
    from app.database import init_db
    await init_db()

    from dotenv import dotenv_values

    from app.services.token_automation import (
        run_directory_import,
        start_token_automation_scheduler,
        stop_token_automation_scheduler,
    )
    from app.admin.config_manager import (
        ENV_PATH,
        apply_db_overrides,
        get_config_source_snapshot,
        load_db_overrides,
    )

    env_values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    env_anonymous_mode = _stringify_bool_env(os.getenv("ANONYMOUS_MODE"))
    db_overrides = await load_db_overrides()
    config_sources = get_config_source_snapshot(
        env_values=env_values,
        db_values=db_overrides,
        keys=(
            "ANONYMOUS_MODE",
            "TOKEN_AUTO_IMPORT_ENABLED",
            "TOKEN_AUTO_MAINTENANCE_ENABLED",
            "MODEL_AUTO_REFRESH_HOURS",
        ),
    )

    # 加载数据库配置覆盖
    await apply_db_overrides(settings)
    from app.utils.token_pool import normalize_token_load_balance_strategy

    settings.TOKEN_LOAD_BALANCE_STRATEGY = normalize_token_load_balance_strategy(
        settings.TOKEN_LOAD_BALANCE_STRATEGY
    )
    setup_logger(log_dir=DEFAULT_LOG_DIR, debug_mode=settings.DEBUG_LOGGING)
    logger.info(
        "[config.startup] ANONYMOUS_MODE env=%s db=%s effective=%s TOKEN_AUTO_IMPORT_ENABLED env=%s db=%s effective=%s TOKEN_AUTO_MAINTENANCE_ENABLED env=%s db=%s effective=%s MODEL_AUTO_REFRESH_HOURS env=%s db=%s effective=%s",
        env_anonymous_mode,
        config_sources.get("ANONYMOUS_MODE", {}).get("db", "<unset>"),
        settings.ANONYMOUS_MODE,
        config_sources.get("TOKEN_AUTO_IMPORT_ENABLED", {}).get("env", "<unset>"),
        config_sources.get("TOKEN_AUTO_IMPORT_ENABLED", {}).get("db", "<unset>"),
        settings.TOKEN_AUTO_IMPORT_ENABLED,
        config_sources.get("TOKEN_AUTO_MAINTENANCE_ENABLED", {}).get("env", "<unset>"),
        config_sources.get("TOKEN_AUTO_MAINTENANCE_ENABLED", {}).get("db", "<unset>"),
        settings.TOKEN_AUTO_MAINTENANCE_ENABLED,
        config_sources.get("MODEL_AUTO_REFRESH_HOURS", {}).get("env", "<unset>"),
        config_sources.get("MODEL_AUTO_REFRESH_HOURS", {}).get("db", "<unset>"),
        settings.MODEL_AUTO_REFRESH_HOURS,
    )

    if settings.TOKEN_AUTO_IMPORT_ENABLED and settings.TOKEN_AUTO_IMPORT_SOURCE_DIR.strip():
        try:
            await run_directory_import(
                settings.TOKEN_AUTO_IMPORT_SOURCE_DIR,
                provider="zai",
            )
            logger.info("✅ 启动阶段已完成一次目录自动导入")
        except Exception as exc:
            logger.warning("⚠️ 启动阶段目录自动导入失败", exc_info=True)

    # 从数据库初始化认证 token 池
    from app.utils.token_pool import initialize_token_pool_from_db
    token_pool = await initialize_token_pool_from_db(
        provider="zai",
        failure_threshold=settings.TOKEN_FAILURE_THRESHOLD,
        recovery_timeout=settings.TOKEN_RECOVERY_TIMEOUT,
        strategy=settings.TOKEN_LOAD_BALANCE_STRATEGY,
    )

    if not token_pool and not settings.ANONYMOUS_MODE:
        logger.warning("⚠️ 未找到可用 Token 且未启用匿名模式，服务可能无法正常工作")

    if settings.ANONYMOUS_MODE:
        from app.utils.guest_session_pool import initialize_guest_session_pool

        logger.info(
            "[guest_session.startup] enabling guest pool because ANONYMOUS_MODE effective=%s env=%s db=%s pool_size=%s maintenance_interval=%s",
            settings.ANONYMOUS_MODE,
            config_sources.get("ANONYMOUS_MODE", {}).get("env", env_anonymous_mode),
            config_sources.get("ANONYMOUS_MODE", {}).get("db", "<unset>"),
            settings.GUEST_POOL_SIZE,
            settings.GUEST_POOL_MAINTENANCE_INTERVAL,
        )
        guest_pool = await initialize_guest_session_pool(
            pool_size=settings.GUEST_POOL_SIZE,
            session_max_age=settings.GUEST_SESSION_MAX_AGE,
            maintenance_interval=settings.GUEST_POOL_MAINTENANCE_INTERVAL,
        )
        guest_status = guest_pool.get_pool_status()
        logger.info(
            "🫥 匿名会话池已就绪: "
            f"{guest_status.get('valid_sessions', 0)} 个可用会话"
        )
    else:
        logger.info(
            "[guest_session.startup] guest pool disabled because ANONYMOUS_MODE effective=%s env=%s db=%s",
            settings.ANONYMOUS_MODE,
            config_sources.get("ANONYMOUS_MODE", {}).get("env", env_anonymous_mode),
            config_sources.get("ANONYMOUS_MODE", {}).get("db", "<unset>"),
        )

    await warmup_upstream_client()

    if settings.CAPTCHA_ENABLED:
        from app.core.captcha_client import create_captcha_client

        create_captcha_client(
            service_url=settings.CAPTCHA_SERVICE_URL,
            timeout=settings.CAPTCHA_SERVICE_TIMEOUT,
        )
        logger.info("captcha client initialized")

    await start_token_automation_scheduler()

    # Start async log writer
    from app.services.log_writer import get_log_writer
    await get_log_writer().start()
    from app.services.token_stats_writer import get_token_stats_writer
    await get_token_stats_writer().start()

    # 可选：在线模型自动刷新后台任务
    _model_refresh_task = None
    if settings.MODEL_AUTO_REFRESH_HOURS > 0:
        async def _model_auto_refresh_loop():
            interval = settings.MODEL_AUTO_REFRESH_HOURS * 3600
            while True:
                await asyncio.sleep(interval)
                try:
                    from app.core.openai import get_upstream_client_if_ready
                    client = get_upstream_client_if_ready()
                    if client:
                        client._online_models_time = 0.0
                        await client.get_online_models()
                        logger.info("在线模型自动刷新完成")
                except Exception as exc:
                    logger.warning("在线模型自动刷新失败", exc_info=True)

        _model_refresh_task = asyncio.create_task(_model_auto_refresh_loop())
        logger.info(f"在线模型自动刷新已启用，间隔 {settings.MODEL_AUTO_REFRESH_HOURS} 小时")

    yield

    logger.info("🔄 应用正在关闭...")

    if _model_refresh_task and not _model_refresh_task.done():
        _model_refresh_task.cancel()
        try:
            await _model_refresh_task
        except asyncio.CancelledError:
            pass

    await stop_token_automation_scheduler()
    logger.info("🔄 正在停止 guest session pool...")
    if settings.ANONYMOUS_MODE:
        from app.utils.guest_session_pool import close_guest_session_pool

        await close_guest_session_pool()

    logger.info("🔄 正在停止 upstream client...")
    from app.core.openai import get_upstream_client_if_ready
    upstream_client = get_upstream_client_if_ready()
    if upstream_client:
        await upstream_client.close()

    if settings.CAPTCHA_ENABLED:
        from app.core.captcha_client import close_captcha_client

        await close_captcha_client()

    logger.info("🔄 正在停止 async log writer...")
    from app.services.log_writer import get_log_writer
    await get_log_writer().stop()
    logger.info("🔄 正在停止 async token stats writer...")
    from app.services.token_stats_writer import get_token_stats_writer
    await get_token_stats_writer().stop()

    logger.info("🔄 正在关闭数据库连接...")

    try:
        from app.database import close_db
        await close_db()
        logger.info("✅ 数据库连接已关闭")
    except Exception as e:
        logger.exception("❌ 关闭数据库连接时出错")


# Create FastAPI app with lifespan
# root_path is used for reverse proxy path prefix (e.g., /api or /path-prefix)
# Disable FastAPI's built-in schema and documentation endpoints by default.
app = FastAPI(
    lifespan=lifespan,
    root_path=settings.ROOT_PATH,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Register global exception handlers
register_exception_handlers(app)

cors_origins_str = os.getenv("CORS_ORIGINS", "")
cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Include OpenAI-compatible API router
app.include_router(openai.router)

# Include admin routers
app.include_router(admin_routes.router)
app.include_router(admin_api.router)


@app.options("/")
async def handle_options():
    """Handle OPTIONS requests"""
    return Response(status_code=200)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "running"}


def run_server():
    service_name = settings.SERVICE_NAME

    logger.info(f"🚀 starting {service_name} service...")
    logger.info(f"📡 listen address: 0.0.0.0:{settings.LISTEN_PORT}")
    logger.info(f"🔧 mode: debug {'enabled' if settings.DEBUG_LOGGING else 'disabled'}, anonymous {'enabled' if settings.ANONYMOUS_MODE else 'disabled'}")

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=settings.LISTEN_PORT,
            workers=1,
            loop="uvloop",
            http="httptools",
            log_level="warning",
            **get_uvicorn_reload_config(),
        )
    except KeyboardInterrupt:
        logger.info("🛑 received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception("❌ service startup failed")
        sys.exit(1)


if __name__ == "__main__":
    run_server()
