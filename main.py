#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.admin import api as admin_api
from app.admin import routes as admin_routes
from app.core import claude, openai
from app.core.config import settings
from app.utils.logger import setup_logger
from app.utils.reload_config import get_uvicorn_reload_config

# Setup logger
logger = setup_logger(log_dir="logs", debug_mode=settings.DEBUG_LOGGING)


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
                logger.warning(f"首次拉取在线模型失败，使用硬编码默认值: {exc}")
        logger.info(f"✅ 上游适配器已就绪，支持 {len(client.get_supported_models())} 个模型")
    except Exception as exc:
        logger.warning(f"⚠️ 上游适配器预热失败: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化数据库表结构
    from app.database import init_db
    await init_db()

    from app.services.token_automation import (
        run_directory_import,
        start_token_automation_scheduler,
        stop_token_automation_scheduler,
    )
    from app.admin.config_manager import apply_db_overrides

    # 加载数据库配置覆盖
    await apply_db_overrides(settings)

    if settings.TOKEN_AUTO_IMPORT_ENABLED and settings.TOKEN_AUTO_IMPORT_SOURCE_DIR.strip():
        try:
            await run_directory_import(
                settings.TOKEN_AUTO_IMPORT_SOURCE_DIR,
                provider="zai",
            )
            logger.info("✅ 启动阶段已完成一次目录自动导入")
        except Exception as exc:
            logger.warning(f"⚠️ 启动阶段目录自动导入失败: {exc}")

    # 从数据库初始化认证 token 池
    from app.utils.token_pool import initialize_token_pool_from_db
    token_pool = await initialize_token_pool_from_db(
        provider="zai",
        failure_threshold=settings.TOKEN_FAILURE_THRESHOLD,
        recovery_timeout=settings.TOKEN_RECOVERY_TIMEOUT
    )

    if not token_pool and not settings.ANONYMOUS_MODE:
        logger.warning("⚠️ 未找到可用 Token 且未启用匿名模式，服务可能无法正常工作")

    if settings.ANONYMOUS_MODE:
        from app.utils.guest_session_pool import initialize_guest_session_pool

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

    await warmup_upstream_client()
    await start_token_automation_scheduler()

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
                    logger.warning(f"在线模型自动刷新失败: {exc}")

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
        session_manager = getattr(upstream_client, "session_manager", None)
        if session_manager is None:
            session_manager = getattr(upstream_client, "_session_manager", None)
        if session_manager is not None and hasattr(session_manager, "close"):
            await session_manager.close()
        await upstream_client.close()

    logger.info("🔄 正在关闭数据库连接...")

    try:
        from app.database import close_db
        await close_db()
        logger.info("✅ 数据库连接已关闭")
    except Exception as e:
        logger.error(f"❌ 关闭数据库连接时出错: {e}")


# Create FastAPI app with lifespan
# root_path is used for reverse proxy path prefix (e.g., /api or /path-prefix)
app = FastAPI(lifespan=lifespan, root_path=settings.ROOT_PATH)

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

# 挂载web端静态文件目录
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except RuntimeError:
    # 如果 static 目录不存在，创建它
    os.makedirs("app/static/css", exist_ok=True)
    os.makedirs("app/static/js", exist_ok=True)
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include API routers
app.include_router(openai.router)
app.include_router(claude.router)

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
    return {"message": "OpenAI Compatible API Server"}


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
        logger.error(f"❌ service startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()
