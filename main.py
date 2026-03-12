#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from granian import Granian

from app.admin import api as admin_api
from app.admin import routes as admin_routes
from app.core import claude, openai
from app.core.config import settings
from app.utils.logger import setup_logger
from app.utils.reload_config import RELOAD_CONFIG

# Setup logger
logger = setup_logger(log_dir="logs", debug_mode=settings.DEBUG_LOGGING)


async def warmup_upstream_client():
    """可选预热上游适配器，提前初始化动态依赖。"""
    try:
        from app.utils.fe_version import get_latest_fe_version
        from app.core.openai import get_upstream_client
        await get_latest_fe_version()
        client = get_upstream_client()
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

    yield

    logger.info("🔄 应用正在关闭...")

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
        Granian(
            "main:app",
            interface="asgi",
            address="0.0.0.0",
            port=settings.LISTEN_PORT,
            reload=False,  # 生产环境关闭热重载
            workers=1,     # ✅ 已经安全开启多进程模式
            process_name=service_name,  # 设置进程名称
            **RELOAD_CONFIG,    # 热重载配置
        ).serve()
    except KeyboardInterrupt:
        logger.info("🛑 received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"❌ service startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()