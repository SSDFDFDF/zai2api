import os
import ssl as _stdlib_ssl
from typing import AsyncGenerator
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings


def _clean_db_url(url: str) -> tuple[str, dict]:
    """Strip unsupported parameters from a PostgreSQL URL and return (cleaned_url, connect_args).

    - asyncpg does not accept `sslmode` as a query-string parameter (that is a
      libpq / psycopg2 convention). We translate it into asyncpg's `ssl`
      connect-arg instead.
    - asyncpg does not accept `channel_binding`. We strip it.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    sslmode = qs.pop("sslmode", [None])[0]
    # Remove channel_binding to prevent TypeError in asyncpg connect()
    qs.pop("channel_binding", None)

    # Rebuild the URL without the stripped parameters
    new_query = urlencode({k: v[0] for k, v in qs.items()})
    cleaned_url = urlunparse(parsed._replace(query=new_query))

    connect_args: dict = {}
    if sslmode:
        if sslmode == "disable":
            connect_args["ssl"] = False
        elif sslmode in ("require", "prefer"):
            # asyncpg accepts an ssl.SSLContext or the string "require" (>= 0.29)
            # Use a permissive SSLContext that skips certificate verification,
            # which matches the behavior of libpq's `sslmode=require`.
            ctx = _stdlib_ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = _stdlib_ssl.CERT_NONE
            connect_args["ssl"] = ctx
        elif sslmode in ("verify-ca", "verify-full"):
            connect_args["ssl"] = True  # uses default SSL context with verification

    return cleaned_url, connect_args


def get_db_url(db_url: str = None) -> tuple[str, dict]:
    """获取数据库连接 URL 和 connect_args"""
    url = db_url or settings.DATABASE_URL
    connect_args: dict = {}

    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        # 如果已经是 postgresql+asyncpg 或者是 sqlite+aiosqlite，则直接使用
        elif url.startswith("sqlite"):
            if not url.startswith("sqlite+aiosqlite"):
                url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)

        # Strip unsupported parameters for asyncpg compatibility
        if "asyncpg" in url:
            url, connect_args = _clean_db_url(url)

        return url, connect_args
    
    # 默认使用 SQLite
    db_path = settings.DB_PATH
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(db_path)
    return f"sqlite+aiosqlite:///{db_path}", {}


# 全局引擎和会话工厂
_db_url, _connect_args = get_db_url()
engine = create_async_engine(_db_url, echo=False, pool_pre_ping=True, connect_args=_connect_args)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db():
    """初始化数据库表"""
    from app.utils.logger import logger
    
    async with engine.begin() as conn:
        # 自动迁移检查（使用 Alembic）
        try:
            from alembic.config import Config
            from alembic import command
            import os
            
            logger.info("🔧 自动数据库迁移: 准备执行 Alembic...")
            
            # 获取项目根目录
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alembic_cfg = Config(os.path.join(base_dir, "alembic.ini"))
            alembic_cfg.set_main_option("script_location", os.path.join(base_dir, "migrations"))
            
            def run_upgrade(connection):
                logger.info("  -> run_upgrade: setting connection...")
                alembic_cfg.attributes['connection'] = connection
                logger.info("  -> run_upgrade: calling command.upgrade...")
                command.upgrade(alembic_cfg, "head")
                logger.info("  -> run_upgrade: command.upgrade finished")
                
            logger.info("🔧 自动数据库迁移: 正在 run_sync(run_upgrade)...")
            await conn.run_sync(run_upgrade)
            logger.info("✅ 迁移成功")
        except Exception as e:
            logger.error(f"⚠️ 自动迁移过程中出错: {e}")

async def close_db():
    """关闭数据库连接"""
    await engine.dispose()
