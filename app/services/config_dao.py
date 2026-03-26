"""
配置项数据访问层 (DAO) - SQLAlchemy 版
提供配置项的 CRUD 操作
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_db_session as global_async_session
from app.models.db_models import ConfigItem
from app.utils.logger import logger

_config_dao: Optional["ConfigDAO"] = None


class ConfigDAO:
    """配置项数据访问对象 - SQLAlchemy 版"""

    def __init__(self, db_path: str = None, db_url: str = None):
        """
        初始化 DAO。
        提供 db_path 或 db_url 可以用于测试隔离，默认使用全局 session 工厂。
        """
        if db_url:
            self._engine = create_async_engine(db_url, echo=False)
            self.session_factory = async_sessionmaker(self._engine, expire_on_commit=False, class_=AsyncSession)
        elif db_path:
            url = f"sqlite+aiosqlite:///{db_path}"
            self._engine = create_async_engine(url, echo=False)
            self.session_factory = async_sessionmaker(self._engine, expire_on_commit=False, class_=AsyncSession)
        else:
            self._engine = None
            self.session_factory = global_async_session

    async def init_database(self) -> None:
        """初始化配置表（为保持接口兼容性保留，实际建表由全局统一处理）"""
        if self._engine:
            from app.models.db_models import Base
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        logger.debug("✅ 配置表初始化调用完成 (SQLAlchemy)")

    async def get(self, key: str) -> Optional[str]:
        """读取单个配置值"""
        async with self.session_factory() as session:
            stmt = select(ConfigItem).filter(ConfigItem.key == key)
            result = await session.execute(stmt)
            item = result.scalar_one_or_none()
            return item.value if item else None

    async def get_all(self) -> Dict[str, str]:
        """读取全部配置"""
        async with self.session_factory() as session:
            stmt = select(ConfigItem)
            result = await session.execute(stmt)
            return {item.key: item.value for item in result.scalars()}

    async def set(self, key: str, value: str) -> None:
        """写入/更新单条配置（UPSERT）"""
        async with self.session_factory() as session:
            stmt = select(ConfigItem).filter(ConfigItem.key == key)
            result = await session.execute(stmt)
            item = result.scalar_one_or_none()
            if item:
                item.value = value
            else:
                session.add(ConfigItem(key=key, value=value))
            await session.commit()

    async def set_many(self, items: Dict[str, str]) -> None:
        """批量写入/更新配置"""
        if not items:
            return
        async with self.session_factory() as session:
            keys = list(items.keys())
            stmt = select(ConfigItem).filter(ConfigItem.key.in_(keys))
            result = await session.execute(stmt)
            existing_items = {item.key: item for item in result.scalars()}
            
            for key, value in items.items():
                if key in existing_items:
                    existing_items[key].value = value
                else:
                    session.add(ConfigItem(key=key, value=value))
            await session.commit()

    async def delete(self, key: str) -> None:
        """删除单条配置"""
        async with self.session_factory() as session:
            stmt = delete(ConfigItem).where(ConfigItem.key == key)
            await session.execute(stmt)
            await session.commit()

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None


def get_config_dao() -> ConfigDAO:
    """获取全局 ConfigDAO 实例"""
    global _config_dao
    if _config_dao is None:
        _config_dao = ConfigDAO()
    return _config_dao


async def init_config_database() -> None:
    """初始化配置数据库"""
    dao = get_config_dao()
    await dao.init_database()
