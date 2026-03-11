"""
Token 数据访问层 (DAO) - SQLAlchemy 版
提供 Token 的 CRUD 操作和查询功能
"""
import os
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import async_session as global_async_session
from app.models.db_models import Token, TokenStats
from app.utils.logger import logger


class TokenDAO:
    """Token 数据访问对象"""

    def __init__(self, db_path: str = None, db_url: str = None):
        """初始化 DAO"""
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

    async def init_database(self):
        """初始化数据库表结构"""
        if self._engine:
            from app.models.db_models import Base
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        logger.debug("✅ Token 数据库初始化调用完成 (SQLAlchemy)")

    # ==================== Token CRUD 操作 ====================

    async def add_token(
        self,
        provider: str,
        token: str,
        token_type: str = "user",
        priority: int = 0,
        validate: bool = True
    ) -> Optional[int]:
        try:
            # 对于 zai 提供商，强制验证 Token
            if provider == "zai" and validate:
                from app.utils.token_pool import ZAITokenValidator

                validated_type, is_valid, error_msg = await ZAITokenValidator.validate_token(token)

                # 拒绝 guest token
                if validated_type == "guest":
                    logger.warning(f"🚫 拒绝添加匿名用户 Token: {token[:20]}... - {error_msg}")
                    return None

                # 拒绝无效 token
                if not is_valid:
                    logger.warning(f"🚫 Token 验证失败: {token[:20]}... - {error_msg}")
                    return None

                # 使用验证后的类型
                token_type = validated_type

            async with self.session_factory() as session:
                # 检查是否存在
                stmt = select(Token).filter_by(provider=provider, token=token)
                existing = (await session.execute(stmt)).scalar_one_or_none()
                if existing:
                    logger.warning(f"⚠️ Token 已存在: {provider} - {token[:20]}...")
                    return None

                new_token = Token(
                    provider=provider,
                    token=token,
                    token_type=token_type,
                    priority=priority,
                )
                session.add(new_token)
                await session.flush()  # 获得 ID

                # 同时创建统计记录
                new_stats = TokenStats(token_id=new_token.id)
                session.add(new_stats)
                await session.commit()
                
                logger.info(f"✅ 添加 Token: {provider} ({token_type}) - {token[:20]}...")
                return new_token.id
        except Exception as e:
            logger.error(f"❌ 添加 Token 失败: {e}")
            return None

    def _format_token_row(self, token_item: Token) -> Dict:
        """Helper to format Token with related stats"""
        d = {
            "id": token_item.id,
            "provider": token_item.provider,
            "token": token_item.token,
            "token_type": token_item.token_type,
            "priority": token_item.priority,
            "is_enabled": 1 if token_item.is_enabled else 0,
            "created_at": str(token_item.created_at) if token_item.created_at else None,
            "total_requests": getattr(token_item.stats, "total_requests", 0) if token_item.stats else 0,
            "successful_requests": getattr(token_item.stats, "successful_requests", 0) if token_item.stats else 0,
            "failed_requests": getattr(token_item.stats, "failed_requests", 0) if token_item.stats else 0,
            "last_success_time": str(token_item.stats.last_success_time) if token_item.stats and token_item.stats.last_success_time else None,
            "last_failure_time": str(token_item.stats.last_failure_time) if token_item.stats and token_item.stats.last_failure_time else None,
        }
        return d

    async def get_tokens_by_provider(
        self,
        provider: str,
        enabled_only: bool = True,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict]:
        try:
            from sqlalchemy.orm import selectinload
            async with self.session_factory() as session:
                stmt = select(Token).options(selectinload(Token.stats)).filter_by(provider=provider)
                if enabled_only:
                    stmt = stmt.filter_by(is_enabled=True)
                
                stmt = stmt.order_by(Token.priority.desc(), Token.id.asc())
                if limit is not None:
                    stmt = stmt.limit(limit).offset(max(0, offset))
                    
                result = await session.execute(stmt)
                return [self._format_token_row(t) for t in result.scalars()]
        except Exception as e:
            logger.error(f"❌ 查询 Token 失败: {e}")
            return []

    async def get_all_tokens(self, enabled_only: bool = False) -> List[Dict]:
        try:
            from sqlalchemy.orm import selectinload
            async with self.session_factory() as session:
                stmt = select(Token).options(selectinload(Token.stats))
                if enabled_only:
                    stmt = stmt.filter_by(is_enabled=True)
                
                stmt = stmt.order_by(Token.provider, Token.priority.desc(), Token.id.asc())
                result = await session.execute(stmt)
                return [self._format_token_row(t) for t in result.scalars()]
        except Exception as e:
            logger.error(f"❌ 查询所有 Token 失败: {e}")
            return []

    async def update_token_status(self, token_id: int, is_enabled: bool):
        try:
            async with self.session_factory() as session:
                stmt = update(Token).where(Token.id == token_id).values(is_enabled=is_enabled)
                await session.execute(stmt)
                await session.commit()
                logger.info(f"✅ 更新 Token 状态: id={token_id}, enabled={is_enabled}")
        except Exception as e:
            logger.error(f"❌ 更新 Token 状态失败: {e}")

    async def update_token_type(self, token_id: int, token_type: str):
        try:
            async with self.session_factory() as session:
                stmt = update(Token).where(Token.id == token_id).values(token_type=token_type)
                await session.execute(stmt)
                await session.commit()
                logger.info(f"✅ 更新 Token 类型: id={token_id}, type={token_type}")
        except Exception as e:
            logger.error(f"❌ 更新 Token 类型失败: {e}")

    async def delete_token(self, token_id: int):
        try:
            async with self.session_factory() as session:
                stmt = delete(Token).where(Token.id == token_id)
                await session.execute(stmt)
                await session.commit()
                logger.info(f"✅ 删除 Token: id={token_id}")
        except Exception as e:
            logger.error(f"❌ 删除 Token 失败: {e}")

    async def delete_tokens_by_ids(self, token_ids: List[int]) -> int:
        if not token_ids:
            return 0
        try:
            async with self.session_factory() as session:
                stmt = delete(Token).where(Token.id.in_(token_ids))
                result = await session.execute(stmt)
                await session.commit()
                deleted_count = result.rowcount
                logger.info(f"✅ 批量删除 Token: {deleted_count} 个")
                return deleted_count
        except Exception as e:
            logger.error(f"❌ 批量删除 Token 失败: {e}")
            return 0

    async def delete_tokens_by_provider(self, provider: str):
        try:
            async with self.session_factory() as session:
                stmt = delete(Token).where(Token.provider == provider)
                await session.execute(stmt)
                await session.commit()
                logger.info(f"✅ 删除提供商所有 Token: {provider}")
        except Exception as e:
            logger.error(f"❌ 删除提供商 Token 失败: {e}")

    # ==================== Token 统计操作 ====================

    async def record_success(self, token_id: int):
        try:
            async with self.session_factory() as session:
                stmt = update(TokenStats).where(TokenStats.token_id == token_id).values(
                    total_requests=TokenStats.total_requests + 1,
                    successful_requests=TokenStats.successful_requests + 1,
                    last_success_time=func.now()
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.error(f"❌ 记录成功失败: {e}")

    async def record_failure(self, token_id: int):
        try:
            async with self.session_factory() as session:
                stmt = update(TokenStats).where(TokenStats.token_id == token_id).values(
                    total_requests=TokenStats.total_requests + 1,
                    failed_requests=TokenStats.failed_requests + 1,
                    last_failure_time=func.now()
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.error(f"❌ 记录失败失败: {e}")

    async def get_token_stats(self, token_id: int) -> Optional[Dict]:
        try:
            async with self.session_factory() as session:
                stmt = select(TokenStats).filter_by(token_id=token_id)
                result = await session.execute(stmt)
                stats = result.scalar_one_or_none()
                if not stats:
                    return None
                return {
                    "id": stats.id,
                    "token_id": stats.token_id,
                    "total_requests": stats.total_requests,
                    "successful_requests": stats.successful_requests,
                    "failed_requests": stats.failed_requests,
                    "last_success_time": str(stats.last_success_time) if stats.last_success_time else None,
                    "last_failure_time": str(stats.last_failure_time) if stats.last_failure_time else None,
                }
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return None

    # ==================== 批量操作 ====================

    async def bulk_add_tokens(
        self,
        provider: str,
        tokens: List[str],
        token_type: str = "user",
        validate: bool = True
    ) -> Tuple[int, int]:
        added_count = 0
        failed_count = 0

        for token in tokens:
            if token.strip():
                # SQLAlchemy 没有方便的 bulk insert + override 并支持验证的方法，
                # 所以我们还是复用 add_token 提供验证和统计记录的同时插入。
                token_id = await self.add_token(
                    provider,
                    token.strip(),
                    token_type,
                    validate=validate
                )
                if token_id:
                    added_count += 1
                else:
                    failed_count += 1

        logger.info(f"✅ 批量添加完成: {provider} - 成功 {added_count}/{len(tokens)}，失败 {failed_count}")
        return added_count, failed_count

    async def replace_tokens(self, provider: str, tokens: List[str],
                            token_type: str = "user"):
        await self.delete_tokens_by_provider(provider)
        added_count, _ = await self.bulk_add_tokens(provider, tokens, token_type)
        logger.info(f"✅ 替换 Token 完成: {provider} - {added_count} 个")
        return added_count

    async def remove_duplicate_tokens(self, provider: Optional[str] = None) -> int:
        try:
            tokens = (
                await self.get_tokens_by_provider(provider, enabled_only=False)
                if provider
                else await self.get_all_tokens(enabled_only=False)
            )

            seen_keys: set[tuple[str, str]] = set()
            duplicate_ids: list[int] = []

            for token_record in tokens:
                token_value = str(token_record.get("token") or "").strip()
                token_provider = str(token_record.get("provider") or "")
                key = (token_provider, token_value)

                if key in seen_keys:
                    duplicate_ids.append(int(token_record["id"]))
                    continue

                seen_keys.add(key)

            deleted_count = await self.delete_tokens_by_ids(duplicate_ids)
            if deleted_count > 0:
                logger.info(f"✅ 已清理重复 Token: {deleted_count} 个")
            return deleted_count
        except Exception as e:
            logger.error(f"❌ 清理重复 Token 失败: {e}")
            return 0

    # ==================== 实用方法 ====================

    async def get_token_by_id(self, token_id: int) -> Optional[Dict]:
        try:
            from sqlalchemy.orm import selectinload
            async with self.session_factory() as session:
                stmt = select(Token).options(selectinload(Token.stats)).filter_by(id=token_id)
                result = await session.execute(stmt)
                token_item = result.scalar_one_or_none()
                if not token_item:
                    return None
                return self._format_token_row(token_item)
        except Exception as e:
            logger.error(f"❌ 查询 Token 失败: {e}")
            return None

    async def get_token_by_value(self, provider: str, token: str) -> Optional[Dict]:
        try:
            from sqlalchemy.orm import selectinload
            async with self.session_factory() as session:
                stmt = select(Token).options(selectinload(Token.stats)).filter_by(provider=provider, token=token)
                result = await session.execute(stmt)
                token_item = result.scalar_one_or_none()
                if not token_item:
                    return None
                return self._format_token_row(token_item)
        except Exception as e:
            logger.error(f"❌ 查询 Token 失败: {e}")
            return None

    async def get_provider_stats(self, provider: str) -> Dict:
        try:
            async with self.session_factory() as session:
                from sqlalchemy import text
                query = text("""
                    SELECT
                        COUNT(*) as total_tokens,
                        SUM(CASE WHEN is_enabled = true THEN 1 ELSE 0 END) as enabled_tokens,
                        SUM(ts.total_requests) as total_requests,
                        SUM(ts.successful_requests) as successful_requests,
                        SUM(ts.failed_requests) as failed_requests
                    FROM tokens t
                    LEFT JOIN token_stats ts ON t.id = ts.token_id
                    WHERE t.provider = :provider
                """)
                result = await session.execute(query, {"provider": provider})
                row = result.mappings().first()
                if not row or not row.get("total_tokens"):
                    return {}
                return dict(row)
        except Exception as e:
            logger.error(f"❌ 获取提供商统计失败: {e}")
            return {}

    async def get_provider_token_counts(self, provider: str) -> Dict[str, int]:
        try:
            async with self.session_factory() as session:
                from sqlalchemy import text
                query = text("""
                    SELECT
                        COUNT(*) as total_tokens,
                        SUM(CASE WHEN is_enabled = true THEN 1 ELSE 0 END) as enabled_tokens,
                        SUM(CASE WHEN token_type = 'user' THEN 1 ELSE 0 END) as user_tokens,
                        SUM(CASE WHEN token_type = 'guest' THEN 1 ELSE 0 END) as guest_tokens,
                        SUM(CASE WHEN token_type = 'unknown' THEN 1 ELSE 0 END) as unknown_tokens
                    FROM tokens
                    WHERE provider = :provider
                """)
                result = await session.execute(query, {"provider": provider})
                row = result.mappings().first()

                if not row or not row.get('total_tokens'):
                    return {
                        "total_tokens": 0, "enabled_tokens": 0,
                        "user_tokens": 0, "guest_tokens": 0, "unknown_tokens": 0,
                    }

                return {
                    "total_tokens": int(row["total_tokens"] or 0),
                    "enabled_tokens": int(row["enabled_tokens"] or 0),
                    "user_tokens": int(row["user_tokens"] or 0),
                    "guest_tokens": int(row["guest_tokens"] or 0),
                    "unknown_tokens": int(row["unknown_tokens"] or 0),
                }
        except Exception as e:
            logger.error(f"❌ 获取 Token 数量统计失败: {e}")
            return {
                "total_tokens": 0, "enabled_tokens": 0,
                "user_tokens": 0, "guest_tokens": 0, "unknown_tokens": 0,
            }

    async def count_tokens_by_provider(
        self,
        provider: str,
        enabled_only: bool = False,
    ) -> int:
        try:
            async with self.session_factory() as session:
                stmt = select(func.count(Token.id)).filter_by(provider=provider)
                if enabled_only:
                    stmt = stmt.filter_by(is_enabled=True)
                result = await session.execute(stmt)
                return result.scalar() or 0
        except Exception as e:
            logger.error(f"❌ 统计 Token 总数失败: {e}")
            return 0

    # ==================== Token 验证操作 ====================
    async def validate_and_update_token(self, token_id: int) -> bool:
        try:
            token_info = await self.get_token_by_id(token_id)
            if not token_info:
                logger.error(f"❌ Token ID {token_id} 不存在")
                return False

            provider = token_info["provider"]
            token = token_info["token"]

            if provider != "zai":
                logger.info(f"⏭️ 跳过非 zai 提供商的 Token 验证: {provider}")
                return True

            from app.utils.token_pool import ZAITokenValidator

            token_type, is_valid, error_msg = await ZAITokenValidator.validate_token(token)
            await self.update_token_type(token_id, token_type)

            if not is_valid:
                logger.warning(f"⚠️ Token 验证失败: id={token_id}, type={token_type}, error={error_msg}")

            return is_valid

        except Exception as e:
            logger.error(f"❌ 验证 Token 失败: {e}")
            return False

    async def validate_tokens_detailed(self, provider: str = "zai") -> Dict[str, Any]:
        try:
            tokens = await self.get_tokens_by_provider(provider, enabled_only=False)

            if not tokens:
                logger.warning(f"⚠️ 没有需要验证的 {provider} Token")
                return {
                    "checked": 0, "valid": 0, "guest": 0, "invalid": 0, "invalid_token_ids": [],
                }

            logger.info(f"🔍 开始批量验证 {len(tokens)} 个 {provider} Token...")

            from app.utils.token_pool import ZAITokenValidator

            stats: Dict[str, Any] = {
                "checked": len(tokens), "valid": 0, "guest": 0, "invalid": 0, "invalid_token_ids": [],
            }

            for token_record in tokens:
                token_id = int(token_record["id"])
                token = str(token_record["token"])

                token_type, is_valid, error_msg = await ZAITokenValidator.validate_token(token)
                await self.update_token_type(token_id, token_type)

                if token_type == "user" and is_valid:
                    stats["valid"] += 1
                elif token_type == "guest":
                    stats["guest"] += 1
                    stats["invalid_token_ids"].append(token_id)
                else:
                    stats["invalid"] += 1
                    stats["invalid_token_ids"].append(token_id)
                    if error_msg:
                        logger.warning("⚠️ Token 验证失败: id={}, type={}, error={}", token_id, token_type, error_msg)

            logger.info("✅ 批量验证完成: 有效 {}, 匿名 {}, 无效 {}", stats["valid"], stats["guest"], stats["invalid"])
            return stats

        except Exception as e:
            logger.error(f"❌ 批量验证失败: {e}")
            return {
                "checked": 0, "valid": 0, "guest": 0, "invalid": 0, "invalid_token_ids": [],
            }

    async def validate_all_tokens(self, provider: str = "zai") -> Dict[str, int]:
        stats = await self.validate_tokens_detailed(provider)
        return {
            "valid": int(stats.get("valid", 0) or 0),
            "guest": int(stats.get("guest", 0) or 0),
            "invalid": int(stats.get("invalid", 0) or 0),
        }

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None


# 全局单例
_token_dao: Optional[TokenDAO] = None

def get_token_dao() -> TokenDAO:
    """获取全局 TokenDAO 实例"""
    global _token_dao
    if _token_dao is None:
        _token_dao = TokenDAO()
    return _token_dao

async def init_token_database():
    """初始化 Token 数据库"""
    dao = get_token_dao()
    await dao.init_database()
