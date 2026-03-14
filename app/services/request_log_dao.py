"""
请求日志数据访问层 (DAO) - SQLAlchemy 版
提供请求日志的 CRUD 操作和查询功能
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_db_session as global_async_session
from app.models.db_models import RequestLog
from app.utils.logger import logger




def _normalize_trend_window(window: Optional[str], days: Optional[int]) -> str:
    """统一趋势窗口参数，兼容旧版 `days` 调用。"""
    if window:
        normalized = str(window).strip().lower()
    elif days == 30:
        normalized = "30d"
    elif days == 1:
        normalized = "24h"
    else:
        normalized = "7d"

    if normalized in {"24h", "7d", "30d"}:
        return normalized
    if normalized == "1d":
        return "24h"
    return "7d"

class RequestLogDAO:
    def __init__(self, db_path: str = None, db_url: str = None):
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

    async def init_db(self):
        """初始化数据库表"""
        if self._engine:
            from app.models.db_models import Base
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

    async def add_log(
        self,
        provider: str,
        endpoint: str,
        source: str,
        protocol: str,
        client_name: str,
        auth_token: Optional[str],
        upstream_auth_token: Optional[str],
        model: str,
        status_code: int,
        success: bool,
        duration: float = 0.0,
        first_token_time: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        total_tokens: Optional[int] = None,
        error_message: str = None
    ) -> int:
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens

        log_item = RequestLog(
            provider=provider,
            endpoint=endpoint,
            source=source,
            protocol=protocol,
            client_name=client_name,
            auth_token=auth_token,
            upstream_auth_token=upstream_auth_token,
            model=model,
            status_code=status_code,
            success=success,
            duration=duration,
            first_token_time=first_token_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            total_tokens=total_tokens,
            error_message=error_message,
        )
        async with self.session_factory() as session:
            session.add(log_item)
            await session.commit()
            return log_item.id

    async def get_recent_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        provider: str = None,
        model: str = None,
        success: bool = None,
        source: str = None,
    ) -> List[Dict]:
        stmt = select(RequestLog.__table__)
        if provider:
            stmt = stmt.where(RequestLog.provider == provider)
        if model:
            stmt = stmt.where(RequestLog.model == model)
        if success is not None:
            stmt = stmt.where(RequestLog.success == success)
        if source:
            stmt = stmt.where(RequestLog.source == source)
        
        stmt = stmt.order_by(RequestLog.timestamp.desc(), RequestLog.id.desc())
        stmt = stmt.limit(limit).offset(max(0, offset))

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            return [dict(row._mapping) for row in result.all()]

    async def count_logs(
        self,
        provider: str = None,
        model: str = None,
        success: bool = None,
        source: str = None,
    ) -> int:
        stmt = select(func.count()).select_from(RequestLog)
        if provider:
            stmt = stmt.where(RequestLog.provider == provider)
        if model:
            stmt = stmt.where(RequestLog.model == model)
        if success is not None:
            stmt = stmt.where(RequestLog.success == success)
        if source:
            stmt = stmt.where(RequestLog.source == source)

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar() or 0

    async def get_logs_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        provider: str = None,
        model: str = None,
        limit: int = 10000,
    ) -> List[Dict]:
        stmt = select(RequestLog.__table__).where(
            RequestLog.timestamp.between(start_time, end_time)
        )
        if provider:
            stmt = stmt.where(RequestLog.provider == provider)
        if model:
            stmt = stmt.where(RequestLog.model == model)

        stmt = stmt.order_by(RequestLog.timestamp.desc(), RequestLog.id.desc()).limit(limit)

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            return [dict(row._mapping) for row in result.all()]

    async def get_provider_request_stats(self, provider: Optional[str] = None) -> Dict:
        # Use raw SQL explicitly because of complex aggregations
        from sqlalchemy import text
        query = """
            SELECT
                COUNT(*) as total_requests,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as failed_requests,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cache_creation_tokens) as cache_creation_tokens,
                SUM(cache_read_tokens) as cache_read_tokens,
                SUM(CASE WHEN cache_creation_tokens > 0 THEN 1 ELSE 0 END) as cache_creation_requests,
                SUM(CASE WHEN cache_read_tokens > 0 THEN 1 ELSE 0 END) as cache_hit_requests,
                AVG(duration) as avg_duration,
                AVG(CASE WHEN first_token_time > 0 THEN first_token_time ELSE NULL END) as avg_first_token_time
            FROM request_logs
        """
        params = {}
        if provider:
            query += " WHERE provider = :provider"
            params["provider"] = provider

        try:
            async with self.session_factory() as session:
                result = await session.execute(text(query), params)
                row = result.mappings().first()

            if not row or not row.get("total_requests"):
                return {
                    "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "cache_creation_tokens": 0, "cache_read_tokens": 0,
                    "cache_creation_requests": 0, "cache_hit_requests": 0,
                    "avg_duration": 0.0, "avg_first_token_time": 0.0,
                }
            
            return {
                "total_requests": int(row["total_requests"] or 0),
                "successful_requests": int(row["successful_requests"] or 0),
                "failed_requests": int(row["failed_requests"] or 0),
                "input_tokens": int(row["input_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
                "cache_creation_tokens": int(row["cache_creation_tokens"] or 0),
                "cache_read_tokens": int(row["cache_read_tokens"] or 0),
                "cache_creation_requests": int(row["cache_creation_requests"] or 0),
                "cache_hit_requests": int(row["cache_hit_requests"] or 0),
                "avg_duration": float(row["avg_duration"] or 0.0),
                "avg_first_token_time": float(row["avg_first_token_time"] or 0.0),
            }
        except Exception as e:
            logger.error(f"❌ 获取请求统计失败: {e}")
            return {
                "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "cache_creation_requests": 0, "cache_hit_requests": 0,
                "avg_duration": 0.0, "avg_first_token_time": 0.0,
            }

    async def get_provider_usage_trend(
        self,
        provider: Optional[str] = None,
        days: Optional[int] = None,
        *,
        window: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> List[Dict]:
        trend_window = _normalize_trend_window(window, days)
        current_time = now or datetime.utcnow()
        async with self.session_factory() as session:
            is_pg = self._is_postgres(session)
            
            if trend_window == "24h":
                bucket_count = 24
                current_hour = current_time.replace(minute=0, second=0, microsecond=0)
                start_time = current_hour - timedelta(hours=bucket_count - 1)
                bucket_expression = self._hourly_bucket_expr(is_pg)
                
                row_key = "trend_bucket"
                label_format = "%H:%M"
                tooltip_format = "%Y-%m-%d %H:00"
                rows = await self._query_usage_trend_rows(session, provider, start_time, bucket_expression, row_key)
                
                rows_by_bucket = {str(row[row_key]): dict(row) for row in rows}
                trend = []
                for offset in range(bucket_count):
                    bucket_time = start_time + timedelta(hours=offset)
                    bucket_key = bucket_time.strftime("%Y-%m-%d %H:00:00")
                    trend.append(self._build_usage_trend_point(
                        row=rows_by_bucket.get(bucket_key, {}),
                        bucket=bucket_key,
                        label=bucket_time.strftime(label_format),
                        tooltip_label=bucket_time.strftime(tooltip_format),
                    ))
                return trend

            bucket_count = 30 if trend_window == "30d" else 7
            start_date = current_time.date() - timedelta(days=bucket_count - 1)
            start_time = datetime.combine(start_date, datetime.min.time())
            
            bucket_expression = "DATE(timestamp)"
            
            rows = await self._query_usage_trend_rows(session, provider, start_time, bucket_expression, "trend_bucket")
            rows_by_bucket = {str(row["trend_bucket"])[:10]: dict(row) for row in rows}
            trend = []
            
            for offset in range(bucket_count):
                bucket_date = start_date + timedelta(days=offset)
                bucket_key = bucket_date.isoformat()
                trend.append(self._build_usage_trend_point(
                    row=rows_by_bucket.get(bucket_key, {}),
                    bucket=bucket_key,
                    label=bucket_date.strftime("%m-%d"),
                    tooltip_label=bucket_date.strftime("%Y-%m-%d"),
                ))
            return trend

    @staticmethod
    def _is_postgres(session) -> bool:
        return session.bind.dialect.name == "postgresql" if session.bind else False

    @staticmethod
    def _hourly_bucket_expr(is_postgres: bool) -> str:
        if is_postgres:
            return "to_char(timestamp, 'YYYY-MM-DD HH24:00:00')"
        return "strftime('%Y-%m-%d %H:00:00', timestamp)"

    async def _query_usage_trend_rows(self, session, provider: Optional[str], start_time: datetime, bucket_expression: str, bucket_alias: str):
        query = f"""
            SELECT
                {bucket_expression} as {bucket_alias},
                COUNT(*) as total_requests,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful_requests,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cache_creation_tokens) as cache_creation_tokens,
                SUM(cache_read_tokens) as cache_read_tokens
            FROM request_logs
            WHERE timestamp >= :start_time
        """
        params = {"start_time": start_time}
        if provider:
            query += " AND provider = :provider"
            params["provider"] = provider
            
        query += f" GROUP BY {bucket_expression} ORDER BY {bucket_alias} ASC"
        
        result = await session.execute(text(query), params)
        return result.mappings().all()

    def _build_usage_trend_point(self, *, row: Dict, bucket: str, label: str, tooltip_label: str) -> Dict:
        total_requests = int(row.get("total_requests") or 0)
        successful_requests = int(row.get("successful_requests") or 0)
        cache_creation_tokens = int(row.get("cache_creation_tokens") or 0)
        cache_read_tokens = int(row.get("cache_read_tokens") or 0)
        return {
            "bucket": bucket,
            "label": label,
            "tooltip_label": tooltip_label,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": max(0, total_requests - successful_requests),
            "input_tokens": int(row.get("input_tokens") or 0),
            "output_tokens": int(row.get("output_tokens") or 0),
            "total_tokens": int(row.get("total_tokens") or 0),
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_total_tokens": cache_creation_tokens + cache_read_tokens,
            "success_rate": round((successful_requests / total_requests * 100) if total_requests > 0 else 0, 1),
        }

    async def get_model_stats_from_db(self, hours: int = 24) -> Dict:
        start_time = datetime.utcnow() - timedelta(hours=hours)
        query = """
            SELECT
                model,
                COUNT(*) as total,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as failed,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                AVG(duration) as avg_duration,
                AVG(CASE WHEN first_token_time > 0 THEN first_token_time ELSE NULL END) as avg_first_token_time
            FROM request_logs
            WHERE timestamp >= :start_time
            GROUP BY model
            ORDER BY total DESC
        """
        async with self.session_factory() as session:
            result = await session.execute(text(query), {"start_time": start_time})
            rows = result.mappings().all()

            out = {}
            for row in rows:
                model = row['model']
                out[model] = {
                    'total': row['total'],
                    'success': row['success'],
                    'failed': row['failed'],
                    'input_tokens': row['input_tokens'] or 0,
                    'output_tokens': row['output_tokens'] or 0,
                    'total_tokens': row['total_tokens'] or 0,
                    'avg_duration': round(row['avg_duration'] or 0, 2),
                    'avg_first_token_time': round(row['avg_first_token_time'] or 0, 2),
                    'success_rate': round((row['success'] / row['total'] * 100) if row['total'] > 0 else 0, 1),
                }
            return out

    async def delete_old_logs(self, days: int = 30) -> int:
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        async with self.session_factory() as session:
            stmt = delete(RequestLog).where(RequestLog.timestamp < cutoff_time)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()
            self._engine = None

_request_log_dao: Optional[RequestLogDAO] = None

def get_request_log_dao() -> RequestLogDAO:
    global _request_log_dao
    if _request_log_dao is None:
        _request_log_dao = RequestLogDAO()
    return _request_log_dao

def init_request_log_dao():
    dao = get_request_log_dao()
    return dao
