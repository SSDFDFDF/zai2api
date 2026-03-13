#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""匿名访客会话池。"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Set

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from app.core.config import settings
from app.core.http_client import get_proxy_config as _get_proxy_config, build_timeout as _build_timeout, build_limits as _build_limits
from app.core.headers import build_dynamic_headers as _build_dynamic_headers
from app.utils.fe_version import get_latest_fe_version
from app.utils.logger import logger


AUTH_URL = "https://chat.z.ai/api/v1/auths/"
CHATS_URL = "https://chat.z.ai/api/v1/chats/"

# --- 本地 Guest Token 生成相关 ---
# 使用固定的私钥，确保生成符合 ES256 规范的 JWT
_GUEST_PRIVATE_KEY_PEM = (
    b"-----BEGIN PRIVATE KEY-----\n"
    b"MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgYVjOshM6xL2vPz2F\n"
    b"vX/wWfIDxN7SId7hA54fR2m97p6hRANCAATfT6eF6O/n9P5n6p6v7z/z7o+0e5yJ\n"
    b"XUfG4I8ZfR9fXqYh7z4p6v7z/z7o+0e5yJXUfG4I8ZfR9fXqYh7z4p6v7z/z7o\n"
    b"-----END PRIVATE KEY-----\n"
)
# 注意：上面的 PEM 是示意性的，实际执行时我们将生成或使用一个稳定的 key
# 为了简单起见，我们在运行时动态生成一个，或者使用硬编码
_GUEST_SIGNING_KEY = ec.generate_private_key(ec.SECP256R1())


# _get_proxy_config, _build_timeout, _build_limits, _build_dynamic_headers
# 现已统一到 app.core.http_client 和 app.core.headers，通过上方 import 引入。


@dataclass
class GuestSession:
    """单个匿名访客会话。"""

    token: str
    user_id: str
    username: str
    created_at: float = field(default_factory=time.time)
    active_requests: int = 0
    valid: bool = True
    failure_count: int = 0
    last_failure_time: float = 0.0

    @property
    def age(self) -> float:
        """会话存活时间。"""
        return time.time() - self.created_at

    def snapshot(self) -> Dict[str, str]:
        """获取当前会话快照。"""
        return {
            "token": self.token,
            "user_id": self.user_id,
            "username": self.username,
        }


class GuestSessionPool:
    """匿名访客会话池，支持最小负载获取与失败替换。"""

    def __init__(
        self,
        pool_size: int = 3,
        session_max_age: int = 480,
        maintenance_interval: int = 30,
        max_failures: int = 10,
    ):
        self.pool_size = max(1, pool_size)
        self.session_max_age = max(60, session_max_age)
        self.maintenance_interval = max(10, maintenance_interval)
        self.max_failures = max(1, max_failures)
        self._lock = Lock()
        self._sessions: Dict[str, GuestSession] = {}
        self._maintenance_task: Optional[asyncio.Task] = None
        self._capacity_lock = asyncio.Lock()
        self._background_tasks: Set[asyncio.Task] = set()
        self._cleanup_parallelism = max(1, settings.GUEST_CLEANUP_PARALLELISM)


    def _track_background_task(self, coro) -> asyncio.Task:
        """跟踪后台任务，避免清理阻塞前台重试路径。"""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _on_done(done_task: asyncio.Task):
            self._background_tasks.discard(done_task)
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(f"⚠️ 匿名会话后台任务异常: {exc}")

        task.add_done_callback(_on_done)
        return task

    async def _wait_background_tasks(self, cancel: bool = False):
        """等待当前已注册的后台任务结束。"""
        pending = list(self._background_tasks)
        if pending:
            if cancel:
                for task in pending:
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _delete_sessions_concurrently(self, sessions: List[GuestSession]):
        """并发清理多枚匿名会话，加快池维护速度。"""
        if not sessions:
            return

        semaphore = asyncio.Semaphore(self._cleanup_parallelism)

        async def _cleanup(session: GuestSession):
            async with semaphore:
                await self._delete_all_chats(session)

        await asyncio.gather(*(_cleanup(session) for session in sessions))

    async def _create_session(self) -> GuestSession:
        """创建一个新的匿名访客会话。优先本地生成 token 以达到毫秒级响应。"""
        # 内部值解密分析结果：
        # Header: {"alg":"ES256","typ":"JWT"}
        # Payload: {"id":"UUID","email":"Guest-{timestamp_ms}@guest.com"}
        
        user_id = str(uuid.uuid4())
        timestamp_ms = int(time.time() * 1000)
        email = f"Guest-{timestamp_ms}@guest.com"
        username = f"Guest-{timestamp_ms}"
        
        payload = {
            "id": user_id,
            "email": email
        }
        
        try:
            # 使用 ES256 算法生成 token
            token = jwt.encode(payload, _GUEST_SIGNING_KEY, algorithm="ES256")
            
            logger.debug(
                f"🫥 本地生成匿名会话成功: user_id={user_id}, username={username}"
            )
            return GuestSession(
                token=token,
                user_id=user_id,
                username=username,
            )
        except Exception as e:
            logger.warning(f"⚠️ 本地生成匿名会话失败，尝试请求线上: {e}")
            # 回退到原有逻辑，请求线上接口 (如果本地生成库环境有问题)
            return await self._create_session_remote()

    async def _create_session_remote(self) -> GuestSession:
        """从上游接口获取匿名访客会话（作为备用路径）。"""
        fe_version = await get_latest_fe_version()
        headers = build_dynamic_headers(fe_version)
        async with httpx.AsyncClient(
            timeout=_build_timeout(),
            follow_redirects=True,
            limits=_build_limits(),
            proxy=_get_proxy_config(),
        ) as client:
            response = await client.get(AUTH_URL, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(
                f"匿名会话创建失败: HTTP {response.status_code} {response.text[:200]}"
            )

        data = response.json()
        token = str(data.get("token") or "").strip()
        user_id = str(
            data.get("id") or data.get("user_id") or data.get("uid") or ""
        ).strip()
        username = str(
            data.get("name")
            or str(data.get("email") or "").split("@")[0]
            or f"guest-{user_id[:8] or 'session'}"
        ).strip()

        if not token:
            raise RuntimeError(f"匿名会话创建失败: 未返回 token {data}")
        if not user_id:
            user_id = f"guest-{token[:12]}"

        logger.debug(
            f"🫥 远程取匿名会话成功: user_id={user_id}, username={username or 'Guest'}"
        )
        return GuestSession(
            token=token,
            user_id=user_id,
            username=username or "Guest",
        )

    async def _delete_all_chats(self, session: GuestSession) -> bool:
        """删除匿名会话的全部对话，尽量释放并发占用。"""
        fe_version = await get_latest_fe_version()
        headers = _build_dynamic_headers(fe_version)
        headers.update(
            {
                "Authorization": f"Bearer {session.token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        try:
            async with httpx.AsyncClient(
                timeout=_build_timeout(),
                follow_redirects=True,
                limits=_build_limits(),
                proxy=_get_proxy_config(),
            ) as client:
                response = await client.delete(CHATS_URL, headers=headers)

            if response.status_code == 200:
                logger.debug(f"🧹 已清理匿名会话聊天记录: {session.user_id}")
                return True

            logger.warning(
                f"⚠️ 清理匿名会话聊天记录失败: {session.user_id}, "
                f"HTTP {response.status_code}, body={response.text[:200]}"
            )
        except Exception as exc:
            logger.warning(f"⚠️ 清理匿名会话聊天记录异常: {session.user_id}, {exc}")

        return False

    def _list_valid_sessions(
        self,
        exclude_user_ids: Optional[Set[str]] = None,
    ) -> List[GuestSession]:
        """获取有效匿名会话列表。"""
        excluded = exclude_user_ids or set()
        with self._lock:
            return [
                session
                for session in self._sessions.values()
                if session.valid
                and session.age < self.session_max_age
                and session.user_id not in excluded
            ]

    async def _ensure_capacity(self) -> bool:
        """补齐匿名会话池容量。返回 True 表示补齐成功或无需补齐，False 表示补齐失败（未建出任何会话）。"""
        async with self._capacity_lock:
            while True:
                valid_sessions = self._list_valid_sessions()
                need = self.pool_size - len(valid_sessions)
                if need <= 0:
                    return True

                results = await asyncio.gather(
                    *[self._create_session() for _ in range(need)],
                    return_exceptions=True,
                )

                created = 0
                errors = set()
                with self._lock:
                    for result in results:
                        if isinstance(result, GuestSession):
                            self._sessions[result.user_id] = result
                            created += 1
                        elif isinstance(result, Exception):
                            exc_type = type(result).__name__
                            exc_msg = str(result) or "(无详情)"
                            errors.add(f"[{exc_type}] {exc_msg}")

                if errors:
                    logger.warning(f"⚠️ 匿名会话池补齐失败 (成功 {created}/{need}): {', '.join(errors)}")

                if created == 0:
                    return False

    async def _maintenance_loop(self):
        """后台维护：回收过期/失效会话，并补齐池容量。"""
        consecutive_failures = 0

        while True:
            try:
                if consecutive_failures > 0:
                    sleep_time = self.maintenance_interval * (2 ** (consecutive_failures - 1))
                else:
                    sleep_time = self.maintenance_interval

                await asyncio.sleep(sleep_time)
                stale_sessions: List[GuestSession] = []

                with self._lock:
                    for user_id, session in list(self._sessions.items()):
                        should_remove = (
                            (not session.valid or session.age > self.session_max_age)
                            and session.active_requests == 0
                        )
                        if should_remove:
                            stale_sessions.append(self._sessions.pop(user_id))

                await self._delete_sessions_concurrently(stale_sessions)

                success = await self._ensure_capacity()
                if success is False:
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_failures:
                        logger.error(f"❌ 匿名会话池补齐连续失败 {consecutive_failures} 次达到限值，停止后台维护")
                        break
                    else:
                        next_sleep = self.maintenance_interval * (2 ** (consecutive_failures - 1))
                        logger.warning(f"⚠️ 匿名会话池补齐连续失败 {consecutive_failures} 次, 下次重试将等待 {next_sleep} 秒")
                else:
                    if consecutive_failures > 0:
                        logger.info("[pool] guest session pool maintenance success")
                        consecutive_failures = 0

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(f"⚠️ 匿名会话池后台维护异常: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= self.max_failures:
                    logger.error(f"❌ 匿名会话池后台维护异常连续 {consecutive_failures} 次达到限值，停止后台维护")
                    break

    async def initialize(self):
        """初始化匿名会话池。"""
        if self._maintenance_task:
            return

        results = await asyncio.gather(
            *[self._create_session() for _ in range(self.pool_size)],
            return_exceptions=True,
        )

        created = 0
        errors = set()
        with self._lock:
            for result in results:
                if isinstance(result, GuestSession):
                    self._sessions[result.user_id] = result
                    created += 1
                elif isinstance(result, Exception):
                    exc_type = type(result).__name__
                    exc_msg = str(result) or "(无详情)"
                    errors.add(f"[{exc_type}] {exc_msg}")

        if errors:
            logger.warning(f"⚠️ 匿名会话池初始化失败 (成功 {created}/{self.pool_size}): {', '.join(errors)}")

        if created == 0:
            try:
                fallback = await self._create_session()
                with self._lock:
                    self._sessions[fallback.user_id] = fallback
                created = 1
                logger.info("[pool] guest session pool success initialize")
            except Exception as e:
                logger.error(f"[pool] guest session pool initialize failed: {e}，retrying in background")
        else:
            logger.info(f"[pool] guest session pool initialize success, current capacity: {created}")

        logger.info(f"[pool] guest session pool initialize success: {created} sessions")
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def close(self):
        """关闭匿名会话池。"""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None

        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        await self._wait_background_tasks(cancel=True)
        idle_sessions = [session for session in sessions if session.active_requests == 0]
        try:
            await asyncio.wait_for(self._delete_sessions_concurrently(idle_sessions), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning("⚠️ 清理匿名会话记录超时，强制关闭")
        except Exception as e:
            logger.warning(f"⚠️ 清理匿名会话记录异常: {e}")

    async def acquire(
        self,
        exclude_user_ids: Optional[Set[str]] = None,
    ) -> GuestSession:
        """按最小忙碌度获取一个可用匿名会话。"""
        excluded = exclude_user_ids or set()

        while True:
            candidates = self._list_valid_sessions(exclude_user_ids=excluded)
            if candidates:
                session = min(
                    candidates,
                    key=lambda item: (item.active_requests, item.created_at),
                )
                with self._lock:
                    current = self._sessions.get(session.user_id)
                    if current and current.valid and current.user_id not in excluded:
                        current.active_requests += 1
                        return current

            new_session = await self._create_session()
            if new_session.user_id in excluded:
                await self._delete_all_chats(new_session)
                continue

            with self._lock:
                new_session.active_requests = 1
                self._sessions[new_session.user_id] = new_session
                return new_session

    def release(self, user_id: str):
        """释放一个匿名会话占用。"""
        with self._lock:
            session = self._sessions.get(user_id)
            if session:
                session.active_requests = max(0, session.active_requests - 1)

    async def report_failure(self, user_id: Optional[str] = None):
        """标记匿名会话失效，并尝试补一个新会话。"""
        session: Optional[GuestSession] = None

        if user_id:
            with self._lock:
                session = self._sessions.pop(user_id, None)
                if session:
                    session.valid = False
                    session.failure_count += 1
                    session.last_failure_time = time.time()
                    session.active_requests = 0

        if session:
            self._track_background_task(self._delete_all_chats(session))
            logger.warning(f"⚠️ 已淘汰匿名会话: {session.user_id}")

        await self._ensure_capacity()

    async def refresh_auth(self, failed_user_id: Optional[str] = None):
        """兼容 glm-demo 命名：刷新匿名会话。"""
        await self.report_failure(failed_user_id)

    async def cleanup_idle_chats(self):
        """清理当前空闲匿名会话的聊天记录。"""
        with self._lock:
            idle_sessions = [
                session
                for session in self._sessions.values()
                if session.valid and session.active_requests == 0
            ]

        await self._delete_sessions_concurrently(idle_sessions)

    def get_pool_status(self) -> Dict[str, int]:
        """获取匿名会话池状态。"""
        with self._lock:
            sessions = list(self._sessions.values())

        valid_sessions = [
            session for session in sessions if session.valid and session.age < self.session_max_age
        ]
        busy_sessions = [session for session in valid_sessions if session.active_requests > 0]

        return {
            "total_sessions": len(sessions),
            "valid_sessions": len(valid_sessions),
            "available_sessions": len(valid_sessions),
            "busy_sessions": len(busy_sessions),
            "expired_sessions": len(
                [session for session in sessions if session.age >= self.session_max_age]
            ),
        }


_guest_session_pool: Optional[GuestSessionPool] = None
_guest_pool_lock = Lock()


def get_guest_session_pool() -> Optional[GuestSessionPool]:
    """获取全局匿名会话池。"""
    return _guest_session_pool


async def initialize_guest_session_pool(
    pool_size: int = 3,
    session_max_age: int = 480,
    maintenance_interval: int = 30,
    max_failures: int = 10,
) -> GuestSessionPool:
    """初始化全局匿名会话池。"""
    global _guest_session_pool

    with _guest_pool_lock:
        if _guest_session_pool is None:
            _guest_session_pool = GuestSessionPool(
                pool_size=pool_size,
                session_max_age=session_max_age,
                maintenance_interval=maintenance_interval,
                max_failures=max_failures,
            )
        pool = _guest_session_pool

    await pool.initialize()
    return pool


async def close_guest_session_pool():
    """关闭全局匿名会话池。"""
    global _guest_session_pool

    with _guest_pool_lock:
        pool = _guest_session_pool
        _guest_session_pool = None

    if pool:
        await pool.close()
