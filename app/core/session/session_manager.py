#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""会话管理器：连续会话的核心协调模块。

职责：
1. 通过指纹匹配检测连续会话
2. 维护 chat_id / parent_id 的生命周期
3. 后台定期清理过期会话
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.session.session_fingerprint import SessionFingerprint
from app.core.session.session_store import SessionStore
from app.utils.logger import get_logger

logger = get_logger()


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ──────────────────────────────────────────────────────────────────────────────
# 数据类
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionResult:
    """会话查找/创建的结果，供 transform_request 使用。"""

    chat_id: str
    """复用或新建的对话 ID。"""

    message_id: str
    """本次请求生成的消息 ID（current_user_message_id）。"""

    parent_id: Optional[str]
    """连续会话时指向上次的 message_id，新对话时为 None。"""

    is_new: bool
    """True = 新建对话，False = 复用已有会话。"""

    bound_token: Optional[str] = None
    """复用会话时绑定的原始 token，新对话时为 None。"""

    trigger_signal: Optional[str] = None
    """该会话绑定的 Toolify XML 触发信号。"""


# ──────────────────────────────────────────────────────────────────────────────
# 存储键辅助
# ──────────────────────────────────────────────────────────────────────────────

_SESSION_PREFIX = "session"
_FP_INDEX_PREFIX = "fp_index"


def _session_key(client_fp: str, chat_id: str) -> str:
    return f"{_SESSION_PREFIX}:{client_fp}:{chat_id}"


def _fp_index_key(client_fp: str) -> str:
    return f"{_FP_INDEX_PREFIX}:{client_fp}"


# ──────────────────────────────────────────────────────────────────────────────
# SessionManager
# ──────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """连续会话管理器。

    单例使用方式（由 UpstreamClient 持有）：
        self._session_manager = SessionManager()
    """

    def __init__(
        self,
        session_ttl: int = 3600,
        max_sessions_per_client: int = 50,
        cleanup_interval: int = 300,
    ) -> None:
        self._ttl = session_ttl
        self._max_per_client = max(1, max_sessions_per_client)
        self._cleanup_interval = cleanup_interval
        self._store = SessionStore()
        self._fp = SessionFingerprint()

        self._cleanup_task: Optional[asyncio.Task] = None

    # ──────────────────────────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────────────────────────

    async def find_session(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        client_id: Optional[str] = None,
    ) -> Optional[SessionResult]:
        """Find matching continuous session by fingerprint. Returns None if not found.

        Unlike find_or_create_session, this method only searches -- it never
        creates a new session.  On match it updates fingerprints, last_message_id
        and timestamps so that subsequent turns keep chaining correctly.

        Args:
            model: Model name used for index grouping.
            messages: Full message list (OpenAI format) for fingerprint matching.
            client_id: Optional explicit client ID (used for index key if provided).

        Returns:
            SessionResult on match, None otherwise.
        """
        self._ensure_cleanup_started()

        index_identifier = client_id or model
        client_fp = self._fp.generate_client_fingerprint(index_identifier, model)

        match = await self._match_session(client_fp, messages)
        if not match:
            return None

        session = match
        message_id = _new_uuid()
        chat_id = session["chat_id"]
        parent_id = session.get("last_message_id")

        # Update session: new fingerprints + new message_id
        new_fps = self._fp.collect_fingerprints(messages)
        session["fingerprints"] = new_fps
        session["last_message_id"] = message_id
        session["last_update"] = time.time()
        await self._save_session(client_fp, chat_id, session)
        await self._touch_fp_index(client_fp, chat_id)

        logger.debug(
            f"♻️ 连续会话复用: chat_id={chat_id[:8]}... "
            f"parent_id={parent_id[:8] if parent_id else 'None'}..."
        )
        return SessionResult(
            chat_id=chat_id,
            message_id=message_id,
            parent_id=parent_id,
            is_new=False,
            bound_token=session.get("auth_token"),
            trigger_signal=session.get("trigger_signal"),
        )

    async def create_session(
        self,
        auth_token: str,
        model: str,
        messages: List[Dict[str, Any]],
        chat_id: str,
        message_id: str,
        trigger_signal: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> SessionResult:
        """Create a new session with an externally-provided chat_id (from upstream).

        This is the counterpart to find_session: the caller has already obtained
        a chat_id from the upstream /chats/new endpoint and wants to store it
        for future reuse.

        Args:
            auth_token: Upstream auth token (stored for bound_token on reuse).
            model: Model name for index grouping.
            messages: Full message list (OpenAI format) for fingerprint collection.
            chat_id: Chat ID returned by the upstream pre-create endpoint.
            message_id: UUID generated for this turn's message.
            trigger_signal: Toolify XML trigger signal for this chat session.
            client_id: Optional explicit client ID (used for index key if provided).

        Returns:
            SessionResult with is_new=True.
        """
        self._ensure_cleanup_started()

        index_identifier = client_id or model
        client_fp = self._fp.generate_client_fingerprint(index_identifier, model)

        fingerprints = self._fp.collect_fingerprints(messages)
        record: Dict[str, Any] = {
            "chat_id": chat_id,
            "client_fingerprint": client_fp,
            "fingerprints": fingerprints,
            "last_message_id": message_id,
            "model": model,
            "token_hash": self._fp.hash_token(auth_token),
            "auth_token": auth_token,
            "trigger_signal": trigger_signal or "",
            "created_at": time.time(),
            "last_update": time.time(),
        }
        await self._save_session(client_fp, chat_id, record)
        await self._update_fp_index(client_fp, chat_id)

        logger.debug(f"❇️ 新建会话: chat_id={chat_id[:8]}... model={model}")
        return SessionResult(
            chat_id=chat_id,
            message_id=message_id,
            parent_id=None,
            is_new=True,
            trigger_signal=trigger_signal,
        )

    async def update_session_message_id(
        self,
        client_fp_or_chat_id: str,
        chat_id: str,
        new_message_id: str,
    ) -> None:
        """（可选）手动刷新会话的 last_message_id。

        通常无需调用——find_or_create_session 已经在创建/匹配时更新。
        当需要从上游响应中提取真实 assistant message_id 时可调用。
        """
        # 先尝试按 chat_id 找 client_fp（遍历 fp_index）
        # 简化：若调用方已知 client_fp 则直接操作
        session = await self._store.get(_session_key(client_fp_or_chat_id, chat_id))
        if not session:
            return
        session["last_message_id"] = new_message_id
        session["last_update"] = time.time()
        await self._store.set(
            _session_key(client_fp_or_chat_id, chat_id), session, ttl=self._ttl
        )

    async def clear_session(self, client_fp: str, chat_id: str) -> None:
        """清除单个会话及其指纹索引条目。"""
        await self._store.delete(_session_key(client_fp, chat_id))
        await self._remove_from_fp_index(client_fp, chat_id)

    async def get_stats(self) -> Dict[str, Any]:
        """返回存储统计信息（用于 Admin 面板）。"""
        session_keys = await self._store.keys(f"{_SESSION_PREFIX}:")
        fp_keys = await self._store.keys(f"{_FP_INDEX_PREFIX}:")
        return {
            "total_sessions": len(session_keys),
            "total_clients": len(fp_keys),
            "store_size": self._store.size(),
            "session_ttl": self._ttl,
            "max_per_client": self._max_per_client,
        }

    async def close(self) -> None:
        """停止后台清理任务。"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        async with self._store._lock:
            self._store._data.clear()

    # ──────────────────────────────────────────────────────────────────
    # 内部：会话匹配
    # ──────────────────────────────────────────────────────────────────

    async def _match_session(
        self,
        client_fp: str,
        messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """在该客户端的会话列表中查找匹配的连续会话。"""
        index = await self._store.get(_fp_index_key(client_fp))
        if not index:
            return None

        chat_ids: List[str] = index.get("chat_ids", [])
        if not chat_ids:
            return None

        # 从最新的会话开始倒序匹配
        for chat_id in reversed(chat_ids):
            session = await self._store.get(_session_key(client_fp, chat_id))
            if not session:
                # 已过期，从索引中清理
                await self._remove_from_fp_index(client_fp, chat_id)
                continue
            cached_fps: List[str] = session.get("fingerprints", [])
            if self._fp.is_continuous_session(messages, cached_fps):
                return session

        return None

    # ──────────────────────────────────────────────────────────────────
    # 内部：存储操作
    # ──────────────────────────────────────────────────────────────────

    async def _save_session(
        self, client_fp: str, chat_id: str, session: Dict[str, Any]
    ) -> None:
        key = _session_key(client_fp, chat_id)
        await self._store.set(key, session, ttl=self._ttl)

    async def _update_fp_index(self, client_fp: str, chat_id: str) -> None:
        """新增 chat_id 到指纹索引，超出上限时驱逐最旧的。"""
        index_key = _fp_index_key(client_fp)
        index = await self._store.get(index_key) or {
            "client_fingerprint": client_fp,
            "chat_ids": [],
            "created_at": time.time(),
        }
        chat_ids: List[str] = index.get("chat_ids", [])
        if chat_id in chat_ids:
            chat_ids.remove(chat_id)
        chat_ids.append(chat_id)

        # 超出上限：驱逐最旧的会话
        while len(chat_ids) > self._max_per_client:
            evict_id = chat_ids.pop(0)
            await self._store.delete(_session_key(client_fp, evict_id))
            logger.debug(f"🗑️ 驱逐超限会话: {evict_id[:8]}...")

        index["chat_ids"] = chat_ids
        index["most_recent"] = chat_id
        index["last_update"] = time.time()
        await self._store.set(index_key, index, ttl=self._ttl)

    async def _touch_fp_index(self, client_fp: str, chat_id: str) -> None:
        """将已匹配的 chat_id 移到列表末尾（最近使用），并刷新 TTL。"""
        index_key = _fp_index_key(client_fp)
        index = await self._store.get(index_key)
        if not index:
            return
        chat_ids: List[str] = index.get("chat_ids", [])
        if chat_id in chat_ids:
            chat_ids.remove(chat_id)
            chat_ids.append(chat_id)
        index["chat_ids"] = chat_ids
        index["most_recent"] = chat_id
        index["last_update"] = time.time()
        await self._store.set(index_key, index, ttl=self._ttl)

    async def _remove_from_fp_index(self, client_fp: str, chat_id: str) -> None:
        """从指纹索引中移除 chat_id（会话已过期或已删除）。"""
        index_key = _fp_index_key(client_fp)
        index = await self._store.get(index_key)
        if not index:
            return
        chat_ids: List[str] = index.get("chat_ids", [])
        if chat_id in chat_ids:
            chat_ids.remove(chat_id)
            if chat_ids:
                index["chat_ids"] = chat_ids
                index["last_update"] = time.time()
                await self._store.set(index_key, index, ttl=self._ttl)
            else:
                await self._store.delete(index_key)

    # ──────────────────────────────────────────────────────────────────
    # 内部：后台清理
    # ──────────────────────────────────────────────────────────────────

    def _ensure_cleanup_started(self) -> None:
        """惰性启动后台清理协程（仅在事件循环中首次调用时启动）。"""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                self._cleanup_task = asyncio.get_event_loop().create_task(
                    self._cleanup_loop()
                )
            except RuntimeError:
                pass  # 无事件循环（测试环境）时忽略

    async def _cleanup_loop(self) -> None:
        """定期清理过期会话。"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                removed = await self._store.cleanup_expired()
                if removed:
                    logger.debug(f"🧹 会话清理：清除 {removed} 条过期记录")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"会话清理异常: {exc}")
