#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""轻量级内存会话存储（支持 TTL 自动过期）。

设计原则：
- 无外部依赖（纯 Python，stdlib 仅需 dict + asyncio）
- 惰性过期：访问时检查 TTL，周期性后台清理
- 线程安全：通过 asyncio.Lock 保护共享状态
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from app.utils.logger import get_logger

logger = get_logger()


class SessionStore:
    """基于内存的 KV 存储，支持 TTL 过期。

    存储格式：
    - Key: 字符串（如 "session:{fp}:{chat_id}" 或 "fp_index:{fp}"）
    - Value: 任意可序列化 dict
    - TTL: 过期秒数，0 表示永不过期
    """

    def __init__(self) -> None:
        # {key: (value, expire_at)}  expire_at=0 表示永不过期
        self._data: Dict[str, Tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """获取 key 对应值，已过期返回 None。"""
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expire_at = entry
            if expire_at and time.monotonic() > expire_at:
                del self._data[key]
                return None
            return value

    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> None:
        """设置 key 的值，ttl 秒后过期（ttl=0 永不过期）。"""
        expire_at = (time.monotonic() + ttl) if ttl > 0 else 0.0
        async with self._lock:
            self._data[key] = (value, expire_at)

    async def delete(self, key: str) -> None:
        """删除指定 key。"""
        async with self._lock:
            self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        """检查 key 是否存在且未过期。"""
        return await self.get(key) is not None

    async def keys(self, prefix: str = "") -> List[str]:
        """返回所有匹配前缀的未过期 key 列表。"""
        now = time.monotonic()
        async with self._lock:
            result = []
            # 顺便清理已过期的 key
            expired = []
            for k, (_, expire_at) in self._data.items():
                if expire_at and now > expire_at:
                    expired.append(k)
                elif not prefix or k.startswith(prefix):
                    result.append(k)
            for k in expired:
                del self._data[k]
            return result

    async def cleanup_expired(self) -> int:
        """主动清理所有过期条目，返回清理数量。"""
        now = time.monotonic()
        async with self._lock:
            expired = [k for k, (_, exp) in self._data.items() if exp and now > exp]
            for k in expired:
                del self._data[k]
            return len(expired)

    def size(self) -> int:
        """返回当前存储条目数（含已过期但未清理的）。"""
        return len(self._data)
