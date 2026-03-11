#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""会话指纹生成与匹配。

使用 blake2s（stdlib，比 MD5 更快）生成消息和客户端指纹。
"""

import hashlib
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger()


def _fast_hash(data: str, length: int = 16) -> str:
    """快速哈希：blake2s（stdlib 内置，比 MD5 约快 30-50%）。"""
    return hashlib.blake2s(data.encode("utf-8"), digest_size=8).hexdigest()[:length]


class SessionFingerprint:
    """处理会话指纹生成和匹配。"""

    # 每个会话最多保留最近多少条消息的指纹
    MAX_CACHED_FINGERPRINTS = 10

    @staticmethod
    def generate_client_fingerprint(identifier: str, model: str) -> str:
        """生成客户端指纹（token + model → 固定 16 字符）。

        Args:
            identifier: 认证 token 或 client_id。
            model: 请求模型名。

        Returns:
            16 字符十六进制指纹。
        """
        base = identifier or ""
        prefix = base[:20] if len(base) > 20 else base
        return _fast_hash(f"{prefix}:{model}", length=16)

    @staticmethod
    def message_fingerprint(message: Dict[str, Any]) -> str:
        """生成单条消息的指纹（role + content）。"""
        role = message.get("role", "")
        content = message.get("content", "")
        # 对列表内容（多模态）做简化处理：转为字符串
        if isinstance(content, list):
            content = str(content)
        return _fast_hash(f"{role}:{content}", length=16)

    @staticmethod
    def collect_fingerprints(messages: List[Dict[str, Any]]) -> List[str]:
        """收集消息列表末尾 N 条的指纹（用于存储到会话中）。"""
        fps = [SessionFingerprint.message_fingerprint(m) for m in messages]
        return fps[-SessionFingerprint.MAX_CACHED_FINGERPRINTS:]

    @staticmethod
    def is_continuous_session(
        new_messages: List[Dict[str, Any]],
        cached_fingerprints: List[str],
    ) -> bool:
        """判断当前请求是否为已知会话的延续。

        判定规则（适配 OpenAI 多轮对话格式）：
        - 消息数 <= 1：必然是新对话（首轮消息）
        - 消息数 == 3：3 条 (user/assistant/user) 时，检查第 1 条是否命中缓存
        - 消息数 >= 5：检查倒数第 3~5 条，命中 2 条以上即视为连续

        Args:
            new_messages: 当前请求的完整消息列表。
            cached_fingerprints: 上次会话保存的指纹列表。

        Returns:
            True 表示是连续会话，False 表示新对话。
        """
        msg_count = len(new_messages)

        if msg_count <= 1:
            return False

        if not cached_fingerprints:
            return False

        new_fps = [SessionFingerprint.message_fingerprint(m) for m in new_messages]
        cached_set = set(cached_fingerprints)

        # 3 条消息：首条匹配即认为连续
        if msg_count == 3:
            if new_fps[0] in cached_set:
                logger.debug("连续会话命中：3 条消息首条匹配缓存")
                return True
            return False

        # 5 条及以上：检查倒数 3~5 条，命中 >= 2 条视为连续
        if msg_count >= 5:
            check_indices = [-5, -4, -3]
            match_count = sum(
                1 for idx in check_indices
                if abs(idx) <= len(new_fps) and new_fps[idx] in cached_set
            )
            if match_count >= 2:
                logger.debug(f"连续会话命中：{msg_count} 条消息，倒数命中 {match_count}/3 条")
                return True

        # 4 条（user/assistant/user/assistant 或其他）：任意一条历史匹配
        if msg_count == 4:
            # 检查前 2 条（历史部分）
            match_count = sum(1 for fp in new_fps[:2] if fp in cached_set)
            if match_count >= 1:
                logger.debug("连续会话命中：4 条消息，历史部分匹配")
                return True

        return False

    @staticmethod
    def hash_token(token: Optional[str]) -> Optional[str]:
        """生成 token 哈希（用于日志脱敏）。"""
        if not token:
            return None
        return _fast_hash(token, length=16)
