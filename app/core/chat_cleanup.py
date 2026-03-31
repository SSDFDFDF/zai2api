"""Background service for token chat cleanup."""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from app.core.config import settings
from app.core.headers import build_dynamic_headers
from app.core.http_client import SharedHttpClients
from app.core.resin_compat import apply_resin_account_header
from app.core.upstream_urls import build_upstream_url
from app.services.token_dao import TokenDAO, get_token_dao
from app.utils.fe_version import get_latest_fe_version
from app.utils.logger import logger

@dataclass(frozen=True)
class ChatCleanupSummary:
    total_checked: int = 0
    success_count: int = 0
    failed_count: int = 0

async def delete_chats_for_token(
    token: str,
    clients: Optional[SharedHttpClients] = None,
    *,
    token_id: Optional[int] = None,
) -> bool:
    """Delete all chat sessions for a given token."""
    owns_clients = clients is None
    if clients is None:
        clients = SharedHttpClients()
    client = clients.get_client()
    
    fe_version = await get_latest_fe_version()
    headers = build_dynamic_headers(fe_version)
    headers["Authorization"] = f"Bearer {token}"
    apply_resin_account_header(headers, token)
    headers["Origin"] = "https://chat.z.ai"
    headers["Referer"] = "https://chat.z.ai/"
    chats_url = build_upstream_url("/api/v1/chats/")
    
    started_at = time.perf_counter()
    try:
        response = await client.request(
            method="DELETE",
            url=chats_url,
            headers=headers,
            json=None
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(
            "[token.chat_cleanup] token_id=%s status=%s bytes=%s elapsed_ms=%.1f",
            token_id,
            response.status_code,
            response.headers.get("content-length") or len(response.content or b""),
            elapsed_ms,
        )
        if response.status_code == 200:
            return True
        logger.warning(
            "⚠️ 清理会话失败 (Token ID: %s, Token: %s...): HTTP %s %s",
            token_id,
            token[:15],
            response.status_code,
            response.text,
        )
        return False
    except Exception as e:
        logger.warning(
            "⚠️ 清理会话时发生错误 (Token ID: %s, Token: %s...)",
            token_id,
            token[:15],
            exc_info=settings.DEBUG_LOGGING,
        )
        return False
    finally:
        if owns_clients:
            await clients.close()

async def run_chat_cleanup(
    interval_days: int,
    dao: Optional[TokenDAO] = None
) -> ChatCleanupSummary:
    """Clean up chat sessions for tokens that haven't been cleaned in `interval_days`."""
    token_dao = dao or get_token_dao()
    started_at = time.perf_counter()

    # 获取需要清理的 Token (启用的 zai Token)
    tokens = await token_dao.get_tokens_needing_chat_cleanup("zai", interval_days)
    if not tokens:
        return ChatCleanupSummary()

    logger.info(
        "[token.chat_cleanup.batch] start total_tokens=%s mode=serial interval_days=%s",
        len(tokens),
        interval_days,
    )
    logger.info(f"🧹 开始执行周期会话清理，共有 {len(tokens)} 个 Token 到期需要清理")

    success_count = 0
    failed_count = 0
    clients = SharedHttpClients()
    try:
        for token_record in tokens:
            token_id = int(token_record["id"])
            token_str = str(token_record["token"])
            
            success = await delete_chats_for_token(
                token_str,
                clients=clients,
                token_id=token_id,
            )
            if success:
                await token_dao.update_last_chat_cleanup(token_id)
                success_count += 1
                logger.debug("成功清理 Token 的会话: id=%s", token_id)
            else:
                failed_count += 1
                logger.debug("清理 Token 的会话失败: id=%s", token_id)
                
            # 间隔2秒，避免并发过高或被风控
            await asyncio.sleep(2.0)
    finally:
        await clients.close()

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "[token.chat_cleanup.batch] done total_tokens=%s success=%s failed=%s mode=serial elapsed_ms=%.1f",
        len(tokens),
        success_count,
        failed_count,
        elapsed_ms,
    )

    return ChatCleanupSummary(
        total_checked=len(tokens),
        success_count=success_count,
        failed_count=failed_count
    )
