"""Background service for token chat cleanup."""

import asyncio
from dataclasses import dataclass
from typing import Optional

from app.core.headers import build_dynamic_headers
from app.core.http_client import SharedHttpClients
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
) -> bool:
    """Delete all chat sessions for a given token."""
    owns_clients = clients is None
    if clients is None:
        clients = SharedHttpClients()
    client = clients.get_client()
    
    fe_version = await get_latest_fe_version()
    headers = build_dynamic_headers(fe_version)
    headers["Authorization"] = f"Bearer {token}"
    headers["Origin"] = "https://chat.z.ai"
    headers["Referer"] = "https://chat.z.ai/"
    
    try:
        response = await client.request(
            method="DELETE",
            url="https://chat.z.ai/api/v1/chats/",
            headers=headers,
            json=None
        )
        if response.status_code == 200:
            return True
        logger.warning(f"⚠️ 清理会话失败 (Token: {token[:15]}...): HTTP {response.status_code} {response.text}")
        return False
    except Exception as e:
        logger.warning(f"⚠️ 清理会话时发生错误 (Token: {token[:15]}...): {e}")
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
    
    # 获取需要清理的 Token (启用的 zai Token)
    tokens = await token_dao.get_tokens_needing_chat_cleanup("zai", interval_days)
    if not tokens:
        return ChatCleanupSummary()
        
    logger.info(f"🧹 开始执行周期会话清理，共有 {len(tokens)} 个 Token 到期需要清理")
    
    success_count = 0
    failed_count = 0
    clients = SharedHttpClients()
    try:
        for token_record in tokens:
            token_id = int(token_record["id"])
            token_str = str(token_record["token"])
            
            success = await delete_chats_for_token(token_str, clients=clients)
            if success:
                await token_dao.update_last_chat_cleanup(token_id)
                success_count += 1
                logger.debug(f"✅ 成功清理 Token 的会话: id={token_id}")
            else:
                failed_count += 1
                logger.debug(f"❌ 清理 Token 的会话失败: id={token_id}")
                
            # 间隔2秒，避免并发过高或被风控
            await asyncio.sleep(2.0)
    finally:
        await clients.close()

    return ChatCleanupSummary(
        total_checked=len(tokens),
        success_count=success_count,
        failed_count=failed_count
    )
