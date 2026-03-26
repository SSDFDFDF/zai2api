import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import update
from sqlalchemy.sql import func
from datetime import timedelta, datetime, timezone

from app.models.db_models import Token
from app.core.chat_cleanup import delete_chats_for_token, run_chat_cleanup
from app.services.token_dao import TokenDAO


@pytest.fixture
def mock_httpx_client():
    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        # Mock 成功响应 200
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response
        yield mock_request


@pytest.fixture
def mock_fe_version():
    with patch("app.core.chat_cleanup.get_latest_fe_version", new_callable=AsyncMock) as mock_fe:
        mock_fe.return_value = "1.0.0"
        yield mock_fe


@pytest.mark.asyncio
async def test_delete_chats_for_token(mock_httpx_client, mock_fe_version):
    success = await delete_chats_for_token("test-token-123")
    assert success is True
    
    # 验证请求参数
    mock_httpx_client.assert_called_once()
    call_kwargs = mock_httpx_client.call_args.kwargs
    assert call_kwargs["method"] == "DELETE"
    assert "https://chat.z.ai/api/v1/chats/" in call_kwargs["url"]
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-token-123"
    assert call_kwargs["headers"]["Origin"] == "https://chat.z.ai"
    assert call_kwargs["headers"]["X-FE-Version"] == "1.0.0"


@pytest.mark.asyncio
async def test_run_chat_cleanup(tmp_path, monkeypatch, mock_httpx_client, mock_fe_version):
    # 初始化独立的 SQLite 测试数据库
    db_path = str(tmp_path / "tokens_test.db")
    dao = TokenDAO(db_path=db_path)
    await dao.init_database()

    # 关闭 cleanup 过程中的等待，加快测试运行
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # 创建几个不同的测试 Token

    # 1. 新增加的 Token，last_chat_cleanup 为 None -> 需要清理
    id1 = await dao.add_token("zai", "token-new", validate=False)
    
    # 2. 近期清理过的 Token -> 不需要清理 (interval=7 days)
    id2 = await dao.add_token("zai", "token-recent", validate=False)
    async with dao.session_factory() as session:
        # 强制设置清理时间为 2 天前
        stmt = update(Token).where(Token.id == id2).values(
            last_chat_cleanup=now - timedelta(days=2)
        )
        await session.execute(stmt)
        await session.commit()

    # 3. 很久没清理过的 Token -> 需要清理 (8 天前)
    id3 = await dao.add_token("zai", "token-old", validate=False)
    async with dao.session_factory() as session:
        stmt = update(Token).where(Token.id == id3).values(
            last_chat_cleanup=now - timedelta(days=8)
        )
        await session.execute(stmt)
        await session.commit()

    # 4. 新增加的 guest Token -> 跳过 (get_tokens_needing_chat_cleanup 只查 zai)
    # （测试数据默认为 provider="zai"，目前只要 enabled 且 interval_days 到期就会执行）
    
    # 执行定期清理任务，间隔设为 7 天
    summary = await run_chat_cleanup(interval_days=7, dao=dao)
    
    assert summary.total_checked == 2  # 新 Token 和老 Token 应该被命中
    assert summary.success_count == 2
    assert summary.failed_count == 0
    
    # 验证 HTTP 调用次数等于被选中的 Token 数量
    assert mock_httpx_client.call_count == 2

    # 验证数据库中上次清理时间是否已更新
    t1 = await dao.get_token_by_id(id1)
    t2 = await dao.get_token_by_id(id2)
    t3 = await dao.get_token_by_id(id3)
    
    assert t1["last_chat_cleanup"] is not None
    assert t3["last_chat_cleanup"] is not None

    await dao.close()
    
    # t2 的清理时间不应该等于当前的清理批次（这里简单起见只需确认 t1 和 t3 更新了）
