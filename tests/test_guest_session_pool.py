import httpx
import jwt
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.utils.guest_session_pool import GuestSession

from app.admin import api as admin_api
from app.core.config import settings
from app.core.http_client import SharedHttpClients
from app.utils import guest_session_pool as guest_session_pool_module
from app.utils.guest_session_pool import AUTH_URL, GuestSessionPool


def test_shared_http_clients_disable_implicit_env_proxy():
    with pytest.MonkeyPatch.context() as monkeypatch:
        async_client_cls = MagicMock(side_effect=["client", "stream-client"])
        monkeypatch.setattr("app.core.http_client.httpx.AsyncClient", async_client_cls)

        clients = SharedHttpClients(follow_redirects=True)

        assert clients.get_client() == "client"
        assert clients.get_stream_client() == "stream-client"

    assert async_client_cls.call_args_list[0].kwargs["trust_env"] is False
    assert async_client_cls.call_args_list[1].kwargs["trust_env"] is False


@pytest.mark.asyncio
async def test_create_session_retries_remote_protocol_error(monkeypatch):
    payload = {
        "id": "9015da82-1ac5-400a-b256-ba915c3d45e1",
        "email": "Guest-1773376198189@guest.com",
    }
    token = jwt.encode(payload, "secret", algorithm="HS256")

    response = httpx.Response(
        200,
        json={"token": token},
        request=httpx.Request("GET", AUTH_URL),
    )
    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[
            httpx.RemoteProtocolError(
                "Server disconnected without sending a response."
            ),
            response,
        ]
    )

    pool = GuestSessionPool(pool_size=1)
    pool._http_clients = MagicMock()
    pool._http_clients.get_client.return_value = client

    sleep_mock = AsyncMock()
    monkeypatch.setattr(
        guest_session_pool_module,
        "get_latest_fe_version",
        AsyncMock(return_value="prod-fe-1.0.107"),
    )
    monkeypatch.setattr(
        guest_session_pool_module,
        "_build_dynamic_headers",
        lambda fe_version: {"X-FE-Version": fe_version},
    )
    monkeypatch.setattr(guest_session_pool_module.asyncio, "sleep", sleep_mock)

    session = await pool._create_session(reason="test")

    assert session.user_id == payload["id"]
    assert session.username == "Guest-1773376198189"
    assert client.get.await_count == 2
    assert sleep_mock.await_count == 2
    sleep_mock.assert_any_await(1)


@pytest.mark.asyncio
async def test_ensure_capacity_stops_on_duplicate_user_ids(monkeypatch):
    pool = GuestSessionPool(pool_size=3)
    duplicate_session = GuestSession(
        token="token-1",
        user_id="duplicate-user",
        username="Guest-duplicate",
    )

    create_mock = AsyncMock(side_effect=[duplicate_session, duplicate_session, duplicate_session])
    monkeypatch.setattr(pool, "_create_session", create_mock)

    success = await pool._ensure_capacity()

    assert success is False
    assert create_mock.await_count == 3
    assert list(pool._sessions.keys()) == ["duplicate-user"]


@pytest.mark.asyncio
async def test_create_session_rate_limit_waits(monkeypatch):
    pool = GuestSessionPool(pool_size=1)
    sleep_mock = AsyncMock()
    now = {"value": 100.0}

    monkeypatch.setattr(guest_session_pool_module.asyncio, "sleep", sleep_mock)
    monkeypatch.setattr(guest_session_pool_module.time, "time", lambda: now["value"])

    pool._next_create_allowed_at = 101.5
    await pool._wait_create_slot("test")

    sleep_mock.assert_awaited_once_with(1.5)
    assert pool._next_create_allowed_at == 101.0


@pytest.mark.asyncio
async def test_initialize_guest_session_pool_rejects_when_anonymous_disabled(monkeypatch):
    monkeypatch.setattr(guest_session_pool_module.settings, "ANONYMOUS_MODE", False)

    with pytest.raises(RuntimeError, match="ANONYMOUS_MODE is disabled"):
        await guest_session_pool_module.initialize_guest_session_pool()