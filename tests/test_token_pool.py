import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.core.retry_policy import RetryPolicy
from app.core.config import settings
from app.services import token_dao as token_dao_module
from app.utils import token_pool as token_pool_module
from app.utils.token_pool import (
    TOKEN_LOAD_BALANCE_LEAST_LOADED,
    TOKEN_LOAD_BALANCE_ROUND_ROBIN,
    TOKEN_LOAD_BALANCE_SMOOTH_WEIGHTED,
    TokenPool,
)


@pytest.mark.asyncio
async def test_token_pool_round_robin_strategy_uses_ring_order():
    pool = TokenPool(
        [
            (1, "token-a", "user"),
            (2, "token-b", "user"),
            (3, "token-c", "user"),
        ],
        strategy=TOKEN_LOAD_BALANCE_ROUND_ROBIN,
    )

    assert await pool.get_next_token() == "token-a"
    assert await pool.get_next_token() == "token-b"
    assert await pool.get_next_token() == "token-c"
    assert await pool.get_next_token() == "token-a"


@pytest.mark.asyncio
async def test_token_pool_least_loaded_strategy_prefers_less_loaded_tokens():
    pool = TokenPool(
        [
            (1, "token-a", "user"),
            (2, "token-b", "user"),
            (3, "token-c", "user"),
        ],
        strategy=TOKEN_LOAD_BALANCE_LEAST_LOADED,
    )

    assert await pool.get_next_token() == "token-a"
    assert await pool.get_next_token() == "token-b"
    assert await pool.get_next_token() == "token-c"

    await pool.mark_token_success("token-a")
    assert await pool.get_next_token() == "token-a"


@pytest.mark.asyncio
async def test_token_pool_circuit_breaker_recovers_after_cooldown(monkeypatch):
    current_time = 1000.0
    monkeypatch.setattr(token_pool_module.time, "time", lambda: current_time)

    pool = TokenPool(
        [
            (1, "token-a", "user"),
            (2, "token-b", "user"),
        ],
        failure_threshold=2,
        recovery_timeout=30,
    )

    await pool.mark_token_failure("token-a", Exception("boom-1"))
    await pool.mark_token_failure("token-a", Exception("boom-2"))

    status = pool.token_statuses["token-a"]
    assert status.is_available is False
    assert status.cooldown_until == 1030.0

    assert await pool.get_next_token() == "token-b"

    current_time = 1031.0
    assert await pool.get_next_token() == "token-a"
    assert status.is_available is True
    assert status.failure_count == 0
    assert status.cooldown_until == 0.0


@pytest.mark.asyncio
async def test_token_pool_sync_from_database_rebuilds_user_order(monkeypatch):
    pool = TokenPool(
        [
            (1, "token-a", "user"),
            (2, "token-b", "user"),
        ]
    )

    class StubDAO:
        async def get_tokens_by_provider(self, provider: str, enabled_only: bool = True):
            assert provider == "zai"
            assert enabled_only is True
            return [
                {"id": 2, "token": "token-b", "token_type": "user", "priority": 10},
                {"id": 1, "token": "token-a", "token_type": "user", "priority": 0},
            ]

    monkeypatch.setattr(token_dao_module, "get_token_dao", lambda: StubDAO())

    await pool.sync_from_database("zai")

    assert await pool.get_next_token() == "token-b"
    assert await pool.get_next_token() == "token-a"


@pytest.mark.asyncio
async def test_token_pool_prefers_higher_priority_on_equal_load():
    pool = TokenPool(
        [
            (1, "token-low", "user", 0),
            (2, "token-high", "user", 20),
        ],
        strategy=TOKEN_LOAD_BALANCE_SMOOTH_WEIGHTED,
    )

    assert await pool.get_next_token() == "token-high"


@pytest.mark.asyncio
async def test_token_pool_smooth_weighted_strategy_spreads_priority_over_time():
    pool = TokenPool(
        [
            (1, "token-low", "user", 0),
            (2, "token-high", "user", 2),
        ],
        strategy=TOKEN_LOAD_BALANCE_SMOOTH_WEIGHTED,
    )

    picks = []
    for _ in range(6):
        token = await pool.get_next_token()
        picks.append(token)
        await pool.mark_token_success(token)

    assert picks == ["token-high", "token-low", "token-high", "token-high", "token-high", "token-low"]


@pytest.mark.asyncio
async def test_token_pool_set_strategy_switches_runtime_behavior():
    pool = TokenPool(
        [
            (1, "token-a", "user"),
            (2, "token-b", "user"),
        ],
        strategy=TOKEN_LOAD_BALANCE_ROUND_ROBIN,
    )

    assert await pool.get_next_token() == "token-a"
    await pool.set_strategy(TOKEN_LOAD_BALANCE_LEAST_LOADED)

    assert await pool.get_next_token() == "token-b"


@pytest.mark.asyncio
async def test_retry_policy_ignores_guest_pool_when_anonymous_disabled(monkeypatch):
    policy = RetryPolicy()
    release_mock = AsyncMock()
    report_mock = AsyncMock()

    monkeypatch.setattr(settings, "ANONYMOUS_MODE", False)
    monkeypatch.setattr(
        "app.core.retry_policy.get_guest_session_pool",
        lambda: SimpleNamespace(release=release_mock, report_failure=report_mock),
    )

    transformed = {"auth_mode": "guest", "guest_user_id": "guest-1"}
    await policy.release_guest_session(transformed)
    await policy.report_guest_session_failure(transformed)

    release_mock.assert_not_awaited()
    report_mock.assert_not_awaited()
    assert policy.should_retry_guest_session(401, False, 0, 2, transformed) is False
