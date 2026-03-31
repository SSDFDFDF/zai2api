import pytest

from app.core.config import settings
from app.core.request_signing import sign_request
from app.core.resin_compat import (
    RESIN_ACCOUNT_HEADER,
    apply_resin_account_header,
    build_resin_account,
)


def test_build_resin_account_is_stable_and_token_specific():
    token = "token-a"
    assert build_resin_account(token) == build_resin_account(token)
    assert build_resin_account(token) != build_resin_account("token-b")
    assert build_resin_account("").strip() == ""


def test_apply_resin_account_header_respects_feature_flag(monkeypatch):
    headers = {"Authorization": "Bearer token-a"}

    monkeypatch.setattr(settings, "RESIN_COMPAT_ENABLED", False)
    apply_resin_account_header(headers, "token-a")
    assert RESIN_ACCOUNT_HEADER not in headers

    monkeypatch.setattr(settings, "RESIN_COMPAT_ENABLED", True)
    apply_resin_account_header(headers, "token-a")
    assert headers[RESIN_ACCOUNT_HEADER] == build_resin_account("token-a")


@pytest.mark.asyncio
async def test_sign_request_includes_resin_account_header_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "RESIN_COMPAT_ENABLED", True)

    signed_url, headers, _ = await sign_request(
        api_endpoint="https://chat.z.ai/api/v2/chat/completions",
        user_id="user-1",
        last_user_text="ping",
        chat_id="chat-1",
        token="token-a",
        fe_version="prod-fe-test",
    )

    assert signed_url.startswith("https://chat.z.ai/api/v2/chat/completions?")
    assert headers[RESIN_ACCOUNT_HEADER] == build_resin_account("token-a")
    assert headers["Authorization"] == "Bearer token-a"
