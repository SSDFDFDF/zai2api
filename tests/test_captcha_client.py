import httpx
import pytest

from app.core.captcha_client import (
    CaptchaClient,
    CaptchaProviderError,
    CaptchaTokenError,
)


@pytest.mark.asyncio
async def test_captcha_client_gets_provider_token_with_secret_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["secret"] = request.headers.get("x-secret")
        return httpx.Response(
            200,
            json={"ok": True, "token": "captcha-param", "cached": True},
        )

    client = CaptchaClient(
        provider_url="http://captcha-provider.local",
        max_retries=1,
        secret="shared-secret",
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        token = await client.get_token()
    finally:
        await client.close()

    assert token == "captcha-param"
    assert seen == {
        "method": "GET",
        "path": "/token",
        "secret": "shared-secret",
    }


@pytest.mark.asyncio
async def test_captcha_client_retries_provider_5xx(monkeypatch):
    responses = [
        httpx.Response(500, json={"ok": False, "error": "captcha timeout"}),
        httpx.Response(200, json={"ok": True, "token": "fresh-token"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("app.core.captcha_client.asyncio.sleep", fake_sleep)

    client = CaptchaClient(
        provider_url="http://captcha-provider.local",
        max_retries=2,
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        token = await client.get_token()
    finally:
        await client.close()

    assert token == "fresh-token"
    assert responses == []


@pytest.mark.asyncio
async def test_captcha_client_rejects_invalid_provider_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "token": ""})

    client = CaptchaClient(
        provider_url="http://captcha-provider.local",
        max_retries=1,
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        with pytest.raises(CaptchaTokenError):
            await client.get_token()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_captcha_client_raises_after_provider_5xx_retries(monkeypatch):
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(500, json={"ok": False, "error": "not ready"})

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("app.core.captcha_client.asyncio.sleep", fake_sleep)

    client = CaptchaClient(
        provider_url="http://captcha-provider.local",
        max_retries=2,
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        with pytest.raises(CaptchaProviderError):
            await client.get_token()
    finally:
        await client.close()

    assert requests == 2
