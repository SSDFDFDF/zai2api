import httpx
import pytest

from main import app


@pytest.mark.asyncio
async def test_fastapi_builtin_docs_endpoints_are_disabled():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        docs_response = await client.get("/docs")
        redoc_response = await client.get("/redoc")
        openapi_response = await client.get("/openapi.json")

    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
    assert docs_response.status_code == 404
    assert redoc_response.status_code == 404
    assert openapi_response.status_code == 404
