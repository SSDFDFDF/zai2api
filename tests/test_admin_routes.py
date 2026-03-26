from fastapi import FastAPI
import httpx
import pytest

from app.admin import routes as admin_routes


@pytest.mark.asyncio
async def test_login_page_renders():
    app = FastAPI()
    app.include_router(admin_routes.router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get("/admin/login")

    assert response.status_code == 200
    assert "API 管理后台" in response.text
    assert "请输入管理密码以继续" in response.text
