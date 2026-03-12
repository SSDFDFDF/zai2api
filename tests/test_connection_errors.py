import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock
from app.core.openai_compat import get_error_message
from app.core.upstream import UpstreamClient
from app.models.schemas import OpenAIRequest

@pytest.mark.parametrize("error_str,expected_part", [
    ("[SSL] unknown error (0xa0003e8)", "SSL/TLS 连接握手失败"),
    ("Proxy error: 403 Forbidden", "代理连接失败"),
    ("ConnectTimeout: connection timed out", "连接上游响应超时"),
    ("Generic connection error", "Generic connection error"),
])
def test_get_error_message(error_str, expected_part):
    error = Exception(error_str)
    message = get_error_message(error)
    assert expected_part in message

@pytest.mark.asyncio
async def test_upstream_connection_error_handling():
    # Mock settings and logger
    client_mock = AsyncMock()
    # Simulate a connection error during send
    client_mock.send.side_effect = httpx.ConnectError("[SSL] unknown error (0xa0003e8)")
    
    upstream = UpstreamClient()
    upstream._get_shared_stream_client = MagicMock(return_value=client_mock)
    
    request = OpenAIRequest(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hello"}],
        stream=True
    )
    
    transformed = {
        "url": "http://api.example.com",
        "headers": {},
        "body": {},
        "token": "test-token",
        "chat_id": "test-chat-id",
        "model": "gpt-3.5-turbo",
    }
    
    # We need to mock _get_total_retry_limit to return 1 for faster testing
    upstream._get_total_retry_limit = AsyncMock(return_value=1)
    
    response = await upstream._create_stream_response(request, transformed)
    
    assert "error" in response
    assert "SSL/TLS 连接握手失败" in response["error"]["message"]
    assert response["error"]["type"] == "stream_error"
