import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock
from app.core.httpx_errors import normalize_httpx_exception, normalize_httpx_response
from app.core.openai_compat import get_error_message, resolve_http_error_status
from app.core.retry_policy import extract_upstream_error_details
from app.core.upstream import UpstreamClient
from app.models.schemas import OpenAIRequest

WAF_HTML = """
<body>
  <div class="site-content">
    <header>
      <h1 class="type-heading-04">403 - Forbidden</h1>
    </header>
    <main class="text-primary">
      <div>
        <p>Your request was blocked by this site's web application firewall (WAF).</p>
      </div>
      <div class="request-id">
        <p>Request ID: <code class="type-mono-01">9dcc25800907c77a</code></p>
        <p>Your IP address: <code class="type-mono-01">223.94.100.107</code></p>
      </div>
    </main>
  </div>
</body>
"""

BLOCKED_HTML = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Blocked</title>
  </head>
  <body>
    <div class="site-content">
      <header>
        <h1 class="type-heading-04">403 - Forbidden</h1>
      </header>
      <main class="text-primary">
        <p>Your request was blocked by this site's web application firewall (WAF).</p>
      </main>
    </div>
  </body>
</html>
"""

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


def test_extract_upstream_error_details_summarizes_waf_html():
    error_code, error_message = extract_upstream_error_details(403, WAF_HTML)

    assert error_code == 403
    assert error_message == "上游网页访问被拦截 (HTTP 403)"
    assert "<body>" not in error_message


def test_get_error_message_summarizes_waf_html():
    message = get_error_message(Exception(WAF_HTML))

    assert message == "上游返回网页错误"


def test_get_error_message_summarizes_blocked_doctype_html():
    message = get_error_message(Exception(BLOCKED_HTML))

    assert message == "上游返回网页错误"


def test_normalize_httpx_exception_fallback_message():
    class EmptyError(Exception):
        def __str__(self):
            return ""

    message = normalize_httpx_exception(
        EmptyError(),
        method="GET",
        url="https://chat.z.ai/api/v1/auths/",
        context="guest_session.create",
        attempt=2,
    )
    assert "guest_session.create" in message
    assert "HTTP GET" in message
    assert "attempt=2" in message
    assert "EmptyError" in message


def test_normalize_httpx_response_summarizes_html():
    message = normalize_httpx_response(
        403,
        WAF_HTML,
        content_type="text/html",
        method="POST",
        url="https://chat.z.ai/api/v1/chats/",
        context="upstream.stream.page",
    )
    assert "upstream.stream.page" in message
    assert "HTTP POST" in message
    assert "403" in message
    assert "上游网页访问被拦截" in message
    assert "<body>" not in message


@pytest.mark.parametrize(
    "error_code,error_type,expected_status",
    [
        (403, "waf_blocked", 403),
        ("model_not_found", "invalid_request_error", 404),
        ("invalid_request_error", "invalid_request_error", 400),
        ("internal_error", "upstream_error", 500),
    ],
)
def test_resolve_http_error_status(error_code, error_type, expected_status):
    assert resolve_http_error_status(error_code, error_type) == expected_status

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


@pytest.mark.asyncio
async def test_stream_html_page_response_is_intercepted():
    client_mock = AsyncMock()
    client_mock.send.return_value = httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<!DOCTYPE html><html><body><h1>Just a moment...</h1></body></html>",
        request=httpx.Request("POST", "http://api.example.com"),
    )

    upstream = UpstreamClient()
    upstream._get_shared_stream_client = MagicMock(return_value=client_mock)
    upstream._get_total_retry_limit = AsyncMock(return_value=1)

    request = OpenAIRequest(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    transformed = {
        "url": "http://api.example.com",
        "headers": {},
        "body": {},
        "token": "test-token",
        "chat_id": "test-chat-id",
        "model": "gpt-3.5-turbo",
    }

    response = await upstream._create_stream_response(request, transformed)

    assert response == {
        "error": {
            "message": "上游网页网关异常 (HTTP 502)",
            "type": "upstream_page_error",
            "code": 502,
        }
    }
