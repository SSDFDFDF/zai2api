#!/usr/bin/env python
"""生成 curl 命令来测试，排除 httpx 影响。"""

import asyncio
import json
import sys
import uuid
import shlex

sys.path.insert(0, ".")

from app.core.request_signing import build_upstream_body, sign_request
from app.utils.jwt_utils import extract_user_id_from_token
from app.core.config import settings

TOKEN = (
    "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpZCI6IjRjNWE4OTY4LTY5Y2QtNGQyNS1iY2ZmLThhY2RiODY3ZjZiYyIs"
    "ImVtYWlsIjoiZ2FicmllbF9jYUBmb2RkcnQuYmFyIn0."
    "1l-1mbas9_qaFFmJOkloJJb5OT6Wm_PiSAUX-rSYiD5ePA7MzwMiLq7k16AZ1mO257N-DCYP_9SH5M7K0HFTTw"
)


async def main():
    user_id = extract_user_id_from_token(TOKEN)
    chat_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    body = build_upstream_body(
        messages=[{"role": "user", "content": "hi"}],
        files=[],
        upstream_model_id="glm-5",
        last_user_text="hi",
        chat_id=chat_id,
        message_id=message_id,
        enable_thinking=False,
        web_search=False,
        auto_web_search=False,
        flags=[],
        extra={},
        mcp_servers=[],
        temperature=None,
        max_tokens=None,
    )

    signed_url, headers, _ = await sign_request(
        api_endpoint=settings.API_ENDPOINT,
        user_id=user_id,
        last_user_text="hi",
        chat_id=chat_id,
        token=TOKEN,
    )

    # 生成 curl 命令
    parts = ["curl", "-sS", "--no-buffer", shlex.quote(signed_url)]
    for k, v in headers.items():
        parts.append(f"-H {shlex.quote(f'{k}: {v}')}")
    body_json = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    parts.append(f"-d {shlex.quote(body_json)}")

    print(" \\\n  ".join(parts))
    print("\n# 查看最后几行 SSE，检查 INTERNAL_ERROR:")
    print("# 在上面命令后面加: | grep -E 'INTERNAL_ERROR|done.*true'")


if __name__ == "__main__":
    asyncio.run(main())
