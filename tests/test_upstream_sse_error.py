#!/usr/bin/env python
"""最小化测试：直接请求上游，检查 done:true 时是否有 INTERNAL_ERROR。"""

import asyncio
import json
import sys
import uuid

import httpx

sys.path.insert(0, ".")

from app.core.request_signing import build_upstream_body, sign_request
from app.core.message_preprocessing import extract_user_id_from_token
from app.core.config import settings


TOKEN = (
    "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpZCI6IjRjNWE4OTY4LTY5Y2QtNGQyNS1iY2ZmLThhY2RiODY3ZjZiYyIs"
    "ImVtYWlsIjoiZ2FicmllbF9jYUBmb2RkcnQuYmFyIn0."
    "1l-1mbas9_qaFFmJOkloJJb5OT6Wm_PiSAUX-rSYiD5ePA7MzwMiLq7k16AZ1mO257N-DCYP_9SH5M7K0HFTTw"
)


async def test_upstream(model: str = "glm-4.7"):
    user_id = extract_user_id_from_token(TOKEN)
    chat_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    body = build_upstream_body(
        messages=[{"role": "user", "content": "hi"}],
        files=[],
        upstream_model_id=model,
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

    print(f"--- Request body ---")
    sanitized = {
        k: (f"[{len(v)} messages]" if k == "messages" else v)
        for k, v in body.items()
    }
    print(json.dumps(sanitized, ensure_ascii=False, indent=2))

    signed_url, headers, fe_ver = await sign_request(
        api_endpoint=settings.API_ENDPOINT,
        user_id=user_id,
        last_user_text="hi",
        chat_id=chat_id,
        token=TOKEN,
    )

    print(f"\n--- Sending to upstream (model={model}) ---")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        req = client.build_request("POST", signed_url, json=body, headers=headers)
        resp = await client.send(req, stream=True)
        print(f"HTTP {resp.status_code}")

        line_num = 0
        found_error = False
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line:
                continue
            line_num += 1

            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload in ("[DONE]", "DONE"):
                    print(f"  #{line_num}: [DONE]")
                    continue
                try:
                    chunk = json.loads(payload)
                    data = chunk.get("data", chunk)
                    if isinstance(data, dict):
                        done = data.get("done")
                        error = data.get("error")
                        phase = data.get("phase", "")
                        content = data.get("content", "")

                        if error:
                            found_error = True
                            print(f"  #{line_num}: *** ERROR at done={done} ***")
                            print(f"           {json.dumps(error, ensure_ascii=False)}")
                        elif done:
                            print(f"  #{line_num}: done=true, phase={phase}")
                        elif phase:
                            preview = content[:50] if content else ""
                            print(f"  #{line_num}: phase={phase}, content={preview!r}")
                except json.JSONDecodeError:
                    print(f"  #{line_num}: (non-json) {payload[:80]}")

        await resp.aclose()

    print(f"\n--- Result: {'ERROR FOUND' if found_error else 'NO ERROR'} ---")
    return found_error


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "glm-4.7"
    asyncio.run(test_upstream(model))
