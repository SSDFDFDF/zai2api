#!/usr/bin/env python
"""验证：先 POST /api/v1/chats/new 创建会话，再发 completions。"""

import asyncio
import json
import sys
import time
import uuid

import httpx

sys.path.insert(0, ".")

from app.core.request_signing import build_upstream_body, sign_request
from app.core.message_preprocessing import extract_user_id_from_token
from app.core.headers import build_dynamic_headers
from app.core.config import settings
from app.utils.fe_version import get_latest_fe_version

TOKEN = (
    "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpZCI6IjRjNWE4OTY4LTY5Y2QtNGQyNS1iY2ZmLThhY2RiODY3ZjZiYyIs"
    "ImVtYWlsIjoiZ2FicmllbF9jYUBmb2RkcnQuYmFyIn0."
    "1l-1mbas9_qaFFmJOkloJJb5OT6Wm_PiSAUX-rSYiD5ePA7MzwMiLq7k16AZ1mO257N-DCYP_9SH5M7K0HFTTw"
)

MODEL = "glm-5"


async def create_chat(client: httpx.AsyncClient, message_id: str, content: str) -> str:
    """调用 /api/v1/chats/new 预创建会话，返回 chat_id。"""
    fe_version = await get_latest_fe_version()
    headers = build_dynamic_headers(fe_version)
    headers["Authorization"] = f"Bearer {TOKEN}"

    now_ts = int(time.time())
    body = {
        "chat": {
            "id": "",
            "title": "新聊天",
            "models": [MODEL],
            "params": {},
            "history": {
                "messages": {
                    message_id: {
                        "id": message_id,
                        "parentId": None,
                        "childrenIds": [],
                        "role": "user",
                        "content": content,
                        "timestamp": now_ts,
                        "models": [MODEL],
                    }
                },
                "currentId": message_id,
            },
            "tags": [],
            "flags": [],
            "features": [
                {"type": "tool_selector", "server": "tool_selector_h", "status": "hidden"}
            ],
            "mcp_servers": [],
            "enable_thinking": False,
            "auto_web_search": False,
            "message_version": 1,
            "extra": {},
            "timestamp": int(time.time() * 1000),
        }
    }

    resp = await client.post(
        "https://chat.z.ai/api/v1/chats/new",
        json=body,
        headers=headers,
    )
    print(f"  /chats/new: HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  Error: {resp.text[:200]}")
        return ""

    data = resp.json()
    chat_id = data.get("id", "")
    print(f"  Got chat_id: {chat_id}")
    return chat_id


async def send_completion(client: httpx.AsyncClient, chat_id: str, message_id: str) -> bool:
    """发送 completions 请求，检查是否有 INTERNAL_ERROR。"""
    user_id = extract_user_id_from_token(TOKEN)

    body = build_upstream_body(
        messages=[{"role": "user", "content": "hi"}],
        files=[],
        upstream_model_id=MODEL,
        last_user_text="hi",
        chat_id=chat_id,
        message_id=str(uuid.uuid4()),
        enable_thinking=False,
        web_search=False,
        auto_web_search=False,
        flags=[],
        extra={},
        mcp_servers=[],
        temperature=None,
        max_tokens=None,
    )
    # 使用和 chats/new 相同的 message_id
    body["current_user_message_id"] = message_id

    signed_url, headers, _ = await sign_request(
        api_endpoint=settings.API_ENDPOINT,
        user_id=user_id,
        last_user_text="hi",
        chat_id=chat_id,
        token=TOKEN,
    )

    resp = await client.send(
        client.build_request("POST", signed_url, json=body, headers=headers),
        stream=True,
    )
    print(f"  completions: HTTP {resp.status_code}")

    found_error = False
    async for line in resp.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("[DONE]", "DONE"):
            continue
        try:
            chunk = json.loads(payload)
            data = chunk.get("data", chunk)
            if isinstance(data, dict) and data.get("error"):
                found_error = True
                print(f"  ERROR at done={data.get('done')}: {data['error']}")
            elif isinstance(data, dict) and data.get("done"):
                print(f"  done=true (no error)")
        except json.JSONDecodeError:
            pass
    await resp.aclose()
    return found_error


async def main():
    message_id = str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        # 测试 1: 有预创建
        print("=== WITH pre-create (/chats/new) ===")
        chat_id = await create_chat(client, message_id, "hi")
        if chat_id:
            err1 = await send_completion(client, chat_id, message_id)
            print(f"  Result: {'ERROR' if err1 else 'OK'}")
        else:
            print("  Failed to create chat, skipping")
            err1 = True

        await asyncio.sleep(1)

        # 测试 2: 无预创建（当前行为）
        print("\n=== WITHOUT pre-create (random chat_id) ===")
        random_chat_id = str(uuid.uuid4())
        err2 = await send_completion(client, random_chat_id, str(uuid.uuid4()))
        print(f"  Result: {'ERROR' if err2 else 'OK'}")

        print(f"\n{'='*40}")
        print(f"With pre-create:    {'ERROR' if err1 else 'OK'}")
        print(f"Without pre-create: {'ERROR' if err2 else 'OK'}")


if __name__ == "__main__":
    asyncio.run(main())
