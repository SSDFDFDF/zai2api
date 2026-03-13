#!/usr/bin/env python
"""精确对照：用浏览器 exact body vs 代码生成 body 对比。"""

import asyncio
import json
import sys
import uuid

import httpx

sys.path.insert(0, ".")

from app.core.request_signing import sign_request
from app.core.message_preprocessing import extract_user_id_from_token

TOKEN = (
    "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpZCI6IjRjNWE4OTY4LTY5Y2QtNGQyNS1iY2ZmLThhY2RiODY3ZjZiYyIs"
    "ImVtYWlsIjoiZ2FicmllbF9jYUBmb2RkcnQuYmFyIn0."
    "1l-1mbas9_qaFFmJOkloJJb5OT6Wm_PiSAUX-rSYiD5ePA7MzwMiLq7k16AZ1mO257N-DCYP_9SH5M7K0HFTTw"
)
API_ENDPOINT = "https://chat.z.ai/api/v2/chat/completions"


async def send_and_check(label: str, body: dict) -> bool:
    user_id = extract_user_id_from_token(TOKEN)
    signed_url, headers, _ = await sign_request(
        api_endpoint=API_ENDPOINT,
        user_id=user_id,
        last_user_text="hi",
        chat_id=body["chat_id"],
        token=TOKEN,
    )

    print(f"\n=== {label} ===")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        resp = await client.send(
            client.build_request("POST", signed_url, json=body, headers=headers),
            stream=True,
        )
        print(f"  HTTP {resp.status_code}")
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
            except json.JSONDecodeError:
                pass
        await resp.aclose()
    print(f"  Result: {'ERROR' if found_error else 'OK'}")
    return found_error


async def main():
    chat_id = str(uuid.uuid4())
    msg_id = str(uuid.uuid4())
    user_msg_id = str(uuid.uuid4())

    # 浏览器 exact body（仅改 chat_id/id/current_user_message_id 为新值）
    browser_body = {
        "stream": True,
        "model": "glm-5",
        "messages": [{"role": "user", "content": "hi"}],
        "signature_prompt": "hi",
        "params": {},
        "extra": {},
        "features": {
            "image_generation": False,
            "web_search": False,
            "auto_web_search": False,
            "preview_mode": True,
            "flags": [],
            "enable_thinking": False,
        },
        "variables": {
            "{{USER_NAME}}": "ddsf",
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": "2026-03-13 10:49:50",
            "{{CURRENT_DATE}}": "2026-03-13",
            "{{CURRENT_TIME}}": "10:49:50",
            "{{CURRENT_WEEKDAY}}": "Friday",
            "{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
            "{{USER_LANGUAGE}}": "zh-CN",
        },
        "chat_id": chat_id,
        "id": msg_id,
        "current_user_message_id": user_msg_id,
        "current_user_message_parent_id": None,
        "background_tasks": {
            "title_generation": True,
            "tags_generation": True,
        },
    }

    # 代码生成的 body（和 build_upstream_body 输出一致）
    code_body = {
        "stream": True,
        "model": "glm-5",
        "messages": [{"role": "user", "content": "hi"}],
        "signature_prompt": "hi",
        "params": {},
        "extra": {},
        "features": {
            "image_generation": False,
            "web_search": False,
            "auto_web_search": False,
            "preview_mode": True,
            "flags": [],
            "enable_thinking": False,
        },
        "background_tasks": {
            "title_generation": True,
            "tags_generation": True,
        },
        "variables": {
            "{{USER_NAME}}": "User",
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": "2026-03-13 10:49:50",
            "{{CURRENT_DATE}}": "2026-03-13",
            "{{CURRENT_TIME}}": "10:49:50",
            "{{CURRENT_WEEKDAY}}": "Friday",
            "{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
            "{{USER_LANGUAGE}}": "zh-CN",
        },
        "chat_id": chat_id,   # 共享同一个 chat_id
        "id": msg_id,
        "current_user_message_id": user_msg_id,
        "current_user_message_parent_id": None,
    }

    # 测试 1: 浏览器格式
    err1 = await send_and_check("Browser-style body", browser_body)

    # 等一秒避免限流
    await asyncio.sleep(1)

    # 测试 2: 代码格式
    # 用不同的 chat_id 避免冲突
    code_body["chat_id"] = str(uuid.uuid4())
    err2 = await send_and_check("Code-style body", code_body)

    print(f"\n{'='*40}")
    print(f"Browser body: {'ERROR' if err1 else 'OK'}")
    print(f"Code body:    {'ERROR' if err2 else 'OK'}")

    if err1 != err2:
        print(">>> 两者结果不同！差异在请求 body 中")
    else:
        print(">>> 两者结果相同，差异可能在 headers/URL/签名")


if __name__ == "__main__":
    asyncio.run(main())
