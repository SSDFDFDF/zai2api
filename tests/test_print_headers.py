#!/usr/bin/env python
"""打印实际发送的 headers 和 URL，用于和浏览器对比。"""

import asyncio
import json
import sys
import uuid

sys.path.insert(0, ".")

from app.core.request_signing import sign_request
from app.utils.jwt_utils import extract_user_id_from_token
from app.core.config import settings
from urllib.parse import urlparse, parse_qs

TOKEN = (
    "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpZCI6IjRjNWE4OTY4LTY5Y2QtNGQyNS1iY2ZmLThhY2RiODY3ZjZiYyIs"
    "ImVtYWlsIjoiZ2FicmllbF9jYUBmb2RkcnQuYmFyIn0."
    "1l-1mbas9_qaFFmJOkloJJb5OT6Wm_PiSAUX-rSYiD5ePA7MzwMiLq7k16AZ1mO257N-DCYP_9SH5M7K0HFTTw"
)


async def main():
    user_id = extract_user_id_from_token(TOKEN)
    chat_id = str(uuid.uuid4())

    signed_url, headers, fe_ver = await sign_request(
        api_endpoint=settings.API_ENDPOINT,
        user_id=user_id,
        last_user_text="hi",
        chat_id=chat_id,
        token=TOKEN,
    )

    parsed = urlparse(signed_url)
    params = parse_qs(parsed.query)

    print("=== URL ===")
    print(f"  Base: {parsed.scheme}://{parsed.netloc}{parsed.path}")
    print(f"  Query params:")
    for k, v in sorted(params.items()):
        val = v[0] if len(v) == 1 else v
        if k == "token":
            val = val[:20] + "..." if isinstance(val, str) else val
        print(f"    {k}: {val}")

    print(f"\n=== Headers ===")
    for k, v in sorted(headers.items()):
        if k == "Authorization":
            v = v[:30] + "..."
        elif k == "X-Signature":
            v = v[:20] + "..."
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
