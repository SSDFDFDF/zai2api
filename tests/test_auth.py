import asyncio
import httpx
from app.core.headers import build_dynamic_headers
async def main():
    client = httpx.AsyncClient(follow_redirects=True)
    headers = build_dynamic_headers("1.0.0")
    print("Call 1:")
    r1 = await client.get("https://chat.z.ai/api/v1/auths/", headers=headers)
    print(r1.json())
    print("Call 2 (same client):")
    r2 = await client.get("https://chat.z.ai/api/v1/auths/", headers=headers)
    print(r2.json())
    
    client2 = httpx.AsyncClient(follow_redirects=True)
    print("Call 3 (new client):")
    r3 = await client2.get("https://chat.z.ai/api/v1/auths/", headers=headers)
    print(r3.json())

asyncio.run(main())
