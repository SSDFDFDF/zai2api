import asyncio
import httpx
from app.core.headers import build_dynamic_headers
async def main():
    client = httpx.AsyncClient(follow_redirects=True)
    headers = build_dynamic_headers("1.0.0")
    
    print("Call 1:")
    client.cookies.clear()
    r1 = await client.get("https://chat.z.ai/api/v1/auths/", headers=headers)
    print("C1:", r1.json().get('id'))
    
    print("Call 2 (same client, clear cookies again):")
    client.cookies.clear()
    r2 = await client.get("https://chat.z.ai/api/v1/auths/", headers=headers)
    print("C2:", r2.json().get('id'))

asyncio.run(main())
