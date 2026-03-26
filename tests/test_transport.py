import asyncio
import httpx
from app.core.headers import build_dynamic_headers
async def main():
    transport = httpx.AsyncHTTPTransport()
    headers = build_dynamic_headers("1.0.0")
    
    # Client 1
    async with httpx.AsyncClient(transport=transport) as client1:
        r1 = await client1.get("https://chat.z.ai/api/v1/auths/", headers=headers)
        print("C1:", r1.json().get('id'))
        
    # Client 2 (using same transport, meaning same connection pool)
    async with httpx.AsyncClient(transport=transport) as client2:
        r2 = await client2.get("https://chat.z.ai/api/v1/auths/", headers=headers)
        print("C2:", r2.json().get('id'))

asyncio.run(main())
