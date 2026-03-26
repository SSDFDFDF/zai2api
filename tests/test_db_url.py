import asyncio
from app.database import get_db_url

url, connect_args = get_db_url("postgresql://user:pass@host/db?sslmode=require&channel_binding=require")
print("URL:", url)
print("ARGS:", connect_args)

