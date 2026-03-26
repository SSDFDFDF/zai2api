import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from app.database import get_db_session, init_db, close_db
from app.models.db_models import RequestLog
from sqlalchemy import select

async def main():
    await init_db()
    async with get_db_session() as session:
        stmt = select(RequestLog).order_by(RequestLog.id.desc()).limit(1)
        res = await session.execute(stmt)
        row = res.first()
        if row:
            print(dict(row._mapping))
        else:
            print("No rows found")
    await close_db()

asyncio.run(main())
