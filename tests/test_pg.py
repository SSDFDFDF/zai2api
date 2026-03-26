import asyncio
from app.services.request_log_dao import get_request_log_dao, init_request_log_dao
from app.core.config import settings

async def main():
    print(f"DATABASE_URL: {settings.DATABASE_URL}")
    dao = init_request_log_dao()
    
    # Initialize DB schema if it's PG
    if hasattr(dao, 'init_db'):
        await dao.init_db()
    
    count = await dao.count_logs()
    print(f"Total count: {count}")
    
    logs = await dao.get_recent_logs(limit=2)
    print(f"Recent logs: {logs}")
    
    if logs:
        from app.admin.api import _is_redundant_source, _humanize_protocol
        row = logs[0]
        # Simulate api.py handling
        timestamp = (
            row.get("timestamp")
            or row.get("created_at")
        )
        print(f"Timestamp parsed: {repr(timestamp)}")
        
if __name__ == "__main__":
    asyncio.run(main())
