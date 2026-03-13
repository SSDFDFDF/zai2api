import asyncio
import time
import json
import logging
from app.utils.request_logging import wrap_claude_stream_with_logging
from app.utils.request_source import RequestSourceInfo
import app.utils.request_logging as rl

async def mock_write_request_log(*args, **kwargs):
    first_token_time = kwargs.get("first_token_time")
    output_tokens = kwargs.get("output_tokens")
    print(f"first_token_time = {first_token_time}")
    print(f"output_tokens = {output_tokens}")
    assert 0.4 < first_token_time < 0.6, f"Expected ~0.5s, got {first_token_time}"

async def mock_stream():
    # Simulate TTFB delay (e.g., waiting for upstream to start responding)
    await asyncio.sleep(1.0)
    yield 'event: message_start\ndata: {"type": "message_start"}\n\n'
    # Simulate delay between chunks before first content token
    await asyncio.sleep(0.5)
    yield 'event: content_block_start\ndata: {"type": "content_block_start"}\n\n'
    yield 'event: content_block_delta\ndata: {"type": "content_block_delta"}\n\n'
    yield 'event: message_delta\ndata: {"type": "message_delta", "usage": {"output_tokens": 512}}\n\n'
    yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

async def main():
    original_write = rl.write_request_log
    rl.write_request_log = mock_write_request_log
    
    try:
        started_at = time.perf_counter()
        # Also simulate some local processing delay before calling the wrapper
        await asyncio.sleep(0.1)
        source_info = RequestSourceInfo("test", "test", "test", "test", "test")
        stream = wrap_claude_stream_with_logging(
            mock_stream(),
            provider="zai",
            model="test",
            source_info=source_info,
            started_at=started_at,
            input_tokens=10,
        )
        async for chunk in stream:
            pass
    finally:
        rl.write_request_log = original_write

if __name__ == "__main__":
    asyncio.run(main())
