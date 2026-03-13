import asyncio
import time
from app.utils.request_logging import wrap_openai_stream_with_logging
from app.utils.request_source import RequestSourceInfo

async def mock_stream():
    yield 'data: {"id":"1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}\n\n'
    await asyncio.sleep(0.1)
    yield 'data: {"id":"1","object":"chat.completion.chunk","choices":[{"delta":{"content":" World"}}]}\n\n'
    await asyncio.sleep(0.1)
    yield 'data: {"id":"1","object":"chat.completion.chunk","choices":[{"delta":{}, "finish_reason": "stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n'
    yield 'data: [DONE]\n\n'

async def main():
    source_info = RequestSourceInfo(endpoint="/v1/chat/completions", source="test", protocol="openai", client_name="test", user_agent="test-agent")
    started_at = time.perf_counter()
    await asyncio.sleep(0.1) # Simulate TTFB overhead
    stream = wrap_openai_stream_with_logging(
        mock_stream(),
        provider="zai",
        model="test-model",
        source_info=source_info,
        started_at=started_at
    )
    async for chunk in stream:
        print(f"Yield: {chunk.strip()}")

if __name__ == "__main__":
    asyncio.run(main())
