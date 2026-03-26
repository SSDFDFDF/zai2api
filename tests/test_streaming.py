import asyncio
from app.core.response_handler import ResponseHandler

async def test_response_handler():
    handler = ResponseHandler()

    # 模拟上游带有 <details> 残留的跨 Chunk 场景
    chunks = [
        'data: {"type": "chat:completion", "data": {"phase": "thinking", "delta_content": "<details type=\\"reasoning\\" done=\\"false\\">\\n> 正在思考"}}\n\n',
        'data: {"type": "chat:completion", "data": {"phase": "thinking", "delta_content": "如何调用工具"}}\n\n',
        'data: {"type": "chat:completion", "data": {"phase": "answer", "delta_content": "决定调用工具\\n<details type=\\"reasoning\\" done=\\"true\\">\\n> 正在思考"}}\n\n',
        'data: {"type": "chat:completion", "data": {"phase": "answer", "delta_content": "如何调用工具\\n</details>\\nHello World"}}\n\n',
    ]

    class MockResponse:
        def __init__(self, c):
            self.c = c
        async def aiter_lines(self):
            for line in self.c:
                yield line

    response = MockResponse(chunks)

    print("Testing response handler with simulated chunks:")
    async for chunk in handler.handle_stream_response(response, "test_id", "glm-5"):
        print(f"OUTPUT YIELDED: {repr(chunk)}")

asyncio.run(test_response_handler())
