#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio

from app.core.response_handler import ResponseHandler


READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
            },
            "required": ["file_path"],
        },
    },
}


class MockResponse:
    def __init__(self, chunks):
        self.chunks = chunks

    async def aiter_lines(self):
        for line in self.chunks:
            yield line


def test_non_stream_tool_turn_drops_text_when_tool_calls_exist():
    handler = ResponseHandler()
    response = MockResponse(
        [
            'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"让我先检查一下"}}',
            'data: {"type":"chat:completion","data":{"phase":"answer","tool_calls":[{"id":"call_read_1","type":"function","function":{"name":"Read","arguments":"{\\"file_path\\":\\"/tmp/a.txt\\"}"}}]}}',
            'data: {"type":"chat:completion","data":{"phase":"done","done":true}}',
        ]
    )

    result = asyncio.run(
        handler.handle_non_stream_response(
            response,
            "chat_non_stream_tool",
            "GLM-5-Thinking",
            tools_defs=[READ_TOOL],
            tool_strategy="native",
        )
    )

    message = result["choices"][0]["message"]
    assert message["content"] == ""
    assert "reasoning_content" not in message
    assert message["tool_calls"][0]["function"]["name"] == "Read"
    assert result["choices"][0]["finish_reason"] == "tool_calls"


def test_non_stream_text_turn_keeps_text_when_no_tool_call_exists():
    handler = ResponseHandler()
    response = MockResponse(
        [
            'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"纯文本结果"}}',
            'data: {"type":"chat:completion","data":{"phase":"done","done":true}}',
        ]
    )

    result = asyncio.run(
        handler.handle_non_stream_response(
            response,
            "chat_non_stream_text",
            "GLM-5-Thinking",
            tools_defs=[READ_TOOL],
            tool_strategy="native",
        )
    )

    message = result["choices"][0]["message"]
    assert message["content"] == "纯文本结果"
    assert "tool_calls" not in message
    assert result["choices"][0]["finish_reason"] == "stop"
