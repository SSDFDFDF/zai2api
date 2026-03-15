#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json

from app.core.response_handler import ResponseHandler
from app.models.schemas import OpenAIRequest


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


def _build_request() -> OpenAIRequest:
    return OpenAIRequest(
        model="GLM-5-Thinking",
        messages=[{"role": "user", "content": "check file"}],
        tools=[READ_TOOL],
        stream=True,
    )


async def _collect_outputs(chunks, *, tool_strategy="native"):
    handler = ResponseHandler()
    response = MockResponse(chunks)
    outputs = []
    async for item in handler.handle_stream_response(
        response,
        "chat_turn_test",
        "GLM-5-Thinking",
        _build_request(),
        {
            "trigger_signal": "",
            "tools": [READ_TOOL],
            "tool_strategy": tool_strategy,
        },
    ):
        outputs.append(item)
    return outputs


def _extract_payloads(outputs):
    payloads = []
    for sse in outputs:
        stripped = sse.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[5:].strip()
        if payload == "[DONE]":
            continue
        payloads.append(json.loads(payload))
    return payloads


def test_native_tool_turn_drops_buffered_text_before_tool_calls():
    chunks = [
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"让我先检查一下"}}',
        'data: {"type":"chat:completion","data":{"phase":"answer","tool_calls":[{"id":"call_read_1","type":"function","function":{"name":"Read","arguments":"{\\"file_path\\":\\"/tmp/a.txt\\"}"}}]}}',
    ]

    outputs = asyncio.run(_collect_outputs(chunks))
    payloads = _extract_payloads(outputs)

    contents = []
    tool_calls = []
    finish_reasons = []
    for payload in payloads:
        choice = payload["choices"][0]
        finish_reasons.append(choice.get("finish_reason"))
        delta = choice.get("delta", {})
        if delta.get("content"):
            contents.append(delta["content"])
        tool_calls.extend(delta.get("tool_calls", []) or [])

    assert contents == []
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "Read"
    assert "tool_calls" in finish_reasons


def test_text_turn_flushes_at_stream_end_when_no_tool_call_happens():
    chunks = [
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"纯文本答案"}}',
        'data: {"type":"chat:completion","data":{"phase":"done","done":true}}',
    ]

    outputs = asyncio.run(_collect_outputs(chunks))
    payloads = _extract_payloads(outputs)

    contents = []
    finish_reason = None
    for payload in payloads:
        choice = payload["choices"][0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta", {})
        if delta.get("content"):
            contents.append(delta["content"])

    assert "".join(contents) == "纯文本答案"
    assert finish_reason == "stop"


def test_late_native_tool_calls_are_ignored_after_text_turn_commit():
    chunks = [
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"第一段"}}',
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"第二段"}}',
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"第三段"}}',
        'data: {"type":"chat:completion","data":{"phase":"answer","tool_calls":[{"id":"call_read_2","type":"function","function":{"name":"Read","arguments":"{\\"file_path\\":\\"/tmp/b.txt\\"}"}}]}}',
        'data: {"type":"chat:completion","data":{"phase":"done","done":true}}',
    ]

    outputs = asyncio.run(_collect_outputs(chunks))
    payloads = _extract_payloads(outputs)

    content_parts = []
    tool_calls = []
    finish_reason = None
    for payload in payloads:
        choice = payload["choices"][0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta", {})
        if delta.get("content"):
            content_parts.append(delta["content"])
        tool_calls.extend(delta.get("tool_calls", []) or [])

    assert "".join(content_parts) == "第一段第二段第三段"
    assert tool_calls == []
    assert finish_reason == "stop"


def test_reasoning_is_dropped_when_native_tool_turn_commits():
    chunks = [
        'data: {"type":"chat:completion","data":{"phase":"thinking","delta_content":"先思考一下"}}',
        'data: {"type":"chat:completion","data":{"phase":"answer","tool_calls":[{"id":"call_read_3","type":"function","function":{"name":"Read","arguments":"{\\"file_path\\":\\"/tmp/c.txt\\"}"}}]}}',
    ]

    outputs = asyncio.run(_collect_outputs(chunks))
    payloads = _extract_payloads(outputs)

    reasoning_parts = []
    tool_calls = []
    for payload in payloads:
        delta = payload["choices"][0].get("delta", {})
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        tool_calls.extend(delta.get("tool_calls", []) or [])

    assert reasoning_parts == []
    assert len(tool_calls) == 1


def test_reasoning_is_preserved_for_text_turn():
    chunks = [
        'data: {"type":"chat:completion","data":{"phase":"thinking","delta_content":"先思考一下"}}',
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"最终答案"}}',
        'data: {"type":"chat:completion","data":{"phase":"done","done":true}}',
    ]

    outputs = asyncio.run(_collect_outputs(chunks))
    payloads = _extract_payloads(outputs)

    reasoning_parts = []
    content_parts = []
    for payload in payloads:
        delta = payload["choices"][0].get("delta", {})
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        if delta.get("content"):
            content_parts.append(delta["content"])

    assert "".join(reasoning_parts) == "先思考一下"
    assert "".join(content_parts) == "最终答案"
