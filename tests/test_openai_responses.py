import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import openai_responses
from app.core.openai_responses_request_adapter import (
    responses_request_to_openai_request,
)
from app.core.openai_responses_serializer import (
    openai_chat_response_to_openai_response,
    openai_chat_stream_to_openai_responses_stream,
)
from app.models.openai_responses import OpenAIResponsesRequest
from app.core.config import settings


def _parse_sse_chunk(chunk: str) -> tuple[str | None, dict]:
    event_type = None
    payload = {}
    for line in chunk.strip().split("\n"):
        if line.startswith("event: "):
            event_type = line[7:].strip()
        elif line.startswith("data: "):
            payload = json.loads(line[6:].strip())
    return event_type, payload


def test_responses_request_to_openai_request_converts_tools_and_tool_results():
    body = OpenAIResponsesRequest(
        model="glm-5",
        instructions="You are a coding assistant.",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "list_files",
                "arguments": {"path": "."},
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": {"files": ["a.py"]},
            },
        ],
        tools=[
            {
                "type": "function",
                "name": "list_files",
                "description": "List files",
                "parameters": {"type": "object"},
            }
        ],
        tool_choice={"type": "function", "name": "list_files"},
        max_output_tokens=123,
        reasoning={"effort": "high"},
    )

    request = responses_request_to_openai_request(body)

    assert request.model == "glm-5"
    assert request.max_tokens == 123
    assert request.enable_thinking is True
    assert [message.role for message in request.messages] == [
        "developer",
        "user",
        "assistant",
        "tool",
    ]
    assert request.tools[0]["function"]["name"] == "list_files"
    assert request.tool_choice["function"]["name"] == "list_files"
    assert request.messages[2].tool_calls[0]["id"] == "call_123"
    assert request.messages[3].tool_call_id == "call_123"


def test_non_stream_serializer_prefers_tool_turn_when_tool_calls_present():
    chat_response = {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1730000000,
        "model": "glm-5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check that.",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "list_files",
                                "arguments": "{\"path\":\".\"}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    response_payload = openai_chat_response_to_openai_response(chat_response)

    assert response_payload["object"] == "response"
    assert response_payload["status"] == "completed"
    assert [item["type"] for item in response_payload["output"]] == ["function_call"]
    assert response_payload["output"][0]["call_id"] == "call_abc"
    assert response_payload["usage"]["input_tokens"] == 10
    assert response_payload["usage"]["output_tokens"] == 5


def test_non_stream_serializer_drops_message_and_reasoning_for_tool_turn():
    chat_response = {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1730000000,
        "model": "glm-5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I will check that.",
                    "reasoning_content": "Need to inspect files first.",
                    "tool_calls": [
                        {
                            "id": "call_reasoning_drop",
                            "type": "function",
                            "function": {
                                "name": "list_files",
                                "arguments": "{\"path\":\".\"}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    response_payload = openai_chat_response_to_openai_response(chat_response)

    assert [item["type"] for item in response_payload["output"]] == ["function_call"]
    assert response_payload["output"][0]["call_id"] == "call_reasoning_drop"


@pytest.mark.asyncio
async def test_stream_serializer_buffers_text_and_commits_tool_turn():
    async def chat_stream():
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
            '"model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
            '"model":"glm-5","choices":[{"index":0,"delta":{"content":"Let me check. "},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
            '"model":"glm-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"list_files","arguments":"{\\"path\\":\\".\\"}"}}]},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
            '"model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}\n\n'
        )
        yield "data: [DONE]\n\n"

    chunks = []
    async for chunk in openai_chat_stream_to_openai_responses_stream(
        chat_stream(),
        model="glm-5",
        has_tools=True,
    ):
        chunks.append(chunk)

    parsed = [_parse_sse_chunk(chunk) for chunk in chunks]
    event_names = [event for event, _ in parsed]

    assert "response.output_text.delta" not in event_names
    assert "response.function_call_arguments.done" in event_names
    assert event_names[0] == "response.created"
    assert event_names[-1] == "response.completed"

    completed_payload = parsed[-1][1]["response"]
    assert [item["type"] for item in completed_payload["output"]] == ["function_call"]
    assert completed_payload["output"][0]["call_id"] == "call_abc"
    assert completed_payload["usage"]["input_tokens"] == 10

    added_item_types = [
        payload["item"]["type"]
        for event, payload in parsed
        if event == "response.output_item.added"
    ]
    done_item_types = [
        payload["item"]["type"]
        for event, payload in parsed
        if event == "response.output_item.done"
    ]
    assert added_item_types == ["function_call"]
    assert done_item_types == ["function_call"]


@pytest.mark.asyncio
async def test_stream_serializer_drops_reasoning_for_tool_turn():
    async def chat_stream():
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
            '"model":"glm-5","choices":[{"index":0,"delta":{"reasoning_content":"think first"},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
            '"model":"glm-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_reasoning","type":"function","function":{"name":"list_files","arguments":"{\\"path\\":\\".\\"}"}}]},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
            '"model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}\n\n'
        )
        yield "data: [DONE]\n\n"

    chunks = []
    async for chunk in openai_chat_stream_to_openai_responses_stream(
        chat_stream(),
        model="glm-5",
        has_tools=True,
    ):
        chunks.append(chunk)

    parsed = [_parse_sse_chunk(chunk) for chunk in chunks]
    added_item_types = [
        payload["item"]["type"]
        for event, payload in parsed
        if event == "response.output_item.added"
    ]

    assert "response.output_text.delta" not in [event for event, _ in parsed]
    assert added_item_types == ["function_call"]
    completed_payload = parsed[-1][1]["response"]
    assert [item["type"] for item in completed_payload["output"]] == ["function_call"]



@pytest.mark.asyncio
async def test_stream_serializer_emits_reasoning_and_message_for_text_turn():
    async def chat_stream():
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
            '"model":"glm-5","choices":[{"index":0,"delta":{"reasoning_content":"think first"},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
            '"model":"glm-5","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
            '"model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}\n\n'
        )
        yield "data: [DONE]\n\n"

    chunks = []
    async for chunk in openai_chat_stream_to_openai_responses_stream(
        chat_stream(),
        model="glm-5",
        has_tools=True,
    ):
        chunks.append(chunk)

    parsed = [_parse_sse_chunk(chunk) for chunk in chunks]
    completed_payload = parsed[-1][1]["response"]
    output_types = [item["type"] for item in completed_payload["output"]]

    assert output_types == ["message", "reasoning"]
    assert completed_payload["output"][0]["content"][0]["text"] == "answer"
    assert completed_payload["output"][1]["summary"][0]["text"] == "think first"


def test_responses_route_non_stream(monkeypatch):
    app = FastAPI()
    app.include_router(openai_responses.router)

    class FakeClient:
        async def chat_completion(self, request, http_request=None):
            assert request.messages[0].role == "developer"
            assert request.messages[1].role == "user"
            return (
                {
                    "id": "chatcmpl_test",
                    "object": "chat.completion",
                    "created": 1730000000,
                    "model": "glm-5",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "pong",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "total_tokens": 9,
                    },
                },
                "upstream_token",
            )

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(openai_responses, "get_upstream_client", lambda: FakeClient())
    monkeypatch.setattr(openai_responses, "write_request_log", _noop)
    monkeypatch.setattr(settings, "SKIP_AUTH_TOKEN", True, raising=False)

    client = TestClient(app)
    response = client.post(
        "/v1/responses",
        json={
            "model": "glm-5",
            "instructions": "You are helpful.",
            "input": "ping",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["output"][0]["type"] == "message"
    assert payload["output"][0]["content"][0]["text"] == "pong"


def test_responses_route_stream(monkeypatch):
    app = FastAPI()
    app.include_router(openai_responses.router)

    class FakeClient:
        async def chat_completion(self, request, http_request=None):
            async def _stream():
                yield (
                    'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
                    '"model":"glm-5","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                )
                yield (
                    'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
                    '"model":"glm-5","choices":[{"index":0,"delta":{"content":"pong"},"finish_reason":null}]}\n\n'
                )
                yield (
                    'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1730000000,'
                    '"model":"glm-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                    '"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}\n\n'
                )
                yield "data: [DONE]\n\n"

            return _stream(), "upstream_token"

    async def _passthrough(stream, **kwargs):
        async for chunk in stream:
            yield chunk

    monkeypatch.setattr(openai_responses, "get_upstream_client", lambda: FakeClient())
    monkeypatch.setattr(
        openai_responses,
        "wrap_openai_responses_stream_with_logging",
        _passthrough,
    )
    monkeypatch.setattr(settings, "SKIP_AUTH_TOKEN", True, raising=False)

    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "glm-5",
            "input": "ping",
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: response.created" in body
    assert "event: response.output_text.delta" in body
    assert "event: response.completed" in body
