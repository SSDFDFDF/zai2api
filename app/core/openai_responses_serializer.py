#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Serialize internal chat-compatible responses into OpenAI Responses format."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.turn_engine import TurnEngine, TurnEngineAction, TurnEngineConfig
from app.utils.logger import get_logger
from app.utils.request_logging import extract_openai_usage

logger = get_logger()


def _new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _new_reasoning_id() -> str:
    return f"rs_{uuid.uuid4().hex[:24]}"


def _new_function_item_id(call_id: str) -> str:
    normalized = "".join(ch for ch in call_id if ch.isalnum() or ch in ("_", "-"))
    normalized = normalized[:40] or uuid.uuid4().hex[:24]
    return f"fc_{normalized}"


def _stringify_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments or {}, ensure_ascii=False)
    except Exception:
        return "{}"


def _normalize_usage_from_chat(chat_response: Dict[str, Any]) -> Dict[str, Any]:
    usage = extract_openai_usage(chat_response)
    return {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "input_tokens_details": {
            "cached_tokens": usage["cache_read_tokens"],
        },
    }


def extract_openai_responses_usage(response: Dict[str, Any]) -> Dict[str, int]:
    usage = response.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    input_details = usage.get("input_tokens_details") or {}
    cache_read_tokens = int(input_details.get("cached_tokens") or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": 0,
        "cache_read_tokens": cache_read_tokens,
        "total_tokens": total_tokens,
    }


def _base_response_object(
    response_id: str,
    model: str,
    *,
    output: Optional[List[Dict[str, Any]]] = None,
    status: str,
    created_at: Optional[int] = None,
    usage: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parallel_tool_calls: Optional[bool] = None,
) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at if created_at is not None else int(time.time()),
        "status": status,
        "error": error,
        "incomplete_details": None,
        "model": model,
        "output": output or [],
        "usage": usage,
        "metadata": metadata,
        "parallel_tool_calls": bool(parallel_tool_calls),
    }
    return response


def _build_output_text_part(text: str) -> Dict[str, Any]:
    return {
        "type": "output_text",
        "text": text,
        "annotations": [],
    }


def _build_message_item(text: str, *, item_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": item_id or _new_message_id(),
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [_build_output_text_part(text)],
    }


def _build_reasoning_item(summary_text: str) -> Dict[str, Any]:
    return {
        "id": _new_reasoning_id(),
        "type": "reasoning",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": summary_text}],
    }


def _build_function_call_item(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    function_data = (
        tool_call.get("function")
        if isinstance(tool_call.get("function"), dict)
        else {}
    )
    call_id = str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}")
    return {
        "id": _new_function_item_id(call_id),
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": str(function_data.get("name") or ""),
        "arguments": _stringify_arguments(function_data.get("arguments")),
    }


def openai_chat_response_to_openai_response(
    chat_response: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    parallel_tool_calls: Optional[bool] = None,
    strict_tool_turn: bool = True,
) -> Dict[str, Any]:
    choice = ((chat_response.get("choices") or [{}])[0]) if isinstance(chat_response, dict) else {}
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    reasoning_content = str(message.get("reasoning_content") or "")
    tool_calls = message.get("tool_calls") or []

    output: List[Dict[str, Any]] = []
    if strict_tool_turn and tool_calls:
        if content.strip():
            logger.warning(
                "[responses] dropping assistant text because tool_calls are present in strict mode: {} chars",
                len(content.strip()),
            )
        if reasoning_content.strip():
            logger.warning(
                "[responses] dropping reasoning because tool_calls are present in strict mode: {} chars",
                len(reasoning_content.strip()),
            )
        for tool_call in tool_calls:
            output.append(_build_function_call_item(tool_call))
    else:
        if reasoning_content.strip():
            output.append(_build_reasoning_item(reasoning_content.strip()))

        text = content.strip()
        if not text and reasoning_content.strip() and not tool_calls:
            text = reasoning_content.strip()
        if text:
            output.append(_build_message_item(text))

        for tool_call in tool_calls:
            output.append(_build_function_call_item(tool_call))

    usage = _normalize_usage_from_chat(chat_response)
    response_id = str(chat_response.get("id") or _new_response_id())
    model = str(chat_response.get("model") or "")

    return _base_response_object(
        response_id,
        model,
        output=output,
        status="completed",
        created_at=int(chat_response.get("created") or time.time()),
        usage=usage,
        metadata=metadata,
        parallel_tool_calls=parallel_tool_calls,
    )


def _format_responses_sse(event_type: str, payload: Dict[str, Any]) -> str:
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


def _event_payload(event_type: str, **kwargs: Any) -> Dict[str, Any]:
    payload = {
        "type": event_type,
        "event_id": f"evt_{uuid.uuid4().hex}",
    }
    payload.update(kwargs)
    return payload


@dataclass
class _ResponsesStreamState:
    response_id: str
    model: str
    created_at: int
    metadata: Optional[Dict[str, Any]]
    parallel_tool_calls: Optional[bool]
    engine: TurnEngine

    pending_reasoning_parts: List[str] = field(default_factory=list)
    output_items: List[Dict[str, Any]] = field(default_factory=list)
    message_item: Optional[Dict[str, Any]] = None
    message_output_index: Optional[int] = None
    message_done: bool = False
    usage: Dict[str, Any] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
        }
    )

    def response_object(self, *, status: str, error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return _base_response_object(
            self.response_id,
            self.model,
            output=self.output_items,
            status=status,
            created_at=self.created_at,
            usage=self.usage if status == "completed" else None,
            error=error,
            metadata=self.metadata,
            parallel_tool_calls=self.parallel_tool_calls,
        )

    @property
    def pending_reasoning(self) -> str:
        return "".join(self.pending_reasoning_parts).strip()

    @property
    def turn_state(self) -> str:
        return self.engine.state

    @property
    def pending_text(self) -> str:
        return self.engine.pending_text


def _message_item_started(item: Dict[str, Any]) -> Dict[str, Any]:
    started = dict(item)
    started["status"] = "in_progress"
    started["content"] = []
    return started


def _function_call_item_started(item: Dict[str, Any]) -> Dict[str, Any]:
    started = dict(item)
    started["status"] = "in_progress"
    return started


def _update_usage_from_chat_chunk(state: _ResponsesStreamState, payload: Dict[str, Any]) -> None:
    usage_payload = payload.get("usage")
    if not isinstance(usage_payload, dict):
        return
    chat_like = {"usage": usage_payload}
    state.usage = _normalize_usage_from_chat(chat_like)


def _start_message_item(state: _ResponsesStreamState) -> List[str]:
    if state.message_item is not None:
        return []

    message_item = _build_message_item("", item_id=_new_message_id())
    state.message_item = message_item
    state.message_output_index = len(state.output_items)
    state.output_items.append(message_item)

    output_index = state.message_output_index
    assert output_index is not None

    return [
        _format_responses_sse(
            "response.output_item.added",
            _event_payload(
                "response.output_item.added",
                response_id=state.response_id,
                output_index=output_index,
                item=_message_item_started(message_item),
            ),
        ),
        _format_responses_sse(
            "response.content_part.added",
            _event_payload(
                "response.content_part.added",
                response_id=state.response_id,
                output_index=output_index,
                content_index=0,
                item_id=message_item["id"],
                part=_build_output_text_part(""),
            ),
        ),
    ]


def _append_text_delta(state: _ResponsesStreamState, text: str) -> List[str]:
    if not text:
        return []

    output: List[str] = []
    output.extend(_start_message_item(state))
    assert state.message_item is not None
    assert state.message_output_index is not None

    current_text = str(state.message_item["content"][0]["text"] or "")
    state.message_item["content"][0]["text"] = current_text + text

    output.append(
        _format_responses_sse(
            "response.output_text.delta",
            _event_payload(
                "response.output_text.delta",
                response_id=state.response_id,
                output_index=state.message_output_index,
                content_index=0,
                item_id=state.message_item["id"],
                delta=text,
            ),
        )
    )
    return output


def _apply_turn_actions(
    state: _ResponsesStreamState,
    actions: List[TurnEngineAction],
) -> List[str]:
    output: List[str] = []
    for action in actions:
        if action.kind == "emit_text":
            output.extend(_append_text_delta(state, action.text))
        elif action.kind == "emit_tool_calls":
            output.extend(_render_function_call_items(state, action.tool_calls))
        elif action.kind in (
            "drop_pending_text",
            "ignore_tool_calls",
            "ignore_text",
        ):
            logger.debug(
                "[responses] turn action {} ({})",
                action.kind,
                action.reason,
            )
    return output


def _finish_message_item(state: _ResponsesStreamState) -> List[str]:
    if (
        state.message_done
        or state.message_item is None
        or state.message_output_index is None
    ):
        return []

    item = state.message_item
    text_part = item["content"][0]
    output_index = state.message_output_index

    state.message_done = True

    return [
        _format_responses_sse(
            "response.output_text.done",
            _event_payload(
                "response.output_text.done",
                response_id=state.response_id,
                output_index=output_index,
                content_index=0,
                item_id=item["id"],
                text=text_part["text"],
            ),
        ),
        _format_responses_sse(
            "response.content_part.done",
            _event_payload(
                "response.content_part.done",
                response_id=state.response_id,
                output_index=output_index,
                content_index=0,
                item_id=item["id"],
                part=text_part,
            ),
        ),
        _format_responses_sse(
            "response.output_item.done",
            _event_payload(
                "response.output_item.done",
                response_id=state.response_id,
                output_index=output_index,
                item=item,
            ),
        ),
    ]


def _emit_reasoning_item(state: _ResponsesStreamState) -> List[str]:
    if state.turn_state == "tool_turn":
        state.pending_reasoning_parts = []
        return []

    reasoning = state.pending_reasoning
    state.pending_reasoning_parts = []
    if not reasoning:
        return []

    item = _build_reasoning_item(reasoning)
    output_index = len(state.output_items)
    state.output_items.append(item)
    return [
        _format_responses_sse(
            "response.output_item.added",
            _event_payload(
                "response.output_item.added",
                response_id=state.response_id,
                output_index=output_index,
                item=_message_item_started(item) if item["type"] == "message" else dict(item, status="in_progress"),
            ),
        ),
        _format_responses_sse(
            "response.output_item.done",
            _event_payload(
                "response.output_item.done",
                response_id=state.response_id,
                output_index=output_index,
                item=item,
            ),
        ),
    ]


def _render_function_call_items(
    state: _ResponsesStreamState,
    tool_calls: List[Dict[str, Any]],
) -> List[str]:
    if not tool_calls:
        return []

    output: List[str] = []
    for tool_call in tool_calls:
        item = _build_function_call_item(tool_call)
        output_index = len(state.output_items)
        state.output_items.append(item)

        output.append(
            _format_responses_sse(
                "response.output_item.added",
                _event_payload(
                    "response.output_item.added",
                    response_id=state.response_id,
                    output_index=output_index,
                    item=_function_call_item_started(item),
                ),
            )
        )

        if item["arguments"]:
            output.append(
                _format_responses_sse(
                    "response.function_call_arguments.delta",
                    _event_payload(
                        "response.function_call_arguments.delta",
                        response_id=state.response_id,
                        output_index=output_index,
                        item_id=item["id"],
                        delta=item["arguments"],
                    ),
                )
            )

        output.append(
            _format_responses_sse(
                "response.function_call_arguments.done",
                _event_payload(
                    "response.function_call_arguments.done",
                    response_id=state.response_id,
                    output_index=output_index,
                    item_id=item["id"],
                    arguments=item["arguments"],
                    item=item,
                ),
            )
        )
        output.append(
            _format_responses_sse(
                "response.output_item.done",
                _event_payload(
                    "response.output_item.done",
                    response_id=state.response_id,
                    output_index=output_index,
                    item=item,
                ),
            )
        )

    return output


async def openai_chat_stream_to_openai_responses_stream(
    chat_stream: AsyncGenerator[str, None],
    *,
    model: str,
    has_tools: bool,
    metadata: Optional[Dict[str, Any]] = None,
    parallel_tool_calls: Optional[bool] = None,
    strict_tool_turn: bool = True,
) -> AsyncGenerator[str, None]:
    response_id = _new_response_id()
    created_at = int(time.time())
    engine = TurnEngine(
        TurnEngineConfig(
            has_tools=has_tools,
            strict_tool_turn=strict_tool_turn,
            debug_label=f"responses:{response_id[:16]}",
        )
    )
    state = _ResponsesStreamState(
        response_id=response_id,
        model=model,
        created_at=created_at,
        metadata=metadata,
        parallel_tool_calls=parallel_tool_calls,
        engine=engine,
    )

    yield _format_responses_sse(
        "response.created",
        _event_payload(
            "response.created",
            response=state.response_object(status="in_progress"),
        ),
    )

    try:
        async for chunk in chat_stream:
            if not chunk.startswith("data: "):
                continue

            payload_text = chunk[6:].strip()
            if not payload_text or payload_text == "[DONE]":
                continue

            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                logger.debug(
                    "[responses] skip non-json chat chunk: {}",
                    payload_text[:120],
                )
                continue

            if not isinstance(payload, dict):
                continue

            if "error" in payload:
                error = payload.get("error") or {}
                error_payload = {
                    "message": str(error.get("message") or "Unknown stream error"),
                    "code": error.get("code"),
                    "type": error.get("type") or "stream_error",
                }
                yield _format_responses_sse(
                    "response.failed",
                    _event_payload(
                        "response.failed",
                        response=state.response_object(
                            status="failed",
                            error=error_payload,
                        ),
                    ),
                )
                return

            _update_usage_from_chat_chunk(state, payload)

            choice = ((payload.get("choices") or [{}])[0]) if isinstance(payload, dict) else {}
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            reasoning_delta = str(delta.get("reasoning_content") or "")
            if reasoning_delta:
                state.pending_reasoning_parts.append(reasoning_delta)

            raw_tool_calls = delta.get("tool_calls")
            tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
            if tool_calls:
                if state.pending_reasoning_parts:
                    logger.debug(
                        "[responses] drop buffered reasoning before tool_turn: {} chars",
                        len("".join(state.pending_reasoning_parts).strip()),
                    )
                state.pending_reasoning_parts = []
                for event in _apply_turn_actions(
                    state,
                    state.engine.commit_tool_calls(tool_calls),
                ):
                    yield event
                continue

            content_delta = str(delta.get("content") or "")
            if content_delta:
                for event in _apply_turn_actions(
                    state,
                    state.engine.buffer_text(content_delta),
                ):
                    yield event

            if finish_reason == "stop":
                if state.turn_state == "undecided":
                    if not state.pending_text.strip() and state.pending_reasoning:
                        state.engine.buffer_text(state.pending_reasoning)
                        state.pending_reasoning_parts = []
                    for event in _apply_turn_actions(
                        state,
                        state.engine.flush_text(force=True, reason="finish_reason_stop"),
                    ):
                        yield event
                if state.pending_reasoning and state.turn_state != "tool_turn":
                    for event in _emit_reasoning_item(state):
                        yield event
                if state.turn_state == "text_turn":
                    for event in _finish_message_item(state):
                        yield event

        if state.turn_state == "undecided":
            if not state.pending_text.strip() and state.pending_reasoning:
                state.engine.buffer_text(state.pending_reasoning)
                state.pending_reasoning_parts = []
            for event in _apply_turn_actions(
                state,
                state.engine.flush_text(force=True, reason="stream_end"),
            ):
                yield event
        if state.pending_reasoning and state.turn_state != "tool_turn":
            for event in _emit_reasoning_item(state):
                yield event

        if state.turn_state == "text_turn":
            for event in _finish_message_item(state):
                yield event
        state.engine.mark_completed()

        yield _format_responses_sse(
            "response.completed",
            _event_payload(
                "response.completed",
                response=state.response_object(status="completed"),
            ),
        )
    except Exception as exc:
        logger.error("[responses] stream translation failed: {}", exc)
        yield _format_responses_sse(
            "response.failed",
            _event_payload(
                "response.failed",
                response=state.response_object(
                    status="failed",
                    error={
                        "message": str(exc),
                        "type": "stream_error",
                        "code": 500,
                    },
                ),
            ),
        )
