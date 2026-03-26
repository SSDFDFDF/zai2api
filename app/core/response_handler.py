#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Minimal SSE response handler.

Converts upstream SSE responses to OpenAI-compatible format.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from app.core.openai_compat import (
    create_openai_chunk,
    create_openai_response_with_reasoning,
    format_sse_chunk,
    format_sse_done,
    get_error_message,
    handle_error,
)
from app.utils.logger import logger


# ------------------------------------------------------------------
# UpstreamChunkEvent
# ------------------------------------------------------------------


@dataclass
class UpstreamChunkEvent:
    delta_content: str = ""
    edit_content: str = ""
    edit_index: Optional[int] = None
    phase: str = ""
    done: bool = False
    usage: Optional[Dict] = None
    error: Optional[Dict] = None
    data: Dict = field(default_factory=dict)
    chunk_type: str = ""
    raw_chunk: Optional[Dict] = None


def _parse_chunk_event(chunk: dict) -> Optional[UpstreamChunkEvent]:
    if not isinstance(chunk, dict):
        return None

    chunk_type = str(chunk.get("type", ""))
    data: Any = chunk.get("data", {}) if chunk_type == "chat:completion" else chunk
    if not isinstance(data, dict):
        return None

    error = None
    if chunk.get("error"):
        err = chunk["error"]
        error = err if isinstance(err, dict) else {"detail": str(err), "code": "UNKNOWN"}
    elif data.get("error"):
        err = data["error"]
        error = err if isinstance(err, dict) else {"detail": str(err), "code": "UNKNOWN"}
    elif isinstance(data.get("inner"), dict) and data["inner"].get("error"):
        err = data["inner"]["error"]
        error = err if isinstance(err, dict) else {"detail": str(err), "code": "UNKNOWN"}

    return UpstreamChunkEvent(
        delta_content=str(data.get("delta_content", "") or ""),
        edit_content=str(data.get("edit_content", "") or ""),
        edit_index=data.get("edit_index") if isinstance(data.get("edit_index"), int) else None,
        phase=str(data.get("phase", "") or ""),
        done=bool(data.get("done") is True),
        usage=(
            data.get("usage") if isinstance(data.get("usage"), dict)
            else chunk.get("usage") if isinstance(chunk.get("usage"), dict)
            else None
        ),
        error=error,
        data=data,
        chunk_type=chunk_type,
        raw_chunk=chunk,
    )


# ------------------------------------------------------------------
# StreamContext
# ------------------------------------------------------------------


@dataclass
class StreamContext:
    chat_id: str
    model: str
    stream_id: Optional[str] = None
    created_at: Optional[int] = None
    buffered_content: str = ""
    usage_info: Dict[str, Any] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    has_sent_role: bool = False
    finished: bool = False
    line_count: int = 0
    last_phase: Optional[str] = None

    def ensure_stream_id(self, chunk_data: Optional[Dict] = None) -> str:
        if self.stream_id is None:
            upstream_id = chunk_data.get("id") if isinstance(chunk_data, dict) else None
            self.stream_id = upstream_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
            self.created_at = int(time.time())
        return self.stream_id


# ------------------------------------------------------------------
# ResponseHandler
# ------------------------------------------------------------------


class ResponseHandler:

    # SSE helpers

    def _emit_sse(self, ctx: StreamContext, delta: Dict[str, Any]) -> List[str]:
        output: List[str] = []
        role_sse = self._ensure_role_sse(ctx)
        if role_sse:
            output.append(role_sse)
        output.append(format_sse_chunk(
            create_openai_chunk(ctx.ensure_stream_id(), ctx.model, delta, created=ctx.created_at)
        ))
        return output

    def _terminal_sse(self, ctx: StreamContext, *, finish_reason: str) -> List[str]:
        chunk = create_openai_chunk(ctx.ensure_stream_id(), ctx.model, {}, finish_reason, created=ctx.created_at)
        chunk["usage"] = ctx.usage_info
        return [format_sse_chunk(chunk), format_sse_done()]

    def _ensure_role_sse(self, ctx: StreamContext) -> Optional[str]:
        if ctx.has_sent_role:
            return None
        ctx.has_sent_role = True
        return format_sse_chunk(
            create_openai_chunk(ctx.ensure_stream_id(), ctx.model, {"role": "assistant"}, created=ctx.created_at)
        )

    # SSE line parsing

    @staticmethod
    def _extract_sse_payload(line: str) -> Tuple[str, str]:
        if not line:
            return "skip", ""
        if line.startswith("data:"):
            payload = line[5:].strip()
        else:
            stripped = line.strip()
            if not stripped:
                return "skip", ""
            if not stripped.startswith("data:"):
                return "non_data", stripped
            payload = stripped[5:].strip()
        if not payload:
            return "skip", ""
        if payload == "[DONE]" or payload.casefold() == "done":
            return "done", ""
        return "data", payload

    def _parse_sse_line(self, line: str, ctx: StreamContext) -> Tuple[str, Optional[UpstreamChunkEvent]]:
        if not line.strip():
            return "skip", None
        ctx.line_count += 1
        state, chunk_str = self._extract_sse_payload(line)
        if state == "done":
            return "done", None
        if state != "data":
            return "skip", None
        try:
            chunk = json.loads(chunk_str)
        except json.JSONDecodeError:
            return "skip", None
        if not isinstance(chunk, dict):
            return "skip", None
        event = _parse_chunk_event(chunk)
        return ("ok", event) if event else ("skip", None)

    # State update

    def _update_state(self, ctx: StreamContext, event: UpstreamChunkEvent) -> None:
        ctx.ensure_stream_id(event.raw_chunk)
        phase = event.phase
        if phase and phase != ctx.last_phase:
            ctx.last_phase = phase
        if event.usage and event.usage != ctx.usage_info:
            ctx.usage_info = event.usage

    # Content accumulation (needed for edit_index insertion)

    def _accumulate(self, ctx: StreamContext, event: UpstreamChunkEvent) -> str:
        max_len = 50000
        if event.delta_content:
            ctx.buffered_content = (ctx.buffered_content + event.delta_content)[-max_len:]
            return event.delta_content
        if event.edit_content:
            idx = event.edit_index
            if idx is not None and isinstance(idx, int):
                safe = max(0, min(idx, len(ctx.buffered_content)))
                ctx.buffered_content = ctx.buffered_content[:safe] + event.edit_content + ctx.buffered_content[safe:]
                if len(ctx.buffered_content) > max_len:
                    ctx.buffered_content = ctx.buffered_content[-max_len:]
            else:
                ctx.buffered_content = (ctx.buffered_content + event.edit_content)[-max_len:]
            return event.edit_content
        return ""

    # Error check

    def _check_error(self, ctx: StreamContext, event: UpstreamChunkEvent) -> Optional[List[str]]:
        if not event.error:
            return None
        code = event.error.get("code", "UNKNOWN")
        detail = get_error_message(Exception(str(event.error.get("detail", "Unknown upstream error"))))
        if event.done:
            return None
        logger.error("upstream SSE error: code=%s, detail=%s", code, detail)
        output: List[str] = []
        role = self._ensure_role_sse(ctx)
        if role:
            output.append(role)
        output.append(f'data: {json.dumps({"error": {"message": detail, "type": "upstream_error", "code": code}})}\n\n')
        output.append("data: [DONE]\n\n")
        return output

    # Search results

    def _format_search(self, data: Dict[str, Any]) -> str:
        info = data.get("results") or data.get("sources") or data.get("citations")
        if not isinstance(info, list) or not info:
            return ""
        parts = []
        for i, item in enumerate(info, 1):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or f"Result {i}"
            url = item.get("url") or item.get("link")
            if url:
                parts.append(f"[{i}] [{title}]({url})")
        return ("\n\n---\n" + "\n".join(parts)) if parts else ""

    # Phase output

    def _emit_phase(self, ctx: StreamContext, event: UpstreamChunkEvent, text: str) -> List[str]:
        phase = ctx.last_phase

        if phase == "thinking" and event.delta_content:
            return self._emit_sse(ctx, {"reasoning_content": event.delta_content})

        if phase in ("answer", "other") and text:
            return self._emit_sse(ctx, {"content": text})

        if phase == "search" or event.chunk_type == "web_search":
            citation = self._format_search(event.data)
            if citation:
                return self._emit_sse(ctx, {"content": citation})

        return []

    # Non-stream edit helper

    @staticmethod
    def _apply_edit(parts: List[str], edit: str, idx: Optional[int]) -> None:
        if not edit:
            return
        if idx is not None and isinstance(idx, int):
            joined = "".join(parts)
            safe = max(0, min(idx, len(joined)))
            parts[:] = [joined[:safe] + edit + joined[safe:]]
        else:
            parts.append(edit)

    # ==================================================================
    # Stream response (main entry)
    # ==================================================================

    async def handle_stream_response(
        self,
        response: httpx.Response,
        chat_id: str,
        model: str,
        started_at: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        if started_at:
            logger.debug("[stream] start SSE, ttfb: %.3fs", time.perf_counter() - started_at)

        sse_start = time.perf_counter()
        ctx = StreamContext(chat_id=chat_id, model=model)

        try:
            async for line in response.aiter_lines():
                state, event = self._parse_sse_line(line, ctx)
                if state == "done":
                    break
                if state != "ok" or event is None:
                    continue

                err = self._check_error(ctx, event)
                if err:
                    for sse in err:
                        yield sse
                    return

                self._update_state(ctx, event)
                text = self._accumulate(ctx, event)

                for sse in self._emit_phase(ctx, event, text):
                    yield sse

                if event.done:
                    break

            if not ctx.finished:
                role = self._ensure_role_sse(ctx)
                if role:
                    yield role
                for sse in self._terminal_sse(ctx, finish_reason="stop"):
                    yield sse
                ctx.finished = True

        except Exception as e:
            logger.exception("stream error")
            if not ctx.finished:
                yield format_sse_chunk(
                    create_openai_chunk(ctx.ensure_stream_id(), model, {}, "stop", created=ctx.created_at)
                )
                yield format_sse_done()
        finally:
            elapsed = time.perf_counter() - sse_start
            total = f", total: {time.perf_counter() - started_at:.3f}s" if started_at else ""
            logger.debug("[stream] done %s lines, %.3fs%s", ctx.line_count, elapsed, total)

    # ==================================================================
    # Non-stream response
    # ==================================================================

    async def handle_non_stream_response(
        self,
        response: httpx.Response,
        chat_id: str,
        model: str,
    ) -> Dict[str, Any]:
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_phase = None

        try:
            async for line in response.aiter_lines():
                state, payload = self._extract_sse_payload(line)
                if state == "non_data":
                    try:
                        err = json.loads(payload)
                        if isinstance(err, dict) and ("error" in err or "code" in err or "message" in err):
                            msg = (
                                (err.get("error") or {}).get("message")
                                if isinstance(err.get("error"), dict)
                                else err.get("message")
                            ) or "upstream error"
                            return handle_error(Exception(msg), "API response")
                    except Exception:
                        pass
                    continue
                if state != "data":
                    continue

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                event = _parse_chunk_event(chunk)
                if event is None:
                    continue

                phase = event.phase
                if phase and phase != last_phase:
                    last_phase = phase

                if phase == "thinking" and event.delta_content:
                    reasoning_parts.append(event.delta_content)
                elif phase == "answer":
                    if event.delta_content:
                        content_parts.append(event.delta_content)
                    elif event.edit_content:
                        self._apply_edit(content_parts, event.edit_content, event.edit_index)
                elif phase == "other" and event.edit_content:
                    self._apply_edit(content_parts, event.edit_content, event.edit_index)
                elif phase == "search" or event.chunk_type == "web_search":
                    s = self._format_search(event.data)
                    if s:
                        content_parts.append(s)

                if event.usage:
                    usage = event.usage

        except Exception as e:
            logger.exception("non-stream error")
            return handle_error(e, "non-stream aggregation")

        final = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        if not final and reasoning:
            final = reasoning

        return create_openai_response_with_reasoning(chat_id, model, final, reasoning, usage)
