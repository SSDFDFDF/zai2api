#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Toolify 工具处理器。

聚合 Toolify 相关完整流程：
- 流式检测/解析 `<function_calls>`
- 流结束时工具提取 (XML 优先, JSON 降级)
- 非流式工具提取 (XML 优先, JSON 降级)
"""

import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from app.core.openai_compat import create_openai_chunk, format_sse_chunk
from app.utils.logger import get_logger
from app.utils.tool_call_handler import (
    StreamingFunctionCallDetector,
    looks_like_complete_function_calls,
    parse_and_extract_tool_calls,
    parse_function_calls_xml,
    validate_parsed_tools,
)

logger = get_logger()


class ToolifyContext(Protocol):
    """Toolify 处理所需的上下文协议。"""

    model: str
    has_sent_role: bool
    trigger_signal: str
    tools_defs: Optional[List[Dict[str, Any]]]
    tool_calls_accum: List[Dict[str, Any]]
    buffered_content: str
    detector: Optional[StreamingFunctionCallDetector]
    last_phase: Optional[str]

    def ensure_stream_id(self, chunk_data: Optional[Dict[str, Any]] = None) -> str:
        ...


class ToolifyHandler:
    """Toolify 统一处理器。"""

    def __init__(
        self,
        emit_func: Optional[Callable[[Any, Dict[str, Any]], List[str]]] = None,
        ensure_role_func: Optional[Callable[[Any], Optional[str]]] = None,
        normalize_tool_calls_func: Optional[
            Callable[[Any, int], List[Dict[str, Any]]]
        ] = None,
    ) -> None:
        # emit_func(ctx, delta_dict) -> List[str]
        self.emit_func = emit_func
        # ensure_role_func(ctx) -> Optional[str]
        self.ensure_role_func = ensure_role_func
        # normalize_tool_calls_func(raw_tool_calls, start_index) -> List[Dict]
        self.normalize_tool_calls_func = normalize_tool_calls_func

    # ------------------------------------------------------------------
    # 流式检测/解析
    # ------------------------------------------------------------------

    def handle_detection(
        self, ctx: ToolifyContext, current_text: str
    ) -> Optional[List[str]]:
        """Toolify 流式工具检测 (非 tool_parsing 模式)。"""
        detector = getattr(ctx, "detector", None)
        phase = getattr(ctx, "last_phase", None)

        if (
            not detector
            or not current_text
            or detector.state == "tool_parsing"
            or phase == "thinking"
        ):
            return None

        is_detected, content_to_yield = detector.process_chunk(current_text)

        if is_detected:
            logger.debug("🔧 流式检测器触发工具调用信号, 切换到解析模式")
            output: List[str] = []
            if content_to_yield and self.emit_func:
                output.extend(self.emit_func(ctx, {"content": content_to_yield}))
            return output

        if content_to_yield and self.emit_func:
            return self.emit_func(ctx, {"content": content_to_yield})
        return []

    def handle_parsing(
        self, ctx: ToolifyContext, current_text: str
    ) -> Optional[List[str]]:
        """累积工具调用 XML 并尝试解析。"""
        detector = getattr(ctx, "detector", None)
        if not detector or detector.state != "tool_parsing" or not current_text:
            return None

        detector.content_buffer += current_text

        if "</function_calls>" not in detector.content_buffer:
            return []

        if not looks_like_complete_function_calls(detector.content_buffer):
            logger.debug("🔧 检测到 </function_calls> 但内容不完整, 继续缓冲")
            return []

        logger.debug("🔧 检测到完整的 </function_calls>, 开始解析...")
        parsed = parse_function_calls_xml(
            detector.content_buffer, getattr(ctx, "trigger_signal", "")
        )
        if not parsed:
            logger.warning("⚠️ 检测到 </function_calls> 但 XML 解析失败, 继续缓冲")
            return []

        validation_err = validate_parsed_tools(parsed, getattr(ctx, "tools_defs", None))
        if validation_err:
            logger.warning("⚠️ 流式工具 Schema 验证失败: {}", validation_err)
            return []

        logger.info("[tools] success detect: {} tools", len(parsed))

        tc_chunks = self._build_tool_call_chunks(ctx, parsed)
        ctx.tool_calls_accum = parsed
        return tc_chunks

    def inject_role_and_chunks(
        self, ctx: ToolifyContext, parsed_chunks: List[str]
    ) -> List[str]:
        """为 Toolify 解析出来的 tool_calls 增补 role=assistant。"""
        output: List[str] = []
        if parsed_chunks:
            role_sse = self._ensure_role_sse(ctx)
            if role_sse:
                output.append(role_sse)
        output.extend(parsed_chunks)
        return output

    # ------------------------------------------------------------------
    # 流结束提取
    # ------------------------------------------------------------------

    def finalize_stream_tool_calls(self, ctx: ToolifyContext) -> List[str]:
        """流结束时从累积内容中提取工具调用 (XML 优先, JSON 降级)。"""
        output: List[str] = []
        buffered_content = getattr(ctx, "buffered_content", "")
        trigger_signal = getattr(ctx, "trigger_signal", "")
        tools_defs = getattr(ctx, "tools_defs", None)

        if trigger_signal and trigger_signal in buffered_content:
            parsed = parse_function_calls_xml(buffered_content, trigger_signal)
            if parsed:
                validation_err = validate_parsed_tools(parsed, tools_defs)
                if validation_err:
                    logger.warning("⚠️ 流结束时 Schema 验证失败: {}", validation_err)
                    usable = [
                        p
                        for p in parsed
                        if p.get("name")
                        and isinstance(p.get("args"), dict)
                        and p["args"]
                    ]
                    if usable:
                        logger.warning(
                            "⚠️ Schema 验证失败但参数非空, 强制发送 {} 个工具调用",
                            len(usable),
                        )
                        self._append_tool_calls_output(
                            ctx,
                            output,
                            self._build_tool_call_chunks(ctx, usable),
                            usable,
                        )
                else:
                    self._append_tool_calls_output(
                        ctx,
                        output,
                        self._build_tool_call_chunks(ctx, parsed),
                        parsed,
                    )

        if not ctx.tool_calls_accum:
            json_parsed, _ = parse_and_extract_tool_calls(buffered_content)
            normalized = self._normalize_tool_calls(json_parsed)
            if normalized:
                tool_chunks: List[str] = []
                for tool_call in normalized:
                    sse = format_sse_chunk(
                        create_openai_chunk(
                            ctx.ensure_stream_id(),
                            ctx.model,
                            {"tool_calls": [tool_call]},
                        )
                    )
                    self._log_downstream(ctx, sse)
                    tool_chunks.append(sse)
                self._append_tool_calls_output(
                    ctx,
                    output,
                    tool_chunks,
                    normalized,
                )

        return output

    # ------------------------------------------------------------------
    # 非流式提取
    # ------------------------------------------------------------------

    def extract_non_stream_tool_calls(
        self,
        final_content: str,
        *,
        trigger_signal: str = "",
        tools_defs: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """从非流式聚合文本中提取工具调用。"""
        tool_calls_accum: List[Dict[str, Any]] = []
        cleaned_content = final_content

        if trigger_signal and trigger_signal in cleaned_content:
            parsed = parse_function_calls_xml(cleaned_content, trigger_signal)
            if parsed:
                validation_err = validate_parsed_tools(parsed, tools_defs)
                if not validation_err:
                    normalized = self._normalize_xml_tools(parsed)
                    if normalized:
                        tool_calls_accum = normalized
                        trigger_pos = cleaned_content.find(trigger_signal)
                        if trigger_pos >= 0:
                            cleaned_content = cleaned_content[:trigger_pos].strip()
                        logger.info(
                            "[tools] XML parse success: {} tools", len(normalized)
                        )
                else:
                    logger.warning("⚠️ 非流式 Schema 验证失败: {}", validation_err)

        if not tool_calls_accum:
            parsed_tool_calls, extracted_content = parse_and_extract_tool_calls(
                cleaned_content
            )
            normalized = self._normalize_tool_calls(parsed_tool_calls)
            if normalized:
                tool_calls_accum = normalized
                cleaned_content = extracted_content

        return tool_calls_accum, cleaned_content

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _append_tool_calls_output(
        self,
        ctx: ToolifyContext,
        output: List[str],
        tool_chunks: List[str],
        tool_calls_accum: List[Dict[str, Any]],
    ) -> None:
        if not tool_chunks:
            return
        role_sse = self._ensure_role_sse(ctx)
        if role_sse:
            output.append(role_sse)
        output.extend(tool_chunks)
        ctx.tool_calls_accum = tool_calls_accum

    def _ensure_role_sse(self, ctx: ToolifyContext) -> Optional[str]:
        if self.ensure_role_func:
            return self.ensure_role_func(ctx)
        if ctx.has_sent_role:
            return None
        ctx.has_sent_role = True
        role_sse = format_sse_chunk(
            create_openai_chunk(
                ctx.ensure_stream_id(),
                ctx.model,
                {"role": "assistant"},
            )
        )
        self._log_downstream(ctx, role_sse)
        return role_sse

    def _build_tool_call_chunks(
        self, ctx: ToolifyContext, parsed_tools: List[Dict[str, Any]]
    ) -> List[str]:
        chunks: List[str] = []
        sid = ctx.ensure_stream_id()
        for i, tool in enumerate(parsed_tools):
            name = str(tool.get("name") or "")
            args = tool.get("args")
            if not name:
                continue
            if not isinstance(args, dict):
                args = {}
            tc = {
                "index": i,
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
            sse = format_sse_chunk(
                create_openai_chunk(sid, ctx.model, {"tool_calls": [tc]})
            )
            self._log_downstream(ctx, sse)
            chunks.append(sse)
        return chunks

    def _normalize_tool_calls(
        self,
        raw_tool_calls: Any,
        start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        if self.normalize_tool_calls_func:
            return self.normalize_tool_calls_func(raw_tool_calls, start_index)

        if not raw_tool_calls:
            return []

        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else [raw_tool_calls]
        normalized: List[Dict[str, Any]] = []
        for offset, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function_data = tool_call.get("function") or {}
            normalized.append(
                {
                    "index": tool_call.get("index", start_index + offset),
                    "id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_data.get("name", ""),
                        "arguments": function_data.get("arguments", ""),
                    },
                }
            )
        return normalized

    @staticmethod
    def _normalize_xml_tools(parsed_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for i, tool in enumerate(parsed_tools):
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "")
            args = tool.get("args")
            if not name:
                continue
            if not isinstance(args, dict):
                args = {}
            normalized.append(
                {
                    "index": i,
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        return normalized

    @staticmethod
    def _log_downstream(ctx: ToolifyContext, sse_data: str) -> None:
        if hasattr(ctx, "log_downstream"):
            ctx.log_downstream(sse_data)
