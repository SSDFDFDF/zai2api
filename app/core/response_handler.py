#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""流式和非流式响应处理模块。

将上游响应处理统一收敛到 ResponseHandler。
"""

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.openai_compat import (
    create_openai_chunk,
    create_openai_response_with_reasoning,
    format_sse_chunk,
    handle_error,
)
from app.core.toolify.glm_handler import GLMToolHandler
from app.core.toolify import ToolifyHandler
from app.models.schemas import OpenAIRequest
from app.utils.logger import get_logger
from app.core.toolify.xml_protocol import (
    StreamingFunctionCallDetector,
)

logger = get_logger()

_CITATION_MARKER_RE = re.compile(r"^【turn\d+(?:search|click)\d+】$")


# ------------------------------------------------------------------
# StreamContext: 集中管理流式处理状态
# ------------------------------------------------------------------


@dataclass
class StreamContext:
    """流式处理过程中所有的可变状态。

    替代原来 handle_stream_response 中的 12+ 个 nonlocal 变量，
    使状态流转清晰可追踪。
    """

    chat_id: str
    model: str

    # 流 ID
    stream_id: Optional[str] = None

    # 时间戳缓存（整条流共用一个 created）
    created_at: Optional[int] = None

    # 内容累积
    buffered_content: str = ""

    # Token usage
    usage_info: Dict[str, Any] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )

    # 上游原生 / XML 解析出的工具调用累积
    tool_calls_accum: List[Dict[str, Any]] = field(default_factory=list)

    # SSE 发送状态
    has_sent_role: bool = False
    finished: bool = False
    line_count: int = 0
    downstream_count: int = 0

    # 阶段追踪
    last_phase: Optional[str] = None

    # 思维残留排掉状态
    draining_details: bool = False
    details_drain_buf: str = ""

    # GLM 内部工具执行上下文
    in_glm_tool_execution: bool = False

    # GLM 工具调用提示追踪
    glm_tool_name: str = ""                      # 当前工具名
    phase_before_tool: Optional[str] = None       # 进入 tool_call 前的阶段
    glm_tool_hint_sent: bool = False              # 是否已发送开始提示

    # GLM 引用标记缓冲
    citation_buf: str = ""

    # 重复循环检测
    repeat_buffer: str = ""
    repeat_chunk_count: int = 0

    # 工具检测
    has_tools: bool = False
    trigger_signal: str = ""
    tools_defs: Optional[List[Dict[str, Any]]] = None
    detector: Optional[StreamingFunctionCallDetector] = None
    tool_strategy: str = "xmlfc"

    # GLM native tool_calls: tool_call 阶段结束后待解析标记
    glm_tool_calls_pending: bool = False

    # GLM delta 流式工具调用累积 (delta_name / delta_arguments 格式)
    glm_delta_tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def ensure_stream_id(self, chunk_data: Optional[Dict] = None) -> str:
        """确保 stream_id 已初始化并返回。"""
        if self.stream_id is None:
            upstream_id = (
                chunk_data.get("id") if isinstance(chunk_data, dict) else None
            )
            self.stream_id = upstream_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
            self.created_at = int(time.time())
        return self.stream_id

    def log_downstream(self, sse_data: str) -> None:
        """记录下游 SSE 调试日志。"""
        self.downstream_count += 1
        logger.debug(
            "\U0001f4e4 Client SSE #{}: {}",
            self.downstream_count,
            sse_data.rstrip(),
        )

    def process_citation_marker(self, text: str) -> str:
        """过滤 GLM 内部引用标记 (如 【turn9click0】)。

        批量处理无标记区域，仅对标记区域做精细处理。
        """
        if not text and not self.citation_buf:
            return text

        # 快速路径：无残留缓冲且文本中无引用标记起始符
        if not self.citation_buf and "【" not in text:
            return text

        res_parts: List[str] = []
        i = 0
        text_len = len(text)

        # 先处理跨 chunk 的残留缓冲
        if self.citation_buf:
            while i < text_len:
                char = text[i]
                i += 1
                self.citation_buf += char
                if self.citation_buf.endswith("】"):
                    if _CITATION_MARKER_RE.match(self.citation_buf):
                        self.citation_buf = ""  # 完整引用标记，丢弃
                    else:
                        res_parts.append(self.citation_buf)
                        self.citation_buf = ""
                    break
                elif len(self.citation_buf) > 30:
                    res_parts.append(self.citation_buf)  # 超长，退还
                    self.citation_buf = ""
                    break
                elif (
                    not "【turn".startswith(self.citation_buf)
                    and not self.citation_buf.startswith("【turn")
                ):
                    res_parts.append(self.citation_buf)  # 偏离特征，退还
                    self.citation_buf = ""
                    break

        # 批量处理剩余文本
        while i < text_len:
            next_mark = text.find("【", i)
            if next_mark == -1:
                # 无更多标记，批量追加剩余
                res_parts.append(text[i:])
                break

            # 追加标记前的内容
            if next_mark > i:
                res_parts.append(text[i:next_mark])

            # 从 【 开始精细处理
            self.citation_buf = "【"
            i = next_mark + 1
            while i < text_len:
                char = text[i]
                i += 1
                self.citation_buf += char
                if self.citation_buf.endswith("】"):
                    if _CITATION_MARKER_RE.match(self.citation_buf):
                        self.citation_buf = ""  # 完整引用标记，丢弃
                    else:
                        res_parts.append(self.citation_buf)
                        self.citation_buf = ""
                    break
                elif len(self.citation_buf) > 30:
                    res_parts.append(self.citation_buf)
                    self.citation_buf = ""
                    break
                elif (
                    not "【turn".startswith(self.citation_buf)
                    and not self.citation_buf.startswith("【turn")
                ):
                    res_parts.append(self.citation_buf)
                    self.citation_buf = ""
                    break
            else:
                # text 结束但 citation_buf 仍在缓冲中（跨 chunk），保持
                break

        return "".join(res_parts)


class ResponseHandler:
    """上游 SSE 响应转换为 OpenAI 格式的处理器。

    提供流式和非流式两种处理路径，以及辅助的内容清理方法。
    """

    def __init__(self) -> None:
        self.logger = logger
        # 初始化外部组件
        self.glm_tool_handler = GLMToolHandler(
            enabled=settings.GLM_INTERNAL_TOOL_HINT_ENABLED,
            emit_func=self._emit_sse
        )
        self.toolify_handler = ToolifyHandler(
            emit_func=self._emit_sse,
            ensure_role_func=self._ensure_role_sse,
            normalize_tool_calls_func=self.normalize_tool_calls,
        )

    # ==================================================================
    # 内容辅助方法
    # ==================================================================

    def clean_reasoning_delta(self, delta_content: str) -> str:
        """清理思考阶段的 details 包裹内容。"""
        if not delta_content:
            return ""

        if delta_content.startswith("<details"):
            if "</summary>\n>" in delta_content:
                return delta_content.split("</summary>\n>")[-1].strip()
            if "</summary>\n" in delta_content:
                return delta_content.split("</summary>\n")[-1].lstrip("> ").strip()

        return delta_content

    # 所有可能包裹思维内容的 XML 标签名（通用化，新增标签只需在此添加）
    _THINKING_TAGS = ("details", "think", "reasoning", "thought")
    _THINKING_TAGS_ALT = "|".join(_THINKING_TAGS)
    # 预编译的关闭标签正则：匹配 </details> 或 </think> 等
    _THINKING_CLOSE_RE = re.compile(
        r"</" + "|".join(f"(?:{t})" for t in _THINKING_TAGS) + r">"
    )
    _THINKING_BLOCK_RE = re.compile(
        rf"<(?:{_THINKING_TAGS_ALT})[^>]*>.*?</(?:{_THINKING_TAGS_ALT})>\s*",
        flags=re.DOTALL,
    )
    _THINKING_ORPHAN_CLOSE_RE = re.compile(
        rf'^[^<]*?(?:true|false)?"?>\s*(?:>\s*.*?)?</(?:{_THINKING_TAGS_ALT})>\s*',
        flags=re.DOTALL,
    )
    _THINKING_ORPHAN_OPEN_RE = re.compile(
        rf"<(?:{_THINKING_TAGS_ALT})[^>]*>.*$",
        flags=re.DOTALL,
    )
    _GLM_BLOCK_FULL_RE = re.compile(
        r"<glm_block[^>]*>.*?</glm_block>",
        flags=re.DOTALL,
    )
    _GLM_BLOCK_TAIL_RE = re.compile(
        r"<glm_block[^>]*>.*",
        flags=re.DOTALL,
    )
    _CITATION_FULL_RE = re.compile(r"【turn\d+(?:search|click)\d+】")
    _CITATION_TAIL_RE = re.compile(r"【turn\d*(?:search|click)?\d*$")

    @classmethod
    def strip_thinking_residue(cls, text: str) -> Tuple[str, bool]:
        """Strip thinking-phase residue from non-thinking content.

        Handles ANY wrapper tag listed in ``_THINKING_TAGS`` (``<details>``,
        ``<think>``, ``<reasoning>``, etc.) that may leak across phase
        boundaries.

        Returns:
            Tuple[str, bool]: (cleaned_text, is_unclosed_tag_found)
        """
        if not text:
            return "", False

        is_unclosed = False

        # 1. Strip complete <tag ...>...</tag> blocks
        cleaned = cls._THINKING_BLOCK_RE.sub("", text)

        # 2. Strip orphan closing fragment (tail of a tag opened in thinking)
        cleaned = cls._THINKING_ORPHAN_CLOSE_RE.sub("", cleaned)

        # 3. Strip orphan opening without closing
        m_open = cls._THINKING_ORPHAN_OPEN_RE.search(cleaned)
        if m_open:
            is_unclosed = True
            cleaned = cleaned[: m_open.start()]

        final_cleaned = (
            cleaned.strip() if cleaned.strip() != text.strip() else text
        )
        return final_cleaned, is_unclosed

    def extract_answer_content(self, text: str) -> str:
        """提取思考结束后的答案正文。"""
        if not text:
            return ""

        if "</details>\n" in text:
            return text.split("</details>\n")[-1]

        if "</details>" in text:
            return text.split("</details>")[-1].lstrip()

        return text

    # ==================================================================
    # 重复循环检测
    # ==================================================================

    @staticmethod
    def _detect_repetition_loop(
        text: str,
        min_pattern_len: int = 10,
        min_repeats: int = 8,
        max_pattern_len: int = 60,
    ) -> Optional[str]:
        """检测文本是否进入重复循环。

        在滑动窗口中查找短模式的高频重复。当一个 10–60 字符的子串
        在窗口中出现 8+ 次时，判定为重复循环。

        Returns:
            检测到的重复模式，或 None
        """
        if len(text) < min_pattern_len * min_repeats:
            return None

        upper = min(max_pattern_len, len(text) // min_repeats) + 1
        for plen in range(min_pattern_len, upper):
            pattern = text[-plen:]
            if text.count(pattern) >= min_repeats:
                return pattern

        return None

    # ==================================================================
    # 工具调用辅助
    # ==================================================================

    def normalize_tool_calls(
        self,
        raw_tool_calls: Any,
        start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        """标准化上游工具调用为 OpenAI 兼容格式。"""
        if not raw_tool_calls:
            return []

        tool_calls = (
            raw_tool_calls
            if isinstance(raw_tool_calls, list)
            else [raw_tool_calls]
        )
        normalized: List[Dict[str, Any]] = []

        for offset, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue

            function_data = tool_call.get("function") or {}
            normalized.append(
                {
                    "index": tool_call.get("index", start_index + offset),
                    "id": tool_call.get("id")
                    or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_data.get("name", ""),
                        "arguments": function_data.get("arguments", ""),
                    },
                }
            )

        return normalized

    def format_search_results(self, data: Dict[str, Any]) -> str:
        """将上游搜索结果格式化为可追加的 Markdown 引用。"""
        search_info = (
            data.get("results")
            or data.get("sources")
            or data.get("citations")
        )
        if not isinstance(search_info, list) or not search_info:
            return ""

        citations = []
        for index, item in enumerate(search_info, 1):
            if not isinstance(item, dict):
                continue

            title = (
                item.get("title") or item.get("name") or f"Result {index}"
            )
            url = item.get("url") or item.get("link")
            if url:
                citations.append(f"[{index}] [{title}]({url})")

        if not citations:
            return ""

        return "\n\n---\n" + "\n".join(citations)

    def _emit_sse(
        self,
        ctx: StreamContext,
        delta: Dict[str, Any],
    ) -> List[str]:
        """构建 SSE chunk 并返回待 yield 的列表 (含可能的 role chunk)。"""
        output: List[str] = []

        role_sse = self._ensure_role_sse(ctx)
        if role_sse:
            output.append(role_sse)

        sse = format_sse_chunk(
            create_openai_chunk(
                ctx.ensure_stream_id(), ctx.model, delta,
                created=ctx.created_at,
            )
        )
        ctx.log_downstream(sse)
        output.append(sse)
        return output

    def _ensure_role_sse(self, ctx: StreamContext) -> Optional[str]:
        """确保已发送 role=assistant，返回新增的 role chunk。"""
        if ctx.has_sent_role:
            return None
        ctx.has_sent_role = True
        role_sse = format_sse_chunk(
            create_openai_chunk(
                ctx.ensure_stream_id(),
                ctx.model,
                {"role": "assistant"},
                created=ctx.created_at,
            )
        )
        ctx.log_downstream(role_sse)
        return role_sse

    @staticmethod
    def _extract_sse_data_payload(line: str) -> Tuple[str, str]:
        """从原始 SSE 行提取 data payload。

        Returns:
            (state, payload)
            - state in {"data", "done", "non_data", "skip"}
        """
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

    # ==================================================================
    # 流式处理: SSE 行解析
    # ==================================================================

    def _parse_sse_line(
        self, line: str, ctx: StreamContext
    ) -> Tuple[str, Optional[Tuple[Dict[str, Any], Dict[str, Any]]]]:
        """解析一行 SSE 数据。

        Returns:
            (state, parsed)：
            - state = "done" 时表示命中 [DONE]
            - state = "skip" 时表示应跳过当前行
            - state = "ok" 且 parsed 非空时返回 (chunk, data)
        """
        if not line.strip():
            return "skip", None

        ctx.line_count += 1
        self.logger.debug(
            "🔍 SSE line #{}: {}",
            ctx.line_count,
            line,
        )

        state, chunk_str = self._extract_sse_data_payload(line)
        if state == "done":
            return "done", None
        if state == "non_data":
            if ctx.line_count <= 3:
                self.logger.debug(
                    "🔍 SSE line #{} 跳过 (non-data): {}",
                    ctx.line_count,
                    chunk_str,
                )
            return "skip", None
        if state != "data":
            return "skip", None

        try:
            chunk = json.loads(chunk_str)
        except json.JSONDecodeError as error:
            # 出错时才记录开销较大的日志
            self.logger.debug(
                "❌ JSON解析错误: {}, 内容: {}",
                error,
                chunk_str,
            )
            return "skip", None
            
        if not isinstance(chunk, dict):
            return "skip", None

        chunk_type = chunk.get("type")
        data = (
            chunk.get("data", {})
            if chunk_type == "chat:completion"
            else chunk
        )
        if not isinstance(data, dict):
            return "skip", None

        return "ok", (chunk, data)

    # ==================================================================
    # 流式处理: 状态更新
    # ==================================================================

    def _update_stream_state(
        self,
        ctx: StreamContext,
        data: Dict[str, Any],
        chunk: Dict[str, Any],
    ) -> None:
        """更新阶段追踪、usage、GLM 工具执行状态。"""
        ctx.ensure_stream_id(chunk)

        phase = data.get("phase")
        if phase and phase != ctx.last_phase:
            prev = ctx.last_phase
            self.logger.debug("📈 SSE 阶段: {} → {}", prev, phase)
            ctx.last_phase = phase

            # 追踪 GLM 内部 MCP 工具执行上下文
            if phase == "tool_call":
                ctx.in_glm_tool_execution = True
                ctx.phase_before_tool = prev
                ctx.glm_tool_hint_sent = False
                ctx.glm_tool_name = ""
            elif phase == "tool_response":
                pass  # 由 _handle_glm_tool_hint 处理
            elif phase in ("answer", "thinking"):
                # glmnative: tool_call 阶段结束，标记待解析
                if (
                    ctx.in_glm_tool_execution
                    and ctx.tool_strategy == "glmnative"
                ):
                    ctx.glm_tool_calls_pending = True
                if ctx.in_glm_tool_execution:
                    self.logger.debug(
                        "[glm-tool] 工具执行完毕, 恢复正常输出 (-> {})",
                        phase,
                    )
                ctx.in_glm_tool_execution = False

        # Usage
        data_usage = data.get("usage")
        if data_usage and data_usage != ctx.usage_info:
            ctx.usage_info = data_usage
            self.logger.info("[usage] catch data.usage: {}", ctx.usage_info)
            return

        chunk_usage = chunk.get("usage")
        if chunk_usage and chunk_usage != ctx.usage_info:
            ctx.usage_info = chunk_usage
            self.logger.info("[usage] catch chunk.usage: {}", ctx.usage_info)

    # ==================================================================
    # 流式处理: 内容累积
    # ==================================================================

    def _accumulate_content(
        self, ctx: StreamContext, data: Dict[str, Any]
    ) -> str:
        """累积 delta_content / edit_content 到 buffered_content。

        Returns:
            当前 chunk 的文本内容 (current_text)。
        """
        delta_content = data.get("delta_content", "")
        edit_content = data.get("edit_content", "")
        edit_index = data.get("edit_index")
        max_buffered_len = 50000

        current_text = ""
        if delta_content:
            current_text = delta_content
            ctx.buffered_content = (
                ctx.buffered_content + delta_content
            )[-max_buffered_len:]
        elif edit_content:
            current_text = edit_content
            if edit_index is not None and isinstance(edit_index, int):
                safe_idx = max(
                    0, min(edit_index, len(ctx.buffered_content))
                )
                if safe_idx != edit_index:
                    self.logger.debug(
                        "🔧 edit_index {} 越界 (buffered={}), 截断为 {}",
                        edit_index,
                        len(ctx.buffered_content),
                        safe_idx,
                    )
                ctx.buffered_content = (
                    ctx.buffered_content[:safe_idx]
                    + edit_content
                    + ctx.buffered_content[safe_idx:]
                )
                if len(ctx.buffered_content) > max_buffered_len:
                    ctx.buffered_content = ctx.buffered_content[
                        -max_buffered_len:
                    ]
                self.logger.debug(
                    "🔧 edit_index={}: 在位置 {} 插入 {} 字符, buffered 总长={}",
                    edit_index,
                    safe_idx,
                    len(edit_content),
                    len(ctx.buffered_content),
                )
            else:
                ctx.buffered_content = (
                    ctx.buffered_content + edit_content
                )[-max_buffered_len:]

        return current_text

    # ==================================================================
    # 流式处理: SSE 错误检查
    # ==================================================================

    def _check_sse_error(
        self, ctx: StreamContext, data: Dict[str, Any]
    ) -> Optional[List[str]]:
        """检查上游 SSE 事件中的错误。

        Returns:
            要 yield 的 SSE 列表 (含 DONE)，或 None。
        """
        sse_error = data.get("error")
        if not isinstance(sse_error, dict):
            return None

        error_code_val = sse_error.get("code", "UNKNOWN")
        error_detail = sse_error.get("detail", "Unknown upstream error")

        if data.get("done") is True:
            self.logger.debug(
                "⚠️ 上游 SSE 在 done 状态返回错误 (容忍忽略): code={}, detail={}",
                error_code_val,
                error_detail,
            )
            return None

        self.logger.error(
            "❌ 上游 SSE 返回错误: code={}, detail={}",
            error_code_val,
            error_detail,
        )
        error_response = {
            "error": {
                "message": error_detail,
                "type": "upstream_error",
                "code": error_code_val,
            }
        }

        output: List[str] = []
        role_sse = self._ensure_role_sse(ctx)
        if role_sse:
            output.append(role_sse)

        output.append(f"data: {json.dumps(error_response)}\n\n")
        output.append("data: [DONE]\n\n")
        return output

    # ==================================================================
    # 流式处理: 直接工具调用透传
    # ==================================================================

    def _handle_direct_tool_calls(
        self, ctx: StreamContext, data: Dict[str, Any]
    ) -> List[str]:
        """处理上游原生 tool_calls 直接透传。"""
        direct_tool_calls = self.normalize_tool_calls(
            data.get("tool_calls"), len(ctx.tool_calls_accum)
        )
        if not direct_tool_calls:
            return []

        ctx.tool_calls_accum.extend(direct_tool_calls)
        output: List[str] = []
        for tool_call in direct_tool_calls:
            output.extend(
                self._emit_sse(ctx, {"tool_calls": [tool_call]})
            )
        return output

    # ==================================================================
    # 流式处理: 重复循环检测
    # ==================================================================

    def _check_repetition(
        self, ctx: StreamContext, current_text: str
    ) -> Optional[List[str]]:
        """重复循环检测。

        Returns:
            检测到重复时返回 error SSE 列表 (调用方应 break)，否则 None。
        """
        phase = ctx.last_phase
        if not current_text or phase in ("thinking", "done"):
            return None

        ctx.repeat_buffer = (ctx.repeat_buffer + current_text)[-500:]
        ctx.repeat_chunk_count += 1

        if ctx.repeat_chunk_count < 100 or ctx.repeat_chunk_count % 30 != 0:
            return None

        repeated_pattern = self._detect_repetition_loop(ctx.repeat_buffer)
        if not repeated_pattern:
            return None

        self.logger.warning(
            "⚠️ 检测到模型重复循环! 模式: {!r}, 已处理 {} 行/{} chunks, 强制终止流",
            repeated_pattern[:30],
            ctx.line_count,
            ctx.repeat_chunk_count,
        )
        error_msg = (
            "\n\n[ERROR: Model output repetition loop detected, "
            "stream terminated automatically. Please retry.]"
        )
        return self._emit_sse(ctx, {"content": error_msg})

    # ==================================================================
    # 流式处理: 思维残留清理
    # ==================================================================

    def _handle_thinking_residue(
        self, ctx: StreamContext, current_text: str
    ) -> str:
        """清理非 thinking 阶段的思维标签残留。

        Returns:
            清理后的 current_text。返回空字符串表示本 chunk 应跳过。
        """
        phase = ctx.last_phase
        detector = ctx.detector

        # 只在非 thinking 阶段 & 有内容 & 不在工具解析时处理
        if (
            phase == "thinking"
            or not current_text
            or (detector and detector.state == "tool_parsing")
        ):
            return current_text

        # 正在排掉思维残留
        if ctx.draining_details:
            ctx.details_drain_buf += current_text
            if len(ctx.details_drain_buf) > 65536:
                self.logger.warning("⚠️ details_drain_buf 超过 64KB 上限，强制重置 draining_details 状态")
                ctx.draining_details = False
                ctx.details_drain_buf = ""
                return current_text
            m = self._THINKING_CLOSE_RE.search(ctx.details_drain_buf)
            if m:
                remainder = ctx.details_drain_buf[m.end() :].lstrip()
                if remainder:
                    self.logger.debug(
                        "🧹 排掉思维残留完成, 剩余内容: {}...",
                        remainder[:80],
                    )
                else:
                    self.logger.debug("🧹 排掉思维残留完成, 无剩余内容")
                ctx.draining_details = False
                ctx.details_drain_buf = ""
                if not remainder:
                    return ""
                current_text = remainder
            else:
                return ""  # 继续缓冲
        else:
            # 防范思维标签泄漏
            cleaned, is_unclosed = self.strip_thinking_residue(current_text)

            if is_unclosed:
                ctx.draining_details = True
                ctx.details_drain_buf = current_text
                self.logger.debug(
                    "🧹 检测到未闭合思维标签残留, 开始排掉: {}...",
                    current_text[:80],
                )

            if not cleaned:
                return ""
            elif cleaned != current_text:
                self.logger.debug(
                    "🧹 一次性清理思维残留: {}→{} 字符",
                    len(current_text),
                    len(cleaned),
                )
                current_text = cleaned

        # GLM <glm_block> 过滤
        if current_text and "<glm_block" in current_text:
            self.logger.debug(
                "🧹 过滤 GLM 内部工具调用块: {}...",
                current_text[:80],
            )
            return ""

        return current_text

    # ==================================================================
    # 流式处理: 阶段内容输出
    # ==================================================================

    def _process_phase_output(
        self,
        ctx: StreamContext,
        data: Dict[str, Any],
        current_text: str,
        chunk_type: Optional[str],
    ) -> List[str]:
        """根据 phase 生成内容 SSE。"""
        phase = ctx.last_phase
        delta_content = data.get("delta_content", "")

        if phase == "thinking" and delta_content:
            cleaned = self.clean_reasoning_delta(delta_content)
            if cleaned:
                return self._emit_sse(
                    ctx, {"reasoning_content": cleaned}
                )

        elif phase in ("answer", "other") and current_text:
            # 跳过 GLM 内部工具执行期间的 other 阶段内容
            if ctx.in_glm_tool_execution and phase != "answer":
                self.logger.debug(
                    "🧹 跳过 GLM 工具执行期 other 内容: {}...",
                    current_text[:60],
                )
            else:
                filtered_text = ctx.process_citation_marker(current_text)
                if filtered_text:
                    return self._emit_sse(
                        ctx, {"content": filtered_text}
                    )

        elif phase == "search" or chunk_type == "web_search":
            citation_text = self.format_search_results(data)
            if citation_text:
                return self._emit_sse(
                    ctx, {"content": citation_text}
                )

        return []

    # ==================================================================
    # 流式处理: 流结束处理
    # ==================================================================

    async def _finalize_stream(
        self, ctx: StreamContext
    ) -> AsyncGenerator[str, None]:
        """流结束时的收尾处理。"""
        if ctx.finished:
            return

        # 最后尝试: 从累积内容中提取工具调用 (仅 xmlfc/hybrid)
        if (
            ctx.has_tools
            and not ctx.tool_calls_accum
            and ctx.tool_strategy in ("xmlfc", "hybrid")
        ):
            for sse in self.toolify_handler.finalize_stream_tool_calls(ctx):
                yield sse

        # glmnative: 流结束前刷出未处理的 delta 工具调用
        if (
            ctx.has_tools
            and ctx.tool_strategy == "glmnative"
            and (ctx.glm_delta_tool_calls or ctx.glm_tool_calls_pending)
        ):
            ctx.glm_tool_calls_pending = True
            for sse in self.glm_tool_handler.handle_native_extraction(ctx):
                yield sse

        # 确保 role 已发送
        role_sse = self._ensure_role_sse(ctx)
        if role_sse:
            yield role_sse

        # 刷出残留的引用标记缓冲
        if ctx.citation_buf:
            if not self._CITATION_TAIL_RE.match(ctx.citation_buf):
                sse = format_sse_chunk(
                    create_openai_chunk(
                        ctx.ensure_stream_id(),
                        ctx.model,
                        {"content": ctx.citation_buf},
                        created=ctx.created_at,
                    )
                )
                ctx.log_downstream(sse)
                yield sse

        # 刷出 detector 残留
        if (
            ctx.detector
            and ctx.detector.state != "tool_parsing"
            and not ctx.tool_calls_accum
        ):
            remaining = ctx.detector.flush()
            if remaining:
                self.logger.debug(
                    "🧹 刷出 detector 残留缓冲: {} 字符: {}",
                    len(remaining),
                    remaining[:80],
                )
                sse = format_sse_chunk(
                    create_openai_chunk(
                        ctx.ensure_stream_id(),
                        ctx.model,
                        {"content": remaining},
                        created=ctx.created_at,
                    )
                )
                ctx.log_downstream(sse)
                yield sse

        # finish chunk
        finish_reason = "tool_calls" if ctx.tool_calls_accum else "stop"
        finish_chunk = create_openai_chunk(
            ctx.ensure_stream_id(), ctx.model, {}, finish_reason,
            created=ctx.created_at,
        )
        finish_chunk["usage"] = ctx.usage_info
        finish_sse = format_sse_chunk(finish_chunk)
        ctx.log_downstream(finish_sse)
        yield finish_sse
        ctx.log_downstream("data: [DONE]")
        yield "data: [DONE]\n\n"
        ctx.finished = True

    # ==================================================================
    # 流式响应处理 (主入口)
    # ==================================================================

    async def handle_stream_response(
        self,
        response: httpx.Response,
        chat_id: str,
        model: str,
        request: OpenAIRequest,
        transformed: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """处理上游流式响应，转换为 OpenAI SSE 流。

        集成 Toolify StreamingFunctionCallDetector 实时检测 XML 工具调用。
        """
        start_time = getattr(request, "started_at", None)
        if start_time:
            ttfb = time.perf_counter() - start_time
            self.logger.info(
                "[stream] upstream success, start handle SSE stream, ttfb: {:.3f}s",
                ttfb,
            )
        else:
            self.logger.info(
                "[stream] upstream success, start handle SSE stream"
            )

        sse_start_time = time.perf_counter()

        # 初始化上下文
        trigger_signal = transformed.get("trigger_signal", "")
        tools_defs = transformed.get("tools")
        tool_strategy = transformed.get("tool_strategy", "xmlfc")
        has_tools = tool_strategy != "disabled" and bool(request.tools)
        need_xml = tool_strategy in ("xmlfc", "hybrid") and bool(trigger_signal)

        ctx = StreamContext(
            chat_id=chat_id,
            model=model,
            has_tools=has_tools,
            trigger_signal=trigger_signal,
            tools_defs=tools_defs,
            tool_strategy=tool_strategy,
        )

        if need_xml:
            ctx.detector = StreamingFunctionCallDetector(trigger_signal)
            self.logger.debug(
                "🔧 已初始化流式工具检测器, 触发信号: {}...",
                trigger_signal[:20],
            )

        try:
            async for line in response.aiter_lines():
                # 解析 SSE（包含 DONE 判断）
                parse_state, parsed = self._parse_sse_line(line, ctx)
                if parse_state == "done":
                    break
                if parse_state != "ok" or parsed is None:
                    continue
                chunk, data = parsed

                # 错误检查
                error_output = self._check_sse_error(ctx, data)
                if error_output:
                    for sse in error_output:
                        yield sse
                    return

                # 更新状态
                self._update_stream_state(ctx, data, chunk)

                # 累积内容
                current_text = self._accumulate_content(ctx, data)

                # GLM delta 流式工具调用累积 (glmnative)
                if ctx.tool_strategy == "glmnative":
                    GLMToolHandler.accumulate_delta(ctx, data)

                # GLM native tool_calls 提取（tool_call 阶段结束时触发）
                for sse in self.glm_tool_handler.handle_native_extraction(ctx):
                    yield sse

                # 上游原生 tool_calls 透传（disabled 时跳过）
                if ctx.tool_strategy != "disabled":
                    for sse in self._handle_direct_tool_calls(ctx, data):
                        yield sse

                # GLM 内部工具调用提示
                tool_hint = self.glm_tool_handler.process(ctx, data)
                if tool_hint:
                    for sse in tool_hint:
                        yield sse

                # 重复循环检测
                repeat_result = self._check_repetition(ctx, current_text)
                if repeat_result:
                    for sse in repeat_result:
                        yield sse
                    break

                # 思维残留清理
                current_text = self._handle_thinking_residue(
                    ctx, current_text
                )
                if not current_text:
                    continue

                # Toolify 工具检测 (仅 xmlfc/hybrid)
                if ctx.tool_strategy in ("xmlfc", "hybrid"):
                    detection_result = self.toolify_handler.handle_detection(
                        ctx, current_text
                    )
                    if detection_result is not None:
                        for sse in detection_result:
                            yield sse
                        continue

                    # 工具解析模式
                    parsing_result = self.toolify_handler.handle_parsing(
                        ctx, current_text
                    )
                    if parsing_result is not None:
                        # 增补 role 和发送
                        final_output = self.toolify_handler.inject_role_and_chunks(
                            ctx, parsing_result
                        )
                        for sse in final_output:
                            yield sse

                        if ctx.tool_calls_accum:
                            # 解析完成, 自带结流
                            ctx.finished = True
                            finish_chunk = create_openai_chunk(
                                ctx.ensure_stream_id(), ctx.model, {}, "tool_calls",
                                created=ctx.created_at,
                            )
                            finish_chunk["usage"] = ctx.usage_info
                            finish_sse = format_sse_chunk(finish_chunk)
                            ctx.log_downstream(finish_sse)
                            yield finish_sse
                            ctx.log_downstream("data: [DONE]")
                            yield "data: [DONE]\n\n"
                            return
                        continue

                # 阶段内容输出
                chunk_type = chunk.get("type")
                for sse in self._process_phase_output(
                    ctx, data, current_text, chunk_type
                ):
                    yield sse

                if data.get("done"):
                    break

            # 收尾
            if not ctx.finished:
                async for final_chunk in self._finalize_stream(ctx):
                    yield final_chunk

        except Exception as e:
            self.logger.error("❌ 流式响应处理错误: {}", e)
            if not ctx.finished:
                yield format_sse_chunk(
                    create_openai_chunk(
                        ctx.ensure_stream_id(), model, {}, "stop",
                        created=ctx.created_at,
                    )
                )
                yield "data: [DONE]\n\n"
        finally:
            elapsed = time.perf_counter() - sse_start_time
            total_elapsed = (
                time.perf_counter() - start_time if start_time else None
            )
            if total_elapsed is not None:
                self.logger.info(
                    f"[stream] SSE done {ctx.line_count} lines, "
                    f"SSE time: {elapsed:.3f}s, "
                    f"total time: {total_elapsed:.3f}s"
                )
            else:
                self.logger.info(
                    f"[stream] SSE done {ctx.line_count} lines, "
                    f"SSE time: {elapsed:.3f}s"
                )

    # ==================================================================
    # 非流式响应处理
    # ==================================================================

    async def handle_non_stream_response(
        self,
        response: httpx.Response,
        chat_id: str,
        model: str,
        trigger_signal: str = "",
        tools_defs: Optional[List[Dict[str, Any]]] = None,
        tool_strategy: str = "xmlfc",
    ) -> Dict[str, Any]:
        """处理非流式响应，聚合上游 SSE 为一次性 OpenAI 响应。

        集成 XML 解析优先，JSON 解析降级。
        """
        final_content_parts: List[str] = []
        reasoning_content_parts: List[str] = []
        tool_calls_accum: List[Dict[str, Any]] = []
        usage_info: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        in_glm_tool_execution = False
        last_phase = None
        glm_delta_accum: List[Dict[str, Any]] = []

        try:
            async for line in response.aiter_lines():
                state, payload = self._extract_sse_data_payload(line)
                if state == "non_data":
                    try:
                        maybe_err = json.loads(payload)
                        if isinstance(maybe_err, dict) and (
                            "error" in maybe_err
                            or "code" in maybe_err
                            or "message" in maybe_err
                        ):
                            msg = (
                                (maybe_err.get("error") or {}).get("message")
                                if isinstance(
                                    maybe_err.get("error"), dict
                                )
                                else maybe_err.get("message")
                            ) or "上游返回错误"
                            return handle_error(
                                Exception(msg), "API响应"
                            )
                    except Exception:
                        pass
                    continue
                if state != "data":
                    continue

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                chunk_type = chunk.get("type")
                data = (
                    chunk.get("data", {})
                    if chunk_type == "chat:completion"
                    else chunk
                )
                if not isinstance(data, dict):
                    continue

                phase = data.get("phase")
                # 追踪 GLM 内部 MCP 工具执行上下文
                if phase and phase != last_phase:
                    last_phase = phase
                    if phase == "tool_call":
                        in_glm_tool_execution = True
                    elif phase in ("answer", "thinking"):
                        in_glm_tool_execution = False

                delta_content = data.get("delta_content", "")
                edit_content = data.get("edit_content", "")
                edit_index = data.get("edit_index")

                if phase == "thinking" and delta_content:
                    cleaned_reasoning = self.clean_reasoning_delta(
                        delta_content
                    )
                    if cleaned_reasoning:
                        reasoning_content_parts.append(cleaned_reasoning)

                elif phase == "answer":
                    if delta_content:
                        final_content_parts.append(delta_content)
                    elif edit_content:
                        ec = self.extract_answer_content(edit_content)
                        if edit_index is not None and isinstance(
                            edit_index, int
                        ):
                            final_content = "".join(final_content_parts)
                            safe_idx = max(
                                0, min(edit_index, len(final_content))
                            )
                            final_content = (
                                final_content[:safe_idx]
                                + ec
                                + final_content[safe_idx:]
                            )
                            final_content_parts = [final_content]
                        else:
                            if ec:
                                final_content_parts.append(ec)

                elif phase == "other" and edit_content:
                    if not in_glm_tool_execution:
                        ec = self.extract_answer_content(edit_content)
                        if edit_index is not None and isinstance(
                            edit_index, int
                        ):
                            final_content = "".join(final_content_parts)
                            safe_idx = max(
                                0, min(edit_index, len(final_content))
                            )
                            final_content = (
                                final_content[:safe_idx]
                                + ec
                                + final_content[safe_idx:]
                            )
                            final_content_parts = [final_content]
                        else:
                            if ec:
                                final_content_parts.append(ec)

                elif phase == "search" or chunk_type == "web_search":
                    search_content = self.format_search_results(data)
                    if search_content:
                        final_content_parts.append(search_content)

                # 原生 tool_calls 收集（disabled 时跳过）
                if tool_strategy != "disabled":
                    tool_calls_accum.extend(
                        self.normalize_tool_calls(
                            data.get("tool_calls"),
                            len(tool_calls_accum),
                        )
                    )

                # glmnative: 累积 delta_name / delta_arguments
                if tool_strategy == "glmnative":
                    delta_name = data.get("delta_name", "")
                    delta_args = data.get("delta_arguments", "")
                    if delta_name:
                        metadata = data.get("metadata", {})
                        call_id = metadata.get("tool_call_id") or f"call_{uuid.uuid4().hex[:24]}"
                        glm_delta_accum.append({
                            "id": call_id, "name": delta_name, "arguments": delta_args,
                        })
                    elif delta_args and glm_delta_accum:
                        glm_delta_accum[-1]["arguments"] += delta_args

                if data.get("usage"):
                    usage_info = data["usage"]
                elif chunk.get("usage"):
                    usage_info = chunk["usage"]

        except Exception as e:
            self.logger.error("❌ 非流式响应处理错误: {}", e)
            return handle_error(e, "非流式聚合")

        final_content = "".join(final_content_parts)
        # XML 兜底提取（仅 xmlfc/hybrid）
        if not tool_calls_accum and tool_strategy in ("xmlfc", "hybrid"):
            tool_calls_accum, final_content = self.toolify_handler.extract_non_stream_tool_calls(
                final_content,
                trigger_signal=trigger_signal,
                tools_defs=tools_defs,
            )

        final_content = (final_content or "").strip()

        # GLM native tool_calls 提取（glmnative 模式，glm_block 清理前）
        if (
            not tool_calls_accum
            and tool_strategy == "glmnative"
        ):
            allowed = GLMToolHandler._extract_tool_names(tools_defs)
            # 1. <glm_block> 提取
            glm_tool_calls = GLMToolHandler.parse_tool_calls(
                final_content, allowed_names=allowed,
            )
            # 2. delta 累积提取
            for entry in glm_delta_accum:
                name = entry.get("name", "")
                if allowed is not None and name not in allowed:
                    continue
                glm_tool_calls.append({
                    "index": len(glm_tool_calls),
                    "id": entry.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": entry.get("arguments", "{}"),
                    },
                })
            if glm_tool_calls:
                self.logger.info(
                    "[glm-native] 非流式提取 {} 个工具调用",
                    len(glm_tool_calls),
                )
                tool_calls_accum.extend(glm_tool_calls)

        # 清理 GLM 内部工具调用残留
        final_content = self._GLM_BLOCK_FULL_RE.sub("", final_content)
        final_content = self._GLM_BLOCK_TAIL_RE.sub("", final_content)
        # 清理完整和末尾截断的引用标记
        final_content = self._CITATION_FULL_RE.sub("", final_content)
        final_content = self._CITATION_TAIL_RE.sub("", final_content)

        reasoning_content = "".join(reasoning_content_parts).strip()

        if not final_content and reasoning_content_parts:
            final_content = reasoning_content

        return create_openai_response_with_reasoning(
            chat_id,
            model,
            final_content,
            reasoning_content,
            usage_info,
            tool_calls_accum or None,
        )
