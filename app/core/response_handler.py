#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""流式和非流式响应处理模块。

将原 UpstreamClient._handle_stream_response()、_handle_non_stream_response()
及辅助方法提取为 ResponseHandler 类。
所有参数和行为与原实现完全一致。
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
from app.core.response_handler_glm import GLMToolHandler
from app.core.response_handler_toolify import ToolifyHandler
from app.models.schemas import OpenAIRequest
from app.utils.logger import get_logger
from app.utils.tool_call_handler import (
    StreamingFunctionCallDetector,
    parse_and_extract_tool_calls,
)

logger = get_logger()


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
        return self.stream_id

    def log_downstream(self, sse_data: str) -> None:
        """记录下游 SSE 调试日志。"""
        self.downstream_count += 1
        logger.debug(
            f"\U0001f4e4 Client SSE #{self.downstream_count}: "
            f"{sse_data.rstrip()[:200]}"
        )

    def process_citation_marker(self, text: str) -> str:
        """过滤 GLM 内部引用标记 (如 【turn9click0】)。

        逐字符处理，遇到 `【` 开始缓冲，匹配完整引用标记后丢弃，
        超长或不匹配时退还缓冲。
        """
        if not text and not self.citation_buf:
            return text

        res = ""
        for char in text:
            if not self.citation_buf:
                if char == "【":
                    self.citation_buf = char
                else:
                    res += char
            else:
                self.citation_buf += char
                if self.citation_buf.endswith("】"):
                    if re.match(
                        r"^【turn\d+(?:search|click)\d+】$",
                        self.citation_buf,
                    ):
                        self.citation_buf = ""  # 完整引用标记，丢弃
                    else:
                        res += self.citation_buf
                        self.citation_buf = ""
                elif len(self.citation_buf) > 30:
                    res += self.citation_buf  # 超长，退还
                    self.citation_buf = ""
                elif (
                    not "【turn".startswith(self.citation_buf)
                    and not self.citation_buf.startswith("【turn")
                ):
                    res += self.citation_buf  # 偏离特征，退还
                    self.citation_buf = ""
        return res


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
            build_tc_func=self._build_tool_call_chunks
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
    # 预编译的关闭标签正则：匹配 </details> 或 </think> 等
    _THINKING_CLOSE_RE = re.compile(
        r"</" + "|".join(f"(?:{t})" for t in _THINKING_TAGS) + r">"
    )

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

        tags_alt = "|".join(cls._THINKING_TAGS)
        is_unclosed = False

        # 1. Strip complete <tag ...>...</tag> blocks
        cleaned = re.sub(
            rf"<(?:{tags_alt})[^>]*>.*?</(?:{tags_alt})>\s*",
            "",
            text,
            flags=re.DOTALL,
        )

        # 2. Strip orphan closing fragment (tail of a tag opened in thinking)
        cleaned = re.sub(
            rf'^[^<]*?(?:true|false)?"?>\s*(?:>\s*.*?)?</(?:{tags_alt})>\s*',
            "",
            cleaned,
            flags=re.DOTALL,
        )

        # 3. Strip orphan opening without closing
        m_open = re.search(
            rf"<(?:{tags_alt})[^>]*>.*$", cleaned, flags=re.DOTALL
        )
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

    # ==================================================================
    # 流式处理: SSE 构建辅助
    # ==================================================================

    def _build_tool_call_chunks(
        self,
        ctx: StreamContext,
        parsed_tools: List[Dict[str, Any]],
    ) -> List[str]:
        """将 XML 解析结果转换为 OpenAI tool_calls SSE chunks。"""
        chunks: List[str] = []
        sid = ctx.ensure_stream_id()

        for i, tool in enumerate(parsed_tools):
            tc = {
                "index": i,
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "arguments": json.dumps(
                        tool["args"], ensure_ascii=False
                    ),
                },
            }
            chunks.append(
                format_sse_chunk(
                    create_openai_chunk(
                        sid, ctx.model, {"tool_calls": [tc]}
                    )
                )
            )
        return chunks

    def _emit_sse(
        self,
        ctx: StreamContext,
        delta: Dict[str, Any],
    ) -> List[str]:
        """构建 SSE chunk 并返回待 yield 的列表 (含可能的 role chunk)。"""
        output: List[str] = []

        if not ctx.has_sent_role:
            ctx.has_sent_role = True
            role_sse = format_sse_chunk(
                create_openai_chunk(
                    ctx.ensure_stream_id(),
                    ctx.model,
                    {"role": "assistant"},
                )
            )
            ctx.log_downstream(role_sse)
            output.append(role_sse)

        sse = format_sse_chunk(
            create_openai_chunk(ctx.ensure_stream_id(), ctx.model, delta)
        )
        ctx.log_downstream(sse)
        output.append(sse)
        return output

    # ==================================================================
    # 流式处理: SSE 行解析
    # ==================================================================

    def _parse_sse_line(
        self, line: str, ctx: StreamContext
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """解析一行 SSE 数据。

        Returns:
            (chunk, data) 二元组，如果行应被跳过则返回 None。
        """
        ctx.line_count += 1
        if not line:
            return None

        self.logger.debug(f"🔍 SSE line #{ctx.line_count}: {line[:200]}")

        current_line = line.strip()
        if not current_line.startswith("data:"):
            if ctx.line_count <= 3:
                self.logger.debug(
                    f"🔍 SSE line #{ctx.line_count} 跳过 (non-data): "
                    f"{current_line[:100]}"
                )
            return None

        chunk_str = current_line[5:].strip()
        if not chunk_str:
            return None

        if chunk_str == "[DONE]":
            return None  # 调用方通过检查 chunk_str 处理 DONE

        try:
            chunk = json.loads(chunk_str)
        except json.JSONDecodeError as error:
            self.logger.debug(
                f"❌ JSON解析错误: {error}, 内容: {chunk_str[:1000]}"
            )
            return None

        chunk_type = chunk.get("type")
        data = (
            chunk.get("data", {})
            if chunk_type == "chat:completion"
            else chunk
        )
        if not isinstance(data, dict):
            return None

        return chunk, data

    def _is_sse_done(self, line: str) -> bool:
        """检查 SSE 行是否为 [DONE] 信号。"""
        stripped = line.strip()
        if stripped.startswith("data:"):
            return stripped[5:].strip() == "[DONE]"
        return False

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
            self.logger.debug(
                f"📈 SSE 阶段: {prev} → {phase}"
            )
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
                if ctx.in_glm_tool_execution:
                    self.logger.debug(
                        f"[glm-tool] 工具执行完毕, 恢复正常输出 (-> {phase})"
                    )
                ctx.in_glm_tool_execution = False

        # Usage
        if data.get("usage"):
            ctx.usage_info = data["usage"]
            self.logger.info(f"[usage] catch data.usage: {ctx.usage_info}")
        elif chunk.get("usage"):
            ctx.usage_info = chunk["usage"]
            self.logger.info(
                f"[usage] catch chunk.usage: {ctx.usage_info}"
            )

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

        current_text = ""
        if delta_content:
            current_text = delta_content
            ctx.buffered_content += delta_content
        elif edit_content:
            current_text = edit_content
            if edit_index is not None and isinstance(edit_index, int):
                safe_idx = max(
                    0, min(edit_index, len(ctx.buffered_content))
                )
                if safe_idx != edit_index:
                    self.logger.debug(
                        f"🔧 edit_index {edit_index} 越界 "
                        f"(buffered={len(ctx.buffered_content)}), "
                        f"截断为 {safe_idx}"
                    )
                ctx.buffered_content = (
                    ctx.buffered_content[:safe_idx]
                    + edit_content
                    + ctx.buffered_content[safe_idx:]
                )
                self.logger.debug(
                    f"🔧 edit_index={edit_index}: 在位置 {safe_idx} 插入 "
                    f"{len(edit_content)} 字符, "
                    f"buffered 总长={len(ctx.buffered_content)}"
                )
            else:
                ctx.buffered_content += edit_content

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
        self.logger.error(
            f"❌ 上游 SSE 返回错误: code={error_code_val}, "
            f"detail={error_detail}"
        )
        error_response = {
            "error": {
                "message": error_detail,
                "type": "upstream_error",
                "code": error_code_val,
            }
        }

        output: List[str] = []
        if not ctx.has_sent_role:
            ctx.has_sent_role = True
            role_sse = format_sse_chunk(
                create_openai_chunk(
                    ctx.ensure_stream_id(),
                    ctx.model,
                    {"role": "assistant"},
                )
            )
            ctx.log_downstream(role_sse)
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
            f"⚠️ 检测到模型重复循环! "
            f"模式: {repeated_pattern[:30]!r}, "
            f"已处理 {ctx.line_count} 行/{ctx.repeat_chunk_count} chunks, "
            f"强制终止流"
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
            m = self._THINKING_CLOSE_RE.search(ctx.details_drain_buf)
            if m:
                remainder = ctx.details_drain_buf[m.end() :].lstrip()
                self.logger.debug(
                    f"🧹 排掉思维残留完成, 剩余内容: {remainder[:80]}..."
                    if remainder
                    else "🧹 排掉思维残留完成, 无剩余内容"
                )
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
                    f"🧹 检测到未闭合思维标签残留, "
                    f"开始排掉: {current_text[:80]}..."
                )

            if not cleaned:
                return ""
            elif cleaned != current_text:
                self.logger.debug(
                    f"🧹 一次性清理思维残留: "
                    f"{len(current_text)}→{len(cleaned)} 字符"
                )
                current_text = cleaned

        # GLM <glm_block> 过滤
        if current_text and "<glm_block" in current_text:
            self.logger.debug(
                f"🧹 过滤 GLM 内部工具调用块: {current_text[:80]}..."
            )
            return ""

        return current_text

    # ==================================================================
    # 流式处理: Toolify 工具解析完毕回调辅助
    # ==================================================================

    def _inject_toolify_role_and_chunks(
        self, ctx: StreamContext, parsed_chunks: List[str]
    ) -> List[str]:
        """为 Toolify 解析出来的 tool_calls 增补 role=assistant (如果尚未发)。"""
        output = []
        if not ctx.has_sent_role and parsed_chunks:
            ctx.has_sent_role = True
            role_sse = format_sse_chunk(
                create_openai_chunk(
                    ctx.ensure_stream_id(),
                    ctx.model,
                    {"role": "assistant"},
                )
            )
            ctx.log_downstream(role_sse)
            output.append(role_sse)
        output.extend(parsed_chunks)
        return output

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
                    f"🧹 跳过 GLM 工具执行期 other 内容: "
                    f"{current_text[:60]}..."
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

        # 最后尝试: 从累积内容中提取工具调用
        if ctx.has_tools and not ctx.tool_calls_accum:
            for sse in self._finalize_tool_extraction(ctx):
                yield sse

        # 确保 role 已发送
        if not ctx.has_sent_role:
            ctx.has_sent_role = True
            role_sse = format_sse_chunk(
                create_openai_chunk(
                    ctx.ensure_stream_id(),
                    ctx.model,
                    {"role": "assistant"},
                )
            )
            ctx.log_downstream(role_sse)
            yield role_sse

        # 刷出残留的引用标记缓冲
        if ctx.citation_buf:
            if not re.match(
                r"^【turn\d*(?:search|click)?\d*$", ctx.citation_buf
            ):
                sse = format_sse_chunk(
                    create_openai_chunk(
                        ctx.ensure_stream_id(),
                        ctx.model,
                        {"content": ctx.citation_buf},
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
                    f"🧹 刷出 detector 残留缓冲: "
                    f"{len(remaining)} 字符: {remaining[:80]}"
                )
                sse = format_sse_chunk(
                    create_openai_chunk(
                        ctx.ensure_stream_id(),
                        ctx.model,
                        {"content": remaining},
                    )
                )
                ctx.log_downstream(sse)
                yield sse

        # finish chunk
        finish_reason = "tool_calls" if ctx.tool_calls_accum else "stop"
        finish_chunk = create_openai_chunk(
            ctx.ensure_stream_id(), ctx.model, {}, finish_reason
        )
        finish_chunk["usage"] = ctx.usage_info
        finish_sse = format_sse_chunk(finish_chunk)
        ctx.log_downstream(finish_sse)
        yield finish_sse
        ctx.log_downstream("data: [DONE]")
        yield "data: [DONE]\n\n"
        ctx.finished = True

    def _finalize_tool_extraction(
        self, ctx: StreamContext
    ) -> List[str]:
        """流结束时从累积内容中提取工具调用 (XML 优先, JSON 降级)。"""
        output: List[str] = []

        # 优先尝试 XML 解析
        if ctx.trigger_signal and ctx.trigger_signal in ctx.buffered_content:
            parsed = parse_function_calls_xml(
                ctx.buffered_content, ctx.trigger_signal
            )
            if parsed:
                validation_err = validate_parsed_tools(
                    parsed, ctx.tools_defs
                )
                if validation_err:
                    self.logger.warning(
                        f"⚠️ 流结束时 Schema 验证失败: {validation_err}"
                    )
                    # 智能降级: 只要有 name + 非空 args 就视为可用
                    usable = [
                        p
                        for p in parsed
                        if p.get("name")
                        and isinstance(p.get("args"), dict)
                        and p["args"]
                    ]
                    if usable:
                        self.logger.warning(
                            f"⚠️ Schema 验证失败但参数非空, "
                            f"强制发送 {len(usable)} 个工具调用"
                        )
                        tc_chunks = self._build_tool_call_chunks(ctx, usable)
                        output.extend(
                            self._emit_sse(
                                ctx, {"role": "assistant"}
                            )[:1]
                            if not ctx.has_sent_role
                            else []
                        )
                        # 确保 role 已发送
                        if not ctx.has_sent_role:
                            ctx.has_sent_role = True
                            role_sse = format_sse_chunk(
                                create_openai_chunk(
                                    ctx.ensure_stream_id(),
                                    ctx.model,
                                    {"role": "assistant"},
                                )
                            )
                            ctx.log_downstream(role_sse)
                            output.append(role_sse)
                        output.extend(tc_chunks)
                        ctx.tool_calls_accum = usable
                else:
                    tc_chunks = self._build_tool_call_chunks(ctx, parsed)
                    if not ctx.has_sent_role:
                        ctx.has_sent_role = True
                        role_sse = format_sse_chunk(
                            create_openai_chunk(
                                ctx.ensure_stream_id(),
                                ctx.model,
                                {"role": "assistant"},
                            )
                        )
                        ctx.log_downstream(role_sse)
                        output.append(role_sse)
                    output.extend(tc_chunks)
                    ctx.tool_calls_accum = parsed

        # 降级: JSON 解析
        if not ctx.tool_calls_accum:
            json_parsed, _ = parse_and_extract_tool_calls(
                ctx.buffered_content
            )
            normalized = self.normalize_tool_calls(json_parsed)
            if normalized:
                ctx.tool_calls_accum = normalized
                if not ctx.has_sent_role:
                    ctx.has_sent_role = True
                    role_sse = format_sse_chunk(
                        create_openai_chunk(
                            ctx.ensure_stream_id(),
                            ctx.model,
                            {"role": "assistant"},
                        )
                    )
                    ctx.log_downstream(role_sse)
                    output.append(role_sse)
                for tool_call in normalized:
                    sse = format_sse_chunk(
                        create_openai_chunk(
                            ctx.ensure_stream_id(),
                            ctx.model,
                            {"tool_calls": [tool_call]},
                        )
                    )
                    output.append(sse)

        return output

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
                f"[stream] upstream success, start handle SSE stream, "
                f"ttfb: {ttfb:.3f}s"
            )
        else:
            self.logger.info(
                "[stream] upstream success, start handle SSE stream"
            )

        sse_start_time = time.perf_counter()

        # 初始化上下文
        trigger_signal = transformed.get("trigger_signal", "")
        tools_defs = transformed.get("tools")
        has_tools = (
            settings.TOOL_SUPPORT
            and bool(request.tools)
            and bool(trigger_signal)
        )

        ctx = StreamContext(
            chat_id=chat_id,
            model=model,
            has_tools=has_tools,
            trigger_signal=trigger_signal,
            tools_defs=tools_defs,
        )

        if has_tools:
            ctx.detector = StreamingFunctionCallDetector(trigger_signal)
            self.logger.debug(
                f"🔧 已初始化流式工具检测器, "
                f"触发信号: {trigger_signal[:20]}..."
            )

        try:
            async for line in response.aiter_lines():
                # 检查 DONE
                if self._is_sse_done(line):
                    break

                # 解析 SSE
                parsed = self._parse_sse_line(line, ctx)
                if parsed is None:
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

                # 上游原生 tool_calls 透传
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

                # Toolify 工具检测 (非 parsing 模式)
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
                    final_output = self._inject_toolify_role_and_chunks(
                        ctx, parsing_result
                    )
                    for sse in final_output:
                        yield sse

                    if ctx.tool_calls_accum:
                        # 解析完成, 自带结流
                        ctx.finished = True
                        finish_chunk = create_openai_chunk(
                            ctx.ensure_stream_id(), ctx.model, {}, "tool_calls"
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
            self.logger.error(f"❌ 流式响应处理错误: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            if not ctx.finished:
                yield format_sse_chunk(
                    create_openai_chunk(
                        ctx.ensure_stream_id(), model, {}, "stop"
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
    ) -> Dict[str, Any]:
        """处理非流式响应，聚合上游 SSE 为一次性 OpenAI 响应。

        集成 XML 解析优先，JSON 解析降级。
        """
        final_content = ""
        reasoning_content = ""
        tool_calls_accum: List[Dict[str, Any]] = []
        usage_info: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        in_glm_tool_execution = False
        last_phase = None

        try:
            async for line in response.aiter_lines():
                if not line:
                    continue

                line = line.strip()
                if not line.startswith("data:"):
                    try:
                        maybe_err = json.loads(line)
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

                data_str = line[5:].strip()
                if not data_str or data_str in (
                    "[DONE]",
                    "DONE",
                    "done",
                ):
                    continue

                try:
                    chunk = json.loads(data_str)
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
                    reasoning_content += self.clean_reasoning_delta(
                        delta_content
                    )

                elif phase == "answer":
                    if delta_content:
                        final_content += delta_content
                    elif edit_content:
                        ec = self.extract_answer_content(edit_content)
                        if edit_index is not None and isinstance(
                            edit_index, int
                        ):
                            safe_idx = max(
                                0, min(edit_index, len(final_content))
                            )
                            final_content = (
                                final_content[:safe_idx]
                                + ec
                                + final_content[safe_idx:]
                            )
                        else:
                            final_content += ec

                elif phase == "other" and edit_content:
                    if not in_glm_tool_execution:
                        ec = self.extract_answer_content(edit_content)
                        if edit_index is not None and isinstance(
                            edit_index, int
                        ):
                            safe_idx = max(
                                0, min(edit_index, len(final_content))
                            )
                            final_content = (
                                final_content[:safe_idx]
                                + ec
                                + final_content[safe_idx:]
                            )
                        else:
                            final_content += ec

                elif phase == "search" or chunk_type == "web_search":
                    final_content += self.format_search_results(data)

                tool_calls_accum.extend(
                    self.normalize_tool_calls(
                        data.get("tool_calls"),
                        len(tool_calls_accum),
                    )
                )

                if data.get("usage"):
                    usage_info = data["usage"]
                elif chunk.get("usage"):
                    usage_info = chunk["usage"]

        except Exception as e:
            self.logger.error(f"❌ 非流式响应处理错误: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return handle_error(e, "非流式聚合")

        # 优先尝试 XML 解析
        if (
            not tool_calls_accum
            and trigger_signal
            and trigger_signal in final_content
        ):
            parsed = parse_function_calls_xml(
                final_content, trigger_signal
            )
            if parsed:
                validation_err = validate_parsed_tools(
                    parsed, tools_defs
                )
                if not validation_err:
                    normalized = []
                    for i, tool in enumerate(parsed):
                        normalized.append(
                            {
                                "index": i,
                                "id": f"call_{uuid.uuid4().hex[:24]}",
                                "type": "function",
                                "function": {
                                    "name": tool["name"],
                                    "arguments": json.dumps(
                                        tool["args"],
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        )
                    if normalized:
                        tool_calls_accum = normalized
                        trigger_pos = final_content.find(trigger_signal)
                        if trigger_pos >= 0:
                            final_content = final_content[
                                :trigger_pos
                            ].strip()
                        self.logger.info(
                            f"[tools] XML parse success: "
                            f"{len(normalized)} tools"
                        )
                else:
                    self.logger.warning(
                        f"⚠️ 非流式 Schema 验证失败: {validation_err}"
                    )

        # 降级: JSON 解析
        if not tool_calls_accum:
            parsed_tool_calls, cleaned_content = (
                parse_and_extract_tool_calls(final_content)
            )
            normalized = self.normalize_tool_calls(parsed_tool_calls)
            if normalized:
                tool_calls_accum = normalized
                final_content = cleaned_content

        final_content = (final_content or "").strip()

        # 清理 GLM 内部工具调用残留
        final_content = re.sub(
            r"<glm_block[^>]*>.*?</glm_block>",
            "",
            final_content,
            flags=re.DOTALL,
        )
        final_content = re.sub(
            r"<glm_block[^>]*>.*",
            "",
            final_content,
            flags=re.DOTALL,
        )
        # 清理完整和末尾截断的引用标记
        final_content = re.sub(
            r"【turn\d+(?:search|click)\d+】", "", final_content
        )
        final_content = re.sub(
            r"【turn\d*(?:search|click)?\d*$", "", final_content
        )

        reasoning_content = (reasoning_content or "").strip()

        if not final_content and reasoning_content:
            final_content = reasoning_content

        return create_openai_response_with_reasoning(
            chat_id,
            model,
            final_content,
            reasoning_content,
            usage_info,
            tool_calls_accum or None,
        )
