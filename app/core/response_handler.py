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
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.openai_compat import (
    create_openai_chunk,
    create_openai_response_with_reasoning,
    format_sse_chunk,
    handle_error,
)
from app.models.schemas import OpenAIRequest
from app.utils.logger import get_logger
from app.utils.tool_call_handler import (
    StreamingFunctionCallDetector,
    looks_like_complete_function_calls,
    parse_and_extract_tool_calls,
    parse_function_calls_xml,
    validate_parsed_tools,
)

logger = get_logger()


class ResponseHandler:
    """上游 SSE 响应转换为 OpenAI 格式的处理器。

    提供流式和非流式两种处理路径，以及辅助的内容清理方法。
    """

    def __init__(self) -> None:
        self.logger = logger

    # ------------------------------------------------------------------
    # 内容辅助方法
    # ------------------------------------------------------------------

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
        r'</(?:' + '|'.join(_THINKING_TAGS) + r')>'
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

        tags_alt = '|'.join(cls._THINKING_TAGS)
        is_unclosed = False

        # 1. Strip complete <tag ...>...</tag> blocks
        cleaned = re.sub(
            rf'<(?:{tags_alt})[^>]*>.*?</(?:{tags_alt})>\s*',
            '',
            text,
            flags=re.DOTALL,
        )

        # 2. Strip orphan closing fragment (tail of a tag opened in thinking)
        #    Matches: arbitrary text before closing > then content then </tag>
        cleaned = re.sub(
            rf'^[^<]*?(?:true|false)?"?>\s*(?:>\s*.*?)?</(?:{tags_alt})>\s*',
            '',
            cleaned,
            flags=re.DOTALL,
        )

        # 3. Strip orphan opening without closing
        m_open = re.search(rf'<(?:{tags_alt})[^>]*>.*$', cleaned, flags=re.DOTALL)
        if m_open:
            is_unclosed = True
            cleaned = cleaned[:m_open.start()]

        final_cleaned = cleaned.strip() if cleaned.strip() != text.strip() else text
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

    # ------------------------------------------------------------------
    # 重复循环检测
    # ------------------------------------------------------------------

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
            # str.count() 是 C 层实现，比手动循环快
            if text.count(pattern) >= min_repeats:
                return pattern

        return None

    def normalize_tool_calls(
        self,
        raw_tool_calls: Any,
        start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        """标准化上游工具调用为 OpenAI 兼容格式。"""
        if not raw_tool_calls:
            return []

        tool_calls = (
            raw_tool_calls if isinstance(raw_tool_calls, list) else [raw_tool_calls]
        )
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

    def format_search_results(self, data: Dict[str, Any]) -> str:
        """将上游搜索结果格式化为可追加的 Markdown 引用。"""
        search_info = (
            data.get("results") or data.get("sources") or data.get("citations")
        )
        if not isinstance(search_info, list) or not search_info:
            return ""

        citations = []
        for index, item in enumerate(search_info, 1):
            if not isinstance(item, dict):
                continue

            title = item.get("title") or item.get("name") or f"Result {index}"
            url = item.get("url") or item.get("link")
            if url:
                citations.append(f"[{index}] [{title}]({url})")

        if not citations:
            return ""

        return "\n\n---\n" + "\n".join(citations)

    # ------------------------------------------------------------------
    # 流式响应处理
    # ------------------------------------------------------------------

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
            self.logger.info(f"[stream] upstream success, start handle SSE stream, ttfb: {ttfb:.3f}s")
        else:
            self.logger.info("[stream] upstream success, start handle SSE stream")

        sse_start_time = time.perf_counter()

        trigger_signal = transformed.get("trigger_signal", "")
        tools_defs = transformed.get("tools")  # 原始工具定义用于验证
        has_tools = settings.TOOL_SUPPORT and bool(request.tools) and bool(trigger_signal)

        buffered_content = ""
        usage_info: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        tool_calls_accum: List[Dict[str, Any]] = []
        has_sent_role = False
        finished = False
        line_count = 0
        downstream_count = 0
        last_phase = None
        stream_id: Optional[str] = None  # PR #8: 统一 stream ID
        # 状态：是否正在“排掉”从 thinking 泄漏到 answer 的 <details> 片段
        draining_details = False
        details_drain_buf = ""
        # 重复循环检测状态
        repeat_buffer = ""     # 最近输出的滑动窗口 (~500 字符)
        repeat_chunk_count = 0  # 非空 chunk 计数

        def log_downstream(sse_data: str) -> None:
            nonlocal downstream_count
            downstream_count += 1
            self.logger.debug(f"\U0001f4e4 Client SSE #{downstream_count}: {sse_data.rstrip()[:200]}")

        # Toolify 流式检测器
        detector: Optional[StreamingFunctionCallDetector] = None
        if has_tools:
            detector = StreamingFunctionCallDetector(trigger_signal)
            self.logger.debug(f"🔧 已初始化流式工具检测器, 触发信号: {trigger_signal[:20]}...")

        def ensure_stream_id(chunk_data: Optional[Dict] = None) -> str:
            nonlocal stream_id
            if stream_id is None:
                upstream_id = chunk_data.get("id") if isinstance(chunk_data, dict) else None
                stream_id = upstream_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
            return stream_id

        async def ensure_role_sent() -> Optional[str]:
            nonlocal has_sent_role
            if has_sent_role:
                return None
            has_sent_role = True
            chunk = format_sse_chunk(
                create_openai_chunk(
                    ensure_stream_id(), model, {"role": "assistant"}
                )
            )
            log_downstream(chunk)
            return chunk

        def build_tool_calls_chunks(
            parsed_tools: List[Dict[str, Any]],
        ) -> List[str]:
            """XML 解析结果转换为 OpenAI tool_calls SSE chunks"""
            chunks: List[str] = []
            sid = ensure_stream_id()

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
                            sid, model, {"tool_calls": [tc]}
                        )
                    )
                )
            return chunks

        async def finalize_stream() -> AsyncGenerator[str, None]:
            nonlocal finished, tool_calls_accum
            if finished:
                return

            # 最后尝试: 从累积内容中提取工具调用
            if has_tools and not tool_calls_accum:
                # 优先尝试 XML 解析
                if trigger_signal and trigger_signal in buffered_content:
                    parsed = parse_function_calls_xml(
                        buffered_content, trigger_signal
                    )
                    if parsed:
                        validation_err = validate_parsed_tools(
                            parsed, tools_defs
                        )
                        if validation_err:
                            self.logger.warning(
                                f"⚠️ 流结束时 Schema 验证失败: {validation_err}"
                            )
                            # 智能降级: 只要每个调用都有 name 且 args 非空，
                            # 就视为可用，忽略 schema 约束（避免因 CDATA 损坏等
                            # 导致正确解析的工具调用被丢弃）
                            usable = [
                                p for p in parsed
                                if p.get("name") and isinstance(p.get("args"), dict) and p["args"]
                            ]
                            if usable:
                                self.logger.warning(
                                    f"⚠️ Schema 验证失败但参数非空, 强制发送 {len(usable)} 个工具调用"
                                )
                                tc_chunks = build_tool_calls_chunks(usable)
                                role_output = await ensure_role_sent()
                                if role_output:
                                    yield role_output
                                for tc_chunk in tc_chunks:
                                    yield tc_chunk
                                tool_calls_accum = usable
                        else:
                            tc_chunks = build_tool_calls_chunks(parsed)
                            role_output = await ensure_role_sent()
                            if role_output:
                                yield role_output
                            for tc_chunk in tc_chunks:
                                yield tc_chunk
                            tool_calls_accum = parsed

                # 降级: JSON 解析
                if not tool_calls_accum:
                    json_parsed, _ = parse_and_extract_tool_calls(
                        buffered_content
                    )
                    normalized = self.normalize_tool_calls(json_parsed)
                    if normalized:
                        tool_calls_accum = normalized
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        for tool_call in normalized:
                            yield format_sse_chunk(
                                create_openai_chunk(
                                    ensure_stream_id(),
                                    model,
                                    {"tool_calls": [tool_call]},
                                )
                            )

            if not has_sent_role:
                role_output = await ensure_role_sent()
                if role_output:
                    yield role_output

            # 刷出 detector 的 look-ahead 缓冲区中被保留的内容
            if detector and detector.state != "tool_parsing" and not tool_calls_accum:
                remaining = detector.flush()
                if remaining:
                    self.logger.debug(
                        f"🧹 刷出 detector 残留缓冲: {len(remaining)} 字符: {remaining[:80]}"
                    )
                    sse = format_sse_chunk(
                        create_openai_chunk(
                            ensure_stream_id(),
                            model,
                            {"content": remaining},
                        )
                    )
                    log_downstream(sse)
                    yield sse

            finish_reason = "tool_calls" if tool_calls_accum else "stop"
            finish_chunk = create_openai_chunk(
                ensure_stream_id(), model, {}, finish_reason
            )
            finish_chunk["usage"] = usage_info
            finish_sse = format_sse_chunk(finish_chunk)
            log_downstream(finish_sse)
            yield finish_sse
            log_downstream("data: [DONE]")
            yield "data: [DONE]\n\n"
            finished = True

        try:
            async for line in response.aiter_lines():
                line_count += 1
                if not line:
                    continue

                # 调试：捕获原始 SSE 行
                self.logger.debug(f"🔍 SSE line #{line_count}: {line[:200]}")

                current_line = line.strip()
                if not current_line.startswith("data:"):
                    if line_count <= 3:
                        self.logger.debug(
                            f"🔍 SSE line #{line_count} 跳过 (non-data): "
                            f"{current_line[:100]}"
                        )
                    continue

                chunk_str = current_line[5:].strip()
                if not chunk_str:
                    continue

                if chunk_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(chunk_str)
                except json.JSONDecodeError as error:
                    self.logger.debug(
                        f"❌ JSON解析错误: {error}, 内容: {chunk_str[:1000]}"
                    )
                    continue

                # 确保 stream_id 统一 (PR #8)
                ensure_stream_id(chunk)

                chunk_type = chunk.get("type")
                data = (
                    chunk.get("data", {}) if chunk_type == "chat:completion" else chunk
                )
                if not isinstance(data, dict):
                    continue

                phase = data.get("phase")
                delta_content = data.get("delta_content", "")
                edit_content = data.get("edit_content", "")
                edit_index = data.get("edit_index")  # 上游指定的插入/替换位置

                # 检测上游 SSE 事件中的错误
                sse_error = data.get("error")
                if isinstance(sse_error, dict):
                    error_code_val = sse_error.get("code", "UNKNOWN")
                    error_detail = sse_error.get("detail", "Unknown upstream error")
                    self.logger.error(
                        f"❌ 上游 SSE 返回错误: code={error_code_val}, detail={error_detail}"
                    )
                    error_response = {
                        "error": {
                            "message": error_detail,
                            "type": "upstream_error",
                            "code": error_code_val,
                        }
                    }
                    role_output = await ensure_role_sent()
                    if role_output:
                        yield role_output
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if phase and phase != last_phase:
                    self.logger.debug(f"📈 SSE 阶段: {last_phase} → {phase}")
                    prev_phase = last_phase  # 保存切换前的阶段
                    last_phase = phase
                else:
                    prev_phase = last_phase

                if data.get("usage"):
                    usage_info = data["usage"]
                    self.logger.info(f"[usage] catch data.usage: {usage_info}")
                elif chunk.get("usage"):
                    usage_info = chunk["usage"]
                    self.logger.info(f"[usage] catch chunk.usage: {usage_info}")

                # 累积内容
                # edit_index: 上游指定在 buffered_content 中的插入位置
                # 若缺失或越界则退回到追加模式（安全兜底）
                current_text = ""
                if delta_content:
                    current_text = delta_content
                    buffered_content += delta_content
                elif edit_content:
                    current_text = edit_content
                    if edit_index is not None and isinstance(edit_index, int):
                        # 防御性校验: 索引必须在合法范围内
                        safe_idx = max(0, min(edit_index, len(buffered_content)))
                        if safe_idx != edit_index:
                            self.logger.debug(
                                f"🔧 edit_index {edit_index} 越界 (buffered={len(buffered_content)}), "
                                f"截断为 {safe_idx}"
                            )
                        # 在指定位置插入（不替换），保留后续内容
                        buffered_content = (
                            buffered_content[:safe_idx]
                            + edit_content
                            + buffered_content[safe_idx:]
                        )
                        self.logger.debug(
                            f"🔧 edit_index={edit_index}: 在位置 {safe_idx} 插入 "
                            f"{len(edit_content)} 字符, buffered 总长={len(buffered_content)}"
                        )
                    else:
                        buffered_content += edit_content

                # 上游原生 tool_calls 直接透传
                direct_tool_calls = self.normalize_tool_calls(
                    data.get("tool_calls"),
                    len(tool_calls_accum),
                )
                if direct_tool_calls:
                    role_output = await ensure_role_sent()
                    if role_output:
                        yield role_output
                    tool_calls_accum.extend(direct_tool_calls)
                    for tool_call in direct_tool_calls:
                        sse = format_sse_chunk(
                            create_openai_chunk(
                                ensure_stream_id(),
                                model,
                                {"tool_calls": [tool_call]},
                            )
                        )
                        log_downstream(sse)
                        yield sse

                # -- 重复循环检测 --
                # 模型可能进入退化重复循环、无限输出相同内容直到 token 上限。
                # 检测到后提前终止流，避免浪费 token。
                if current_text and phase not in ("thinking", "done"):
                    repeat_buffer = (repeat_buffer + current_text)[-500:]
                    repeat_chunk_count += 1
                    if (
                        repeat_chunk_count >= 100
                        and repeat_chunk_count % 30 == 0
                    ):
                        repeated_pattern = self._detect_repetition_loop(
                            repeat_buffer
                        )
                        if repeated_pattern:
                            self.logger.warning(
                                f"⚠️ 检测到模型重复循环! "
                                f"模式: {repeated_pattern[:30]!r}, "
                                f"已处理 {line_count} 行/{repeat_chunk_count} chunks, "
                                f"强制终止流"
                            )
                            role_output = await ensure_role_sent()
                            if role_output:
                                yield role_output
                            error_msg = (
                                "\n\n[ERROR: Model output repetition loop detected, "
                                "stream terminated automatically. Please retry.]"
                            )
                            sse = format_sse_chunk(
                                create_openai_chunk(
                                    ensure_stream_id(),
                                    model,
                                    {"content": error_msg},
                                )
                            )
                            log_downstream(sse)
                            yield sse
                            break

                # -- 思维残留清理（所有非 thinking 阶段都生效）--
                # 上游在 answer 阶段会重新发送 <details type="reasoning" done="true">...
                # 以及 thinking 阶段未关闭的 <details> 标签尾巴也会泄漏进来
                # 此清理必须在 detector 和普通路径之前执行
                if phase != "thinking" and current_text and not (detector and detector.state == "tool_parsing"):
                    if draining_details:
                        details_drain_buf += current_text
                        # 等待任意思维标签的关闭
                        m = self._THINKING_CLOSE_RE.search(details_drain_buf)
                        if m:
                            remainder = details_drain_buf[m.end():].lstrip()
                            self.logger.debug(
                                f"🧹 排掉思维残留完成, 剩余内容: {remainder[:80]}..."
                                if remainder else "🧹 排掉思维残留完成, 无剩余内容"
                            )
                            draining_details = False
                            details_drain_buf = ""
                            if not remainder:
                                continue
                            current_text = remainder
                        else:
                            continue
                    else:
                        # 防范思维标签泄漏到非 thinking 阶段
                        cleaned, is_unclosed = self.strip_thinking_residue(current_text)

                        if is_unclosed:
                            draining_details = True
                            details_drain_buf = current_text
                            self.logger.debug(
                                f"🧹 检测到未闭合思维标签残留, 开始排掉: {current_text[:80]}..."
                            )

                        if not cleaned:
                            continue
                        elif cleaned != current_text:
                            self.logger.debug(
                                f"🧹 一次性清理思维残留: {len(current_text)}→{len(cleaned)} 字符"
                            )
                            current_text = cleaned

                # Toolify 流式工具检测
                # 注意：thinking 阶段不经过 detector，因为 detector 的缓冲会切断
                # <details> 标签导致 clean_reasoning_delta 无法正确清理
                if detector and current_text and detector.state != "tool_parsing" and phase != "thinking":
                    is_detected, content_to_yield = detector.process_chunk(
                        current_text
                    )

                    if is_detected:
                        self.logger.debug(
                            "🔧 流式检测器触发工具调用信号, 切换到解析模式"
                        )
                        # 先输出触发信号前的内容
                        if content_to_yield:
                            role_output = await ensure_role_sent()
                            if role_output:
                                yield role_output
                            sse = format_sse_chunk(
                                create_openai_chunk(
                                    ensure_stream_id(),
                                    model,
                                    {"content": content_to_yield},
                                )
                            )
                            log_downstream(sse)
                            yield sse
                        continue  # 进入工具解析模式，不再输出内容

                    # 未触发，正常输出内容
                    if content_to_yield:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        sse = format_sse_chunk(
                            create_openai_chunk(
                                ensure_stream_id(),
                                model,
                                {"content": content_to_yield},
                            )
                        )
                        log_downstream(sse)
                        yield sse
                    continue

                # 工具解析模式: 累积内容并尝试早期终止
                if detector and detector.state == "tool_parsing" and current_text:
                    detector.content_buffer += current_text
                    if "</function_calls>" in detector.content_buffer:
                        # PR #8: 完整性守卫
                        if not looks_like_complete_function_calls(
                            detector.content_buffer
                        ):
                            self.logger.debug(
                                "🔧 检测到 </function_calls> 但内容不完整, "
                                "继续缓冲"
                            )
                            continue

                        self.logger.debug(
                            "🔧 检测到完整的 </function_calls>, "
                            "开始解析..."
                        )
                        parsed = parse_function_calls_xml(
                            detector.content_buffer, trigger_signal
                        )
                        if parsed:
                            validation_err = validate_parsed_tools(
                                parsed, tools_defs
                            )
                            if validation_err:
                                self.logger.warning(
                                    f"⚠️ 流式工具 Schema 验证失败: "
                                    f"{validation_err}"
                                )
                            else:
                                self.logger.info(
                                    f"[tools] success detect: {len(parsed)} tools"
                                )
                                tc_chunks = build_tool_calls_chunks(parsed)
                                role_output = await ensure_role_sent()
                                if role_output:
                                    yield role_output
                                for tc_chunk in tc_chunks:
                                    yield tc_chunk
                                tool_calls_accum = parsed
                                # 直接结束流
                                finished = True
                                finish_chunk = create_openai_chunk(
                                    ensure_stream_id(),
                                    model,
                                    {},
                                    "tool_calls",
                                )
                                finish_chunk["usage"] = usage_info
                                yield format_sse_chunk(finish_chunk)
                                yield "data: [DONE]\n\n"
                                return
                        else:
                            self.logger.warning(
                                "⚠️ 检测到 </function_calls> 但 XML "
                                "解析失败, 继续缓冲"
                            )
                    continue

                # 没有 detector 或 detector 未活动时的正常处理
                if phase == "thinking" and delta_content:
                    cleaned = self.clean_reasoning_delta(delta_content)
                    if cleaned:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        sse = format_sse_chunk(
                            create_openai_chunk(
                                ensure_stream_id(),
                                model,
                                {"reasoning_content": cleaned},
                            )
                        )
                        log_downstream(sse)
                        yield sse

                elif phase in ("answer", "other") and current_text:
                    role_output = await ensure_role_sent()
                    if role_output:
                        yield role_output
                    sse = format_sse_chunk(
                        create_openai_chunk(
                            ensure_stream_id(),
                            model,
                            {"content": current_text},
                        )
                    )
                    log_downstream(sse)
                    yield sse

                elif phase == "search" or chunk_type == "web_search":
                    citation_text = self.format_search_results(data)
                    if citation_text:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        sse = format_sse_chunk(
                            create_openai_chunk(
                                ensure_stream_id(),
                                model,
                                {"content": citation_text},
                            )
                        )
                        log_downstream(sse)
                        yield sse

                if data.get("done"):
                    break

            if not finished:
                async for final_chunk in finalize_stream():
                    yield final_chunk

        except Exception as e:
            self.logger.error(f"❌ 流式响应处理错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            if not finished:
                yield format_sse_chunk(
                    create_openai_chunk(
                        ensure_stream_id(), model, {}, "stop"
                    )
                )
                yield "data: [DONE]\n\n"
        finally:
            elapsed = time.perf_counter() - sse_start_time
            total_elapsed = time.perf_counter() - start_time if start_time else None
            if total_elapsed is not None:
                self.logger.info(
                    f"[stream] SSE done {line_count} lines, SSE time: {elapsed:.3f}s, total time: {total_elapsed:.3f}s"
                )
            else:
                self.logger.info(
                    f"[stream] SSE done {line_count} lines, SSE time: {elapsed:.3f}s"
                )

    # ------------------------------------------------------------------
    # 非流式响应处理
    # ------------------------------------------------------------------

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
                                if isinstance(maybe_err.get("error"), dict)
                                else maybe_err.get("message")
                            ) or "上游返回错误"
                            return handle_error(Exception(msg), "API响应")
                    except Exception:
                        pass
                    continue

                data_str = line[5:].strip()
                if not data_str or data_str in ("[DONE]", "DONE", "done"):
                    continue

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                chunk_type = chunk.get("type")
                data = (
                    chunk.get("data", {}) if chunk_type == "chat:completion" else chunk
                )
                if not isinstance(data, dict):
                    continue

                phase = data.get("phase")
                delta_content = data.get("delta_content", "")
                edit_content = data.get("edit_content", "")
                edit_index = data.get("edit_index")  # 上游指定的插入位置

                if phase == "thinking" and delta_content:
                    reasoning_content += self.clean_reasoning_delta(delta_content)

                elif phase == "answer":
                    if delta_content:
                        final_content += delta_content
                    elif edit_content:
                        ec = self.extract_answer_content(edit_content)
                        if edit_index is not None and isinstance(edit_index, int):
                            safe_idx = max(0, min(edit_index, len(final_content)))
                            final_content = final_content[:safe_idx] + ec + final_content[safe_idx:]
                        else:
                            final_content += ec

                elif phase == "other" and edit_content:
                    ec = self.extract_answer_content(edit_content)
                    if edit_index is not None and isinstance(edit_index, int):
                        safe_idx = max(0, min(edit_index, len(final_content)))
                        final_content = final_content[:safe_idx] + ec + final_content[safe_idx:]
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

        except Exception as e:
            self.logger.error(f"❌ 非流式响应处理错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return handle_error(e, "非流式聚合")

        # 优先尝试 XML 解析
        if not tool_calls_accum and trigger_signal and trigger_signal in final_content:
            parsed = parse_function_calls_xml(final_content, trigger_signal)
            if parsed:
                validation_err = validate_parsed_tools(parsed, tools_defs)
                if not validation_err:
                    # 将解析结果转换为 OpenAI 格式
                    normalized = []
                    for i, tool in enumerate(parsed):
                        normalized.append({
                            "index": i,
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "arguments": json.dumps(
                                    tool["args"], ensure_ascii=False
                                ),
                            },
                        })
                    if normalized:
                        tool_calls_accum = normalized
                        # 从内容中移除工具调用部分
                        trigger_pos = final_content.find(trigger_signal)
                        if trigger_pos >= 0:
                            final_content = final_content[:trigger_pos].strip()
                        self.logger.info(
                            f"[tools] XML parse success: {len(normalized)} tools"
                        )
                else:
                    self.logger.warning(
                        f"⚠️ 非流式 Schema 验证失败: {validation_err}"
                    )

        # 降级: JSON 解析
        if not tool_calls_accum:
            parsed_tool_calls, cleaned_content = parse_and_extract_tool_calls(
                final_content
            )
            normalized = self.normalize_tool_calls(parsed_tool_calls)
            if normalized:
                tool_calls_accum = normalized
                final_content = cleaned_content

        final_content = (final_content or "").strip()
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
