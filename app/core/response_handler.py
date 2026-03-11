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
            self.logger.info(f"上游响应成功，开始处理 SSE 流，响应成功耗时: {ttfb:.3f}s")
        else:
            self.logger.info("上游响应成功，开始处理 SSE 流")

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
        last_phase = None
        stream_id: Optional[str] = None  # PR #8: 统一 stream ID
        # 状态：是否正在“排掉”从 thinking 泄漏到 answer 的 <details> 片段
        draining_details = False
        details_drain_buf = ""

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
            return format_sse_chunk(
                create_openai_chunk(
                    ensure_stream_id(), model, {"role": "assistant"}
                )
            )

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

            finish_reason = "tool_calls" if tool_calls_accum else "stop"
            finish_chunk = create_openai_chunk(
                ensure_stream_id(), model, {}, finish_reason
            )
            finish_chunk["usage"] = usage_info
            yield format_sse_chunk(finish_chunk)
            yield "data: [DONE]\n\n"
            finished = True

        try:
            async for line in response.aiter_lines():
                line_count += 1
                if not line:
                    continue

                # 调试：捕获原始 SSE 行
                if line_count <= 3:
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
                    self.logger.debug(f"🔍 [usage] 解析到 data.usage: {usage_info}")
                elif chunk.get("usage"):
                    usage_info = chunk["usage"]
                    self.logger.debug(f"🔍 [usage] 解析到 chunk.usage: {usage_info}")

                # 累积内容
                current_text = ""
                if delta_content:
                    current_text = delta_content
                    buffered_content += delta_content
                elif edit_content:
                    current_text = edit_content
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
                        yield format_sse_chunk(
                            create_openai_chunk(
                                ensure_stream_id(),
                                model,
                                {"tool_calls": [tool_call]},
                            )
                        )

                # Toolify 流式工具检测
                # 注意：thinking 阶段不经过 detector，因为 detector 的缓冲会切断
                # <details> 标签导致 clean_reasoning_delta 无法正确清理
                if detector and current_text and detector.state != "tool_parsing" and phase != "thinking":
                    # -- 排掉从 thinking 泄漏到 answer 的 <details> 残留 --
                    # 上游会在 answer 阶段重新发送 <details type="reasoning" done="true">...
                    # 并且 thinking 阶段未关闭的 <details> 标签尾巴也会泄漏进来
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
                        # 只要不在 thinking 阶段且没在 draining，就随时防范思维标签泄漏
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
                            yield format_sse_chunk(
                                create_openai_chunk(
                                    ensure_stream_id(),
                                    model,
                                    {"content": content_to_yield},
                                )
                            )
                        continue  # 进入工具解析模式，不再输出内容

                    # 未触发，正常输出内容
                    if content_to_yield:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        yield format_sse_chunk(
                            create_openai_chunk(
                                ensure_stream_id(),
                                model,
                                {"content": content_to_yield},
                            )
                        )
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
                                    f"流式检测解析成功: "
                                    f"{len(parsed)} 个工具调用"
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
                        yield format_sse_chunk(
                            create_openai_chunk(
                                ensure_stream_id(),
                                model,
                                {"reasoning_content": cleaned},
                            )
                        )

                elif phase == "answer":
                    text = delta_content or self.extract_answer_content(
                        edit_content
                    )
                    if text:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        yield format_sse_chunk(
                            create_openai_chunk(
                                ensure_stream_id(),
                                model,
                                {"content": text},
                            )
                        )

                elif phase == "other":
                    other_text = self.extract_answer_content(edit_content)
                    if other_text:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        yield format_sse_chunk(
                            create_openai_chunk(
                                ensure_stream_id(),
                                model,
                                {"content": other_text},
                            )
                        )

                elif phase == "search" or chunk_type == "web_search":
                    citation_text = self.format_search_results(data)
                    if citation_text:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        yield format_sse_chunk(
                            create_openai_chunk(
                                ensure_stream_id(),
                                model,
                                {"content": citation_text},
                            )
                        )

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
                    f"SSE 流处理完成，共处理 {line_count} 行数据，处理SSE耗时: {elapsed:.3f}s，总响应耗时: {total_elapsed:.3f}s"
                )
            else:
                self.logger.info(
                    f"SSE 流处理完成，共处理 {line_count} 行数据，处理SSE耗时: {elapsed:.3f}s"
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

                if data.get("usage"):
                    usage_info = data["usage"]
                    self.logger.info(f"🔍 [Non-Stream] 解析到 data.usage: {usage_info}")
                elif chunk.get("usage"):
                    usage_info = chunk["usage"]
                    self.logger.info(f"🔍 [Non-Stream] 解析到 chunk.usage: {usage_info}")

                if phase == "thinking" and delta_content:
                    reasoning_content += self.clean_reasoning_delta(delta_content)

                elif phase == "answer":
                    if delta_content:
                        final_content += delta_content
                    elif edit_content:
                        final_content += self.extract_answer_content(edit_content)

                elif phase == "other" and edit_content:
                    final_content += self.extract_answer_content(edit_content)

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
                            f"✅ 非流式 XML 解析成功: {len(normalized)} 个工具调用"
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
