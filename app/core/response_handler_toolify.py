#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Toolify 第三方工具检测与解析处理器。

负责在流式和非流式响应中检测、累积和解析 `<function_calls>`
XML 格式的工具调用结构。
"""

from typing import Any, Dict, List, Optional
from app.utils.logger import get_logger
from app.utils.tool_call_handler import (
    StreamingFunctionCallDetector,
    looks_like_complete_function_calls,
    parse_and_extract_tool_calls,
    parse_function_calls_xml,
    validate_parsed_tools,
)

logger = get_logger()


class ToolifyHandler:
    """Toolify 流式第三方 XML 工具调用处理器"""

    def __init__(self, emit_func=None, build_tc_func=None):
        """
        Args:
            emit_func: 回调函数, emit_func(ctx, delta_dict) -> List[str]
            build_tc_func: 回调函数, build_tc_func(ctx, parsed_tools) -> List[str]
        """
        self.emit_func = emit_func
        self.build_tc_func = build_tc_func

    def handle_detection(
        self, ctx: Any, current_text: str
    ) -> Optional[List[str]]:
        """Toolify 流式工具检测 (非 tool_parsing 模式)。

        Returns:
            要 yield 的 SSE 列表，或 None 表示不处理 (走后续阶段逻辑)。
            返回空列表 [] 表示已处理但无内容输出。
        """
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
                output.extend(
                    self.emit_func(ctx, {"content": content_to_yield})
                )
            return output  # 进入工具解析模式

        # 未触发，正常输出
        if content_to_yield and self.emit_func:
            return self.emit_func(ctx, {"content": content_to_yield})
        return []  # 已处理，无内容输出

    def handle_parsing(
        self, ctx: Any, current_text: str
    ) -> Optional[List[str]]:
        """累积工具调用 XML 并尝试解析。

        Returns:
            要 yield 的 SSE 列表 (解析成功时)，
            空列表 [] (继续缓冲)，
            或 None (当前不在 tool_parsing 状态)。
        """
        detector = getattr(ctx, "detector", None)
        if not detector or detector.state != "tool_parsing" or not current_text:
            return None

        detector.content_buffer += current_text

        if "</function_calls>" not in detector.content_buffer:
            return []  # 继续缓冲

        # 完整性守卫
        if not looks_like_complete_function_calls(detector.content_buffer):
            logger.debug("🔧 检测到 </function_calls> 但内容不完整, 继续缓冲")
            return []

        logger.debug("🔧 检测到完整的 </function_calls>, 开始解析...")
        parsed = parse_function_calls_xml(
            detector.content_buffer, ctx.trigger_signal
        )
        if not parsed:
            logger.warning("⚠️ 检测到 </function_calls> 但 XML 解析失败, 继续缓冲")
            return []

        validation_err = validate_parsed_tools(parsed, ctx.tools_defs)
        if validation_err:
            logger.warning(f"⚠️ 流式工具 Schema 验证失败: {validation_err}")
            return []

        logger.info(f"[tools] success detect: {len(parsed)} tools")

        output: List[str] = []
        if self.build_tc_func:
            tc_chunks = self.build_tc_func(ctx, parsed)
            if not ctx.has_sent_role:
                ctx.has_sent_role = True
                # 这里我们假设 response_handler 的主循环会注入这段逻辑
                # 但为了封装性更好，我们把 role 的注入留在 _handle_tool_parsing 的外层或者传递进去。
                # 由于这是重构，我们可以把这个通过回调解决，或者在调用的地方补充。
                pass  # 在外部处理 role
            output.extend(tc_chunks)
            ctx.tool_calls_accum = parsed

        return output
