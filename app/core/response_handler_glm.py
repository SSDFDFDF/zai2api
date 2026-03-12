#!/usr/import/env python
# -*- coding: utf-8 -*-

"""GLM 内部工具调用提示处理器。

负责在 GLM 调用内置工具(如 search, browser, Bash)时，
向客户端发送可视化的开始与完成提示。
"""

import re
from typing import Any, Dict, List, Optional
from app.utils.logger import get_logger

logger = get_logger()


class GLMToolHandler:
    """GLM 内部工具调用提示处理器"""

    # 工具名 -> 显示名映射 (不使用 emoji, 保证环境兼容)
    _GLM_TOOL_DISPLAY: Dict[str, str] = {
        "search": "[搜索]",
        "retrieve": "[联网检索]",
        "open": "[打开网页]",
        "Bash": "[代码执行]",
        "Skill": "[技能调用]",
        "browser": "[浏览器]",
    }

    _GLM_BLOCK_NAME_RE = re.compile(r'tool_call_name="([^"]+)"')

    def __init__(self, enabled: bool = False, emit_func=None):
        self.enabled = enabled
        self.emit_func = emit_func  # 用于发出 SSE 的回调: emit_func(ctx, delta_dict)

    @staticmethod
    def _glm_tool_display_name(tool_name: str) -> str:
        """将 GLM 内部工具名映射为用户可读的显示名。"""
        display = GLMToolHandler._GLM_TOOL_DISPLAY.get(tool_name)
        if display:
            return display
        return f"[{tool_name}]" if tool_name else "[工具]"

    def process(self, ctx: Any, data: Dict[str, Any]) -> Optional[List[str]]:
        """GLM 内部工具调用时, 向客户端发送状态提示。

        根据 phase_before_tool 决定发 reasoning_content 还是 content:
        - 前置阶段为 thinking -> reasoning_content
        - 其余 -> content

        Returns:
            要 yield 的 SSE 列表, 或 None (无需处理)。
        """
        if not self.enabled:
            return None

        phase = ctx.last_phase

        # -- tool_call 阶段: 提取工具名并发送开始提示 --
        if phase == "tool_call":
            if not ctx.glm_tool_name:
                # 新版流: delta_name 字段
                name = data.get("delta_name", "")
                if not name:
                    # 旧版流: <glm_block tool_call_name="..."> 中提取
                    ec = data.get("edit_content", "")
                    if ec:
                        m = self._GLM_BLOCK_NAME_RE.search(ec)
                        if m:
                            name = m.group(1)
                if name:
                    ctx.glm_tool_name = name

            if ctx.glm_tool_name and not ctx.glm_tool_hint_sent:
                ctx.glm_tool_hint_sent = True
                display = self._glm_tool_display_name(ctx.glm_tool_name)
                hint = f"\n> 正在调用 {display} ...\n"
                key = (
                    "reasoning_content"
                    if ctx.phase_before_tool == "thinking"
                    else "content"
                )
                logger.debug(
                    f"[glm-tool] 发送工具提示: {display} -> {key}"
                )
                return self.emit_func(ctx, {key: hint})
            return None

        # -- tool_response 阶段: 发送完成提示 (新版流) --
        if phase == "tool_response":
            tool_name = data.get("tool_name", ctx.glm_tool_name)
            status = data.get("status", "")
            display = self._glm_tool_display_name(tool_name)

            if status == "completed":
                hint = f"> {display} 已完成\n\n"
            else:
                hint = f"> {display}: {status}\n\n"

            key = (
                "reasoning_content"
                if ctx.phase_before_tool == "thinking"
                else "content"
            )
            logger.debug(
                f"[glm-tool] 发送完成提示: {display} -> {key}"
            )
            result = self.emit_func(ctx, {key: hint})
            # 重置, 为下一轮工具调用做准备
            ctx.glm_tool_name = ""
            ctx.glm_tool_hint_sent = False
            return result

        # 当 hint 已发但未收到 tool_response, 且阶段切到 answer/thinking
        if (
            ctx.glm_tool_hint_sent
            and phase in ("answer", "thinking")
            and not ctx.in_glm_tool_execution
        ):
            display = self._glm_tool_display_name(
                ctx.glm_tool_name or ""
            )
            hint = f"> {display} 已完成\n\n"
            key = (
                "reasoning_content"
                if ctx.phase_before_tool == "thinking"
                else "content"
            )
            logger.debug(
                f"[glm-tool] 发送完成提示: {display} -> {key}"
            )
            result = self.emit_func(ctx, {key: hint})
            ctx.glm_tool_name = ""
            ctx.glm_tool_hint_sent = False
            return result

        return None
