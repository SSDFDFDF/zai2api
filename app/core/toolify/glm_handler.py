#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GLM 内部工具调用处理器。

负责：
1. 工具执行时向客户端发送可视化的开始与完成提示（hint）。
2. native/hybrid 策略下，从 <glm_block> 中提取工具调用并转为 OpenAI tool_calls 格式。
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional
from app.utils.logger import get_logger

logger = get_logger()


class GLMToolHandler:
    """GLM 内部工具调用处理器"""

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
    _GLM_BLOCK_EXTRACT_RE = re.compile(
        r'<glm_block\s+tool_call_name="([^"]+)">(.*?)</glm_block>',
        flags=re.DOTALL,
    )

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

    # ------------------------------------------------------------------
    # 工具提示 (hint)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # native/hybrid: <glm_block> → OpenAI tool_calls
    # ------------------------------------------------------------------

    @classmethod
    def parse_tool_calls(
        cls,
        content: str,
        allowed_names: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """从文本中解析所有 <glm_block> 为 OpenAI tool_calls 格式。

        Args:
            content: 待解析的文本（buffered_content 或 final_content）。
            allowed_names: 客户端请求的工具名集合，仅匹配的才转为 tool_calls。
                None 表示不过滤。
        """
        tool_calls: List[Dict[str, Any]] = []
        for match in cls._GLM_BLOCK_EXTRACT_RE.finditer(content):
            tool_name = match.group(1)
            if allowed_names is not None and tool_name not in allowed_names:
                logger.debug(
                    "⏭️ 跳过非客户端请求的 GLM 内部工具: {}",
                    tool_name,
                )
                continue
            block_json_str = match.group(2)
            try:
                block_data = json.loads(block_json_str)
                metadata = block_data.get("data", {}).get("metadata", {})
                call_id = metadata.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                arguments = metadata.get("arguments", "{}")
                tool_calls.append({
                    "index": len(tool_calls),
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments,
                    },
                })
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.debug(
                    "⚠️ 解析 glm_block 失败, tool_name={}: {}...",
                    tool_name,
                    block_json_str[:80],
                )
        return tool_calls

    @staticmethod
    def _extract_tool_names(tools_defs: Optional[List[Dict[str, Any]]]) -> Optional[set]:
        """从客户端 tools 定义列表中提取工具名集合。"""
        if not tools_defs:
            return None
        names = set()
        for t in tools_defs:
            func = t.get("function") if isinstance(t, dict) else None
            if isinstance(func, dict) and func.get("name"):
                names.add(func["name"])
        return names or None

    def handle_native_extraction(self, ctx: Any) -> List[str]:
        """native/hybrid 模式下，tool_call 阶段结束后提取 GLM 工具调用。

        仅提取客户端请求中声明的工具，跳过 GLM 内部工具（如 search、browser）。
        """
        if not ctx.glm_tool_calls_pending:
            return []
        ctx.glm_tool_calls_pending = False

        allowed = self._extract_tool_names(ctx.tools_defs)
        tool_calls = self.parse_tool_calls(ctx.buffered_content, allowed_names=allowed)
        if not tool_calls:
            return []

        logger.info(
            "[glm-native] 从 buffered_content 提取 {} 个工具调用",
            len(tool_calls),
        )
        ctx.tool_calls_accum.extend(tool_calls)
        output: List[str] = []
        for tc in tool_calls:
            output.extend(
                self.emit_func(ctx, {"tool_calls": [tc]})
            )
        return output
