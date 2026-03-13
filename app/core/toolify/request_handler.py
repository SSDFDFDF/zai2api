#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Toolify 请求侧处理器。

负责将 OpenAI 请求中的 tools/tool_choice 转换为 Toolify XML 方案所需数据：
- 选择是否启用 tools
- 生成 trigger_signal
- 预处理消息并注入 XML 工具提示词
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.message_preprocessing import preprocess_openai_messages
from app.utils.logger import get_logger
from app.utils.tool_call_handler import (
    generate_trigger_signal,
    process_messages_with_tools,
)

logger = get_logger()


@dataclass
class ToolifyPreparedRequest:
    """Toolify 请求改写结果。"""

    tools: Optional[List[Dict[str, Any]]]
    tool_choice: Any
    trigger_signal: str
    normalized_messages: List[Dict[str, Any]]


class ToolifyRequestHandler:
    """Toolify 请求侧改写。"""

    def prepare(
        self,
        raw_messages: List[Dict[str, Any]],
        request_tools: Any,
        tool_choice: Any,
    ) -> ToolifyPreparedRequest:
        tools = request_tools if settings.TOOL_SUPPORT and request_tools else None
        trigger_signal = generate_trigger_signal() if tools else ""

        if trigger_signal:
            logger.debug("🔧 生成 XML 触发信号: {}", trigger_signal)

        normalized_messages = preprocess_openai_messages(
            raw_messages,
            trigger_signal=trigger_signal,
        )
        normalized_messages = process_messages_with_tools(
            normalized_messages,
            tools=tools,
            tool_choice=tool_choice,
            trigger_signal=trigger_signal,
        )

        return ToolifyPreparedRequest(
            tools=tools,
            tool_choice=tool_choice,
            trigger_signal=trigger_signal,
            normalized_messages=normalized_messages,
        )

