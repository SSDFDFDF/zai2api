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
from app.utils.logger import get_logger
from .message import preprocess_openai_messages
from .xml_protocol import (
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
    tool_strategy: str = "xmlfc"


class ToolifyRequestHandler:
    """Toolify 请求侧改写。"""

    def prepare(
        self,
        raw_messages: List[Dict[str, Any]],
        request_tools: Any,
        tool_choice: Any,
    ) -> ToolifyPreparedRequest:
        strategy = settings.TOOL_STRATEGY

        if strategy == "disabled" or not request_tools:
            # disabled 或无 tools：仅做消息规范化
            normalized_messages = preprocess_openai_messages(
                raw_messages, trigger_signal="",
            )
            return ToolifyPreparedRequest(
                tools=None,
                tool_choice=None,
                trigger_signal="",
                normalized_messages=normalized_messages,
                tool_strategy=strategy,
            )

        if strategy == "native":
            # native：透传 tools，不注入 XML
            normalized_messages = preprocess_openai_messages(
                raw_messages, trigger_signal="",
            )
            return ToolifyPreparedRequest(
                tools=request_tools,
                tool_choice=tool_choice,
                trigger_signal="",
                normalized_messages=normalized_messages,
                tool_strategy=strategy,
            )

        # xmlfc / hybrid：生成 trigger_signal + XML 注入
        tools = request_tools
        trigger_signal = generate_trigger_signal()
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
            tool_strategy=strategy,
        )
