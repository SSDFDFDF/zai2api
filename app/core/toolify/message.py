#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenAI 消息预处理。

将 OpenAI 格式的消息列表规范化为上游服务可接受的形式，
包括 tool 角色转换、assistant tool_calls 序列化等。
"""

import json
from typing import Any, Dict, List

from app.core.toolify.xml_protocol import format_assistant_tool_calls_for_ai


# ---------------------------------------------------------------------------
# 消息内容工具
# ---------------------------------------------------------------------------


def _extract_text_from_content(content: Any) -> str:
    """Extract text parts from OpenAI-compatible content payloads."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(part for part in parts if part).strip()

    if content is None:
        return ""

    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _stringify_tool_arguments(arguments: Any) -> str:
    """Normalize tool-call arguments into a JSON string."""
    if isinstance(arguments, str):
        return arguments

    try:
        return json.dumps(arguments or {}, ensure_ascii=False)
    except Exception:
        return "{}"


# ---------------------------------------------------------------------------
# 工具调用索引与格式化
# ---------------------------------------------------------------------------


def _build_tool_call_index(
    messages: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Index assistant tool calls by id for later tool-result messages."""
    index: Dict[str, Dict[str, str]] = {}

    for message in messages:
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            tool_call_id = tool_call.get("id")
            function_data = (
                tool_call.get("function")
                if isinstance(tool_call.get("function"), dict)
                else {}
            )
            name = str(function_data.get("name", "")).strip()
            if not isinstance(tool_call_id, str) or not name:
                continue

            index[tool_call_id] = {
                "name": name,
                "arguments": _stringify_tool_arguments(
                    function_data.get("arguments")
                ),
            }

    return index


def _format_tool_result_message(
    tool_name: str,
    tool_arguments: str,
    result_content: str,
) -> str:
    """Serialize a tool result into a compact XML block the upstream can consume.

    - Uses ``<tool_response>`` XML wrapper instead of natural-language headers
      to reduce the chance of upstream models echoing the result verbatim.
    - Omits ``tool_arguments`` (the model already knows what it called).
    - Truncates ``result_content`` exceeding ``TOOL_RESULT_MAX_LENGTH``.
    """
    from app.core.config import settings

    max_len = settings.TOOL_RESULT_MAX_LENGTH
    if max_len and len(result_content) > max_len:
        result_content = result_content[:max_len] + "\n... [truncated]"
    return (
        f'<tool_response tool="{tool_name}">\n'
        f"{result_content}\n"
        f"</tool_response>"
    )


def _format_assistant_tool_calls(
    tool_calls: List[Dict[str, Any]],
    trigger_signal: str = "",
) -> str:
    """Serialize historical assistant tool calls into Toolify XML format."""
    if not tool_calls:
        return ""
    return format_assistant_tool_calls_for_ai(tool_calls, trigger_signal)


# ---------------------------------------------------------------------------
# 消息预处理主入口
# ---------------------------------------------------------------------------


def preprocess_openai_messages(
    messages: List[Dict[str, Any]],
    trigger_signal: str = "",
) -> List[Dict[str, Any]]:
    """Normalize OpenAI history into shapes accepted by the upstream service.

    处理以下转换：
    - ``developer`` 角色 → ``system``
    - ``tool`` 角色 → ``user``（将工具结果序列化为文本块）
    - 带有 ``tool_calls`` 的 ``assistant`` → 合并内容与工具调用 XML 序列化

    Args:
        messages: OpenAI 格式的消息列表（已 model_dump）。
        trigger_signal: Toolify XML 触发信号(可选)。

    Returns:
        上游服务可接受的消息列表。
    """
    tool_call_index = _build_tool_call_index(messages)
    normalized: List[Dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")

        if role == "developer":
            converted = dict(message)
            converted["role"] = "system"
            normalized.append(converted)
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            content = _extract_text_from_content(message.get("content"))
            tool_info = tool_call_index.get(
                tool_call_id,
                {
                    "name": str(message.get("name") or "unknown_tool"),
                    "arguments": "{}",
                },
            )
            normalized.append(
                {
                    "role": "user",
                    "content": _format_tool_result_message(
                        tool_info["name"],
                        tool_info["arguments"],
                        content,
                    ),
                }
            )
            continue

        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            content = _extract_text_from_content(message.get("content"))
            tool_calls_text = _format_assistant_tool_calls(
                message["tool_calls"],
                trigger_signal=trigger_signal,
            )
            merged_content = "\n".join(
                part for part in (content, tool_calls_text) if part
            ).strip()
            normalized.append({"role": "assistant", "content": merged_content})
            continue

        normalized.append(dict(message))

    return normalized


def extract_last_user_text(messages: List[Dict[str, Any]]) -> str:
    """Extract the last user text from the original OpenAI message history."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = _extract_text_from_content(message.get("content"))
        if content:
            return content
    return ""
