#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""XMLFC prompt generation and request-side message injection."""

import json
import secrets
import string
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger()


def generate_trigger_signal() -> str:
    """生成随机的自闭合 XML 触发信号，如 <Function_AB1c_Start/>。"""
    chars = string.ascii_letters + string.digits
    random_str = "".join(secrets.choice(chars) for _ in range(4))
    return f"<Function_{random_str}_Start/>"


def generate_tool_prompt(
    tools: Optional[List[Dict[str, Any]]],
    trigger_signal: str,
) -> str:
    """生成 XML 格式的工具调用提示词。"""
    if not tools:
        return ""

    tools_list_str = []
    for i, tool in enumerate(tools):
        if tool.get("type") != "function":
            continue

        func = tool.get("function", {})
        name = func.get("name", "unknown")
        description = func.get("description", "")
        schema: Dict[str, Any] = func.get("parameters", {})

        props = schema.get("properties", {})
        if not isinstance(props, dict):
            props = {}

        required_list = schema.get("required", [])
        if not isinstance(required_list, list):
            required_list = []
        required_list = [k for k in required_list if isinstance(k, str)]

        desc_block = description or "None"
        try:
            schema_block = json.dumps(schema or {}, ensure_ascii=False, indent=2)
        except Exception:
            schema_block = "{}"

        tools_list_str.append(
            f'{i + 1}. <tool name="{name}">\n'
            f"   Description: {desc_block}\n"
            f"   Required parameters: "
            f"{', '.join(required_list) if required_list else 'None'}\n"
            f"   JSON schema:\n```json\n{schema_block}\n```"
        )

    if not tools_list_str:
        return ""

    tools_block = "\n\n".join(tools_list_str)

    prompt = f"""
[XML TOOL CALL PROTOCOL]

Available tools:

{tools_block}

Turn decision rules:
- If no tool is needed, answer normally.
- If any tool is needed, this assistant turn MUST be a pure tool turn.
- A pure tool turn contains only the trigger line and one <function_calls> block.
- Never mix tool XML with natural-language text, reasoning, markdown, status updates, or explanations in the same assistant turn.

Tool result context:
- Previous tool outputs may appear as <tool_response tool="...">...</tool_response>.
- Treat them as machine-readable prior results.
- Use the information, but do not reproduce the raw XML wrapper or dump long raw output verbatim.

Required output format for a pure tool turn:
1. Output exactly this trigger line on its own line:
{trigger_signal}
2. On the very next line, immediately output one complete <function_calls> block.
3. If multiple tools are needed, put every <function_call> inside the same <function_calls> wrapper.
4. No text before the trigger.
5. No text after </function_calls>.
6. Do not output a second trigger line.

Argument rules:
- Use parameter keys EXACTLY as defined. Keep case, punctuation, and leading hyphens unchanged.
- The <tool> value must exactly match one available tool name.
- Use <args_json> for ordinary JSON-safe values.
- Use <args_kv> for multiline strings, code, quoted text, or XML-like text that is risky to escape inside JSON.
- In <args_kv>, encode each value as <arg name="EXACT_KEY">VALUE</arg>.
- Use <![CDATA[...]]> inside <arg> when the value contains code, quotes, or multiple lines.
- You may combine <args_json> and <args_kv> in the same <function_call>.

Correct example:
{trigger_signal}
<function_calls>
    <function_call>
        <tool>Grep</tool>
        <args_json><![CDATA[{{"-i": true, "-C": 2, "path": "."}}]]></args_json>
    </function_call>
    <function_call>
        <tool>search</tool>
        <args_json><![CDATA[{{"keywords": ["Python Document", "how to use python"]}}]]></args_json>
    </function_call>
</function_calls>

Correct example with args_kv:
{trigger_signal}
<function_calls>
    <function_call>
        <tool>Edit</tool>
        <args_json><![CDATA[{{"filePath": "/tmp/AppShell.vue"}}]]></args_json>
        <args_kv>
            <arg name="oldText"><![CDATA[const navItems = [
  {{ to: "/dashboard", label: "Dashboard" }}
];]]></arg>
            <arg name="newText"><![CDATA[const navItems = [
  {{ to: "/dashboard", label: "仪表盘" }}
];]]></arg>
        </args_kv>
    </function_call>
</function_calls>

Incorrect example:
I will call a tool now.
{trigger_signal}
<function_calls>
    <function_call>
        <tool>Grep</tool>
        <args_json><![CDATA[{{"-i": true, "path": "."}}]]></args_json>
    </function_call>
</function_calls>

Incorrect example:
{trigger_signal}
<function_calls>
    <function_call>
        <tool>Edit</tool>
        <args_json><![CDATA[{{"filePath": "/tmp/AppShell.vue", "oldText": "const navItems = [
  {{ to: "/dashboard", label: "Dashboard" }}
];"}}]]></args_json>
    </function_call>
</function_calls>
Done.
"""

    logger.debug(
        "生成 XML 工具提示词, 包含 {} 个工具定义, 触发信号: {}...",
        len(tools_list_str),
        trigger_signal[:20],
    )
    return prompt


def process_tool_choice(
    tool_choice: Any,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """处理 tool_choice 字段，返回需要追加到提示词的额外指令。"""
    del tools

    if tool_choice is None:
        return ""

    if isinstance(tool_choice, str):
        if tool_choice == "none":
            return (
                "\n\n**IMPORTANT:** You are prohibited from using any tools in this "
                "round. Please respond like a normal chat assistant and answer the "
                "user's question directly."
            )
        if tool_choice == "auto":
            return ""
        if tool_choice == "required":
            return (
                "\n\n**IMPORTANT:** You MUST call at least one tool in this response. "
                "Do not respond without using tools."
            )
        logger.warning("⚠️ 未知的 tool_choice 值: {}", tool_choice)
        return ""

    if isinstance(tool_choice, dict):
        function_dict = tool_choice.get("function", {})
        required_tool_name = (
            function_dict.get("name") if isinstance(function_dict, dict) else None
        )
    elif hasattr(tool_choice, "function"):
        function_dict = tool_choice.function
        required_tool_name = (
            function_dict.get("name") if isinstance(function_dict, dict) else None
        )
    else:
        logger.warning("⚠️ 不支持的 tool_choice 类型: {}", type(tool_choice))
        return ""

    if required_tool_name and isinstance(required_tool_name, str):
        return (
            "\n\n**IMPORTANT:** In this round, you must use ONLY the tool named "
            f"`{required_tool_name}`. Generate the necessary parameters and output "
            "a pure tool turn in the specified XML format, with no natural-language text."
        )

    return ""


def process_messages_with_tools(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Any = "auto",
    trigger_signal: str = "",
) -> List[Dict[str, Any]]:
    """将 XML 工具定义注入到消息列表中。"""
    tc_mode = tool_choice if isinstance(tool_choice, str) else "auto"
    if not tools or tc_mode == "none":
        return messages

    tools_prompt = generate_tool_prompt(tools, trigger_signal)
    if not tools_prompt:
        return messages

    choice_prompt = process_tool_choice(tool_choice, tools)
    if choice_prompt:
        tools_prompt += choice_prompt

    processed = []
    has_system = any(m.get("role") == "system" for m in messages)

    if has_system:
        system_injected = False
        for msg in messages:
            if msg.get("role") == "system" and not system_injected:
                new_msg = msg.copy()
                content = new_msg.get("content", "")
                if isinstance(content, list):
                    content_str = " ".join(
                        [
                            item.get("text", "")
                            if isinstance(item, dict) and item.get("type") == "text"
                            else ""
                            for item in content
                        ]
                    )
                else:
                    content_str = str(content)
                new_msg["content"] = content_str.rstrip() + "\n\n" + tools_prompt.strip()
                processed.append(new_msg)
                system_injected = True
            else:
                processed.append(msg)
    else:
        processed.append(
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to tools."
                    f"\n\n{tools_prompt.strip()}"
                ),
            }
        )
        processed.extend(messages)

    logger.debug("XML 工具提示已注入到消息列表, 共 {} 条消息", len(processed))
    return processed
