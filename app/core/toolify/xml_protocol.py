#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具调用处理模块（Toolify XML 方案集成）

核心机制：
1. 随机 XML 触发信号 — 防止 LLM 误触发
2. <function_calls> XML 格式 — 对 LLM 更友好的工具调用格式
3. 流式实时检测（StreamingFunctionCallDetector）— 支持 <think> 块过滤
4. Schema 验证 — 校验工具调用参数合法性
5. 完整性守卫 — 防止流式 XML 片段导致过早解析

基于 Toolify (https://github.com/funnycups/Toolify) 核心逻辑移植。
"""

import json
import html
import re
import secrets
import string
import textwrap
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.utils.logger import get_logger

logger = get_logger()

_FALLBACK_TRIGGER_RE = re.compile(r"<Function_[A-Za-z0-9]{4}_Start/>")
_BARE_FUNCTION_CALLS_OPEN_RE = re.compile(
    r"<\s*function[\s_-]*calls\s*>",
    re.IGNORECASE,
)
_FUNCTION_CALLS_BLOCK_RE = re.compile(
    r"<function_calls>([\s\S]*?)</function_calls>"
)

THINKING_XML_TAGS = ("details", "think", "reasoning", "thought")

_XMLFC_CONTAINER_TAGS = frozenset(
    {"function_calls", "function_call", "args_kv", "args"}
)
_XMLFC_LEAF_TAGS = frozenset({"tool", "args_json", "arg"})
_XMLFC_ALL_TAGS = _XMLFC_CONTAINER_TAGS | _XMLFC_LEAF_TAGS
_XMLFC_CLOSE_TAG_TEXT = {tag: f"</{tag}>" for tag in _XMLFC_ALL_TAGS}

_THINKING_TAG_TOKEN_RE = re.compile(
    rf"<\s*(?P<close>/)?\s*(?P<tag>{'|'.join(THINKING_XML_TAGS)})\b[^>]*?>",
    re.IGNORECASE,
)
_XMLFC_TAG_TOKEN_RE = re.compile(
    r"<\s*(?P<close>/)?\s*(?P<tag>function_calls|function_call|tool|args_json|args_kv|args|arg)\b[^>]*?>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _KnownTagToken:
    name: str
    is_close: bool
    is_self_closing: bool
    start: int
    end: int
    raw: str


@dataclass
class _XMLFCStructureScan:
    is_complete: bool
    is_repairable: bool
    fatal: bool
    open_stack: List[str] = field(default_factory=list)
    auto_closed_tags: List[str] = field(default_factory=list)
    dropped_closing_tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 触发信号生成
# ---------------------------------------------------------------------------

def generate_trigger_signal() -> str:
    """生成随机的自闭合 XML 触发信号，如 <Function_AB1c_Start/>

    随机字符串确保 LLM 不会在普通文本中意外输出该信号。
    """
    chars = string.ascii_letters + string.digits
    random_str = ''.join(secrets.choice(chars) for _ in range(4))
    return f"<Function_{random_str}_Start/>"


# ---------------------------------------------------------------------------
# 工具提示词生成（Toolify XML 风格）
# ---------------------------------------------------------------------------

def generate_tool_prompt(
    tools: Optional[List[Dict[str, Any]]],
    trigger_signal: str,
) -> str:
    """
    生成 XML 格式的工具调用提示词

    将 OpenAI tools 定义转换为结构化的工具说明文档，包含详细参数规格、
    触发信号和 XML 输出格式要求。

    Args:
        tools: OpenAI 格式的工具定义列表
        trigger_signal: XML 触发信号字符串

    Returns:
        str: 完整的工具使用系统提示词
    """
    if not tools or len(tools) == 0:
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

        # 参数摘要
        params_summary = ", ".join([
            f"{p_name} ({(p_info or {}).get('type', 'any')})"
            for p_name, p_info in props.items()
        ]) or "None"

        # 参数详情
        detail_lines: List[str] = []
        for p_name, p_info in props.items():
            p_info = p_info or {}
            p_type = p_info.get("type", "any")
            is_required = "Yes" if p_name in required_list else "No"
            p_desc = p_info.get("description")
            enum_vals = p_info.get("enum")
            default_val = p_info.get("default")

            detail_lines.append(f"- {p_name}:")
            detail_lines.append(f"  - type: {p_type}")
            detail_lines.append(f"  - required: {is_required}")
            if p_desc:
                detail_lines.append(f"  - description: {p_desc}")
            if enum_vals is not None:
                try:
                    detail_lines.append(f"  - enum: {json.dumps(enum_vals, ensure_ascii=False)}")
                except Exception:
                    detail_lines.append(f"  - enum: {enum_vals}")
            if default_val is not None:
                try:
                    detail_lines.append(f"  - default: {json.dumps(default_val, ensure_ascii=False)}")
                except Exception:
                    detail_lines.append(f"  - default: {default_val}")

            # 数组元素类型
            if p_type == "array":
                items = p_info.get("items") or {}
                if isinstance(items, dict):
                    itype = items.get("type")
                    if itype:
                        detail_lines.append(f"  - items.type: {itype}")

        detail_block = "\n".join(detail_lines) if detail_lines else "(no parameter details)"
        desc_block = f"```\n{description}\n```" if description else "None"

        tools_list_str.append(
            f"{i + 1}. <tool name=\"{name}\">\n"
            f"   Description:\n{desc_block}\n"
            f"   Parameters summary: {params_summary}\n"
            f"   Required parameters: {', '.join(required_list) if required_list else 'None'}\n"
            f"   Parameter details:\n{detail_block}"
        )

    if not tools_list_str:
        return ""

    tools_block = "\n\n".join(tools_list_str)

    prompt = f"""
You have access to the following available tools to help solve problems:

{tools_block}

**IMPORTANT CONTEXT NOTES:**
1. You can call MULTIPLE tools in a single response if needed.
2. Even though you can call multiple tools, you MUST respect the user's later constraints and preferences (e.g., the user may request no tools, only one tool, or a specific tool/workflow).
3. The conversation context may already contain tool execution results from previous function calls. Review the conversation history carefully to avoid unnecessary duplicate tool calls.
4. Tool execution results from previous calls appear as <tool_response tool="...">...</tool_response>.
   These are internal metadata — absorb the information and continue your task.
   NEVER repeat, quote, or reference the raw tool response content in your output.
5. This is the ONLY format you can use for tool calls, and any deviation will result in failure.

When you need to use tools, you **MUST** strictly follow this format. Do NOT include any extra text, explanations, or dialogue on the first and second lines of the tool call syntax:

1. When starting tool calls, begin on a new line with exactly:
{trigger_signal}
No leading or trailing spaces, output exactly as shown above. The trigger signal MUST be on its own line and appear only once. Do not output a trigger signal for each tool call.

2. Starting from the second line, **immediately** follow with the complete <function_calls> XML block.

3. For multiple tool calls, include multiple <function_call> blocks within the same <function_calls> wrapper, not separate blocks. Output the trigger signal only once, then one <function_calls> with all <function_call> children.

4. Do not add any text or explanation after the closing </function_calls> tag.

STRICT ARGUMENT KEY RULES:
- You MUST use parameter keys EXACTLY as defined (case- and punctuation-sensitive). Do NOT rename, add, or remove characters.
- If a key starts with a hyphen (e.g., "-i", "-C"), you MUST keep the leading hyphen in the JSON key. Never convert "-i" to "i" or "-C" to "C".
- The <tool> tag must contain the exact name of a tool from the list. Any other tool name is invalid.
- Use <args_json> when all arguments are short and easy to encode as JSON.
- If any string argument contains multi-line text, source code, quotes, XML-like text, or other content that is hard to escape safely in JSON, you MUST use <args_kv> for those string arguments instead of putting them into <args_json>.
- In <args_kv>, each argument MUST be encoded as <arg name="EXACT_KEY">VALUE</arg>.
- You MAY combine both tags in the same <function_call>: keep simple/scalar values in <args_json> and put complex multi-line string values in <args_kv>.
- For <args_kv>, use <![CDATA[...]]> inside <arg> whenever the value contains code, quotes, or multiple lines.

CORRECT Example (multiple tool calls):
...response content (optional)...
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

CORRECT Example (large code string via args_kv):
...response content (optional)...
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

INCORRECT Example (extra text + wrong key names — DO NOT DO THIS):
...response content (optional)...
{trigger_signal}
I will call the tools for you.
<function_calls>
    <function_call>
        <tool>Grep</tool>
        <args>
            <i>true</i>
            <C>2</C>
            <path>.</path>
        </args>
    </function_call>
</function_calls>

INCORRECT Example (do not put raw multi-line code directly into args_json):
<function_calls>
    <function_call>
        <tool>Edit</tool>
        <args_json><![CDATA[{{"filePath": "/tmp/AppShell.vue", "oldText": "const navItems = [
  {{ to: "/dashboard", label: "Dashboard" }}
];"}}]]></args_json>
    </function_call>
</function_calls>

Now please be ready to strictly follow the above specifications.
"""

    logger.debug("生成 XML 工具提示词, 包含 {} 个工具定义, 触发信号: {}...", len(tools_list_str), trigger_signal[:20])
    return prompt


def process_tool_choice(
    tool_choice: Any,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """处理 tool_choice 字段，返回需要追加到提示词的额外指令。

    Args:
        tool_choice: 工具选择策略（str 或 dict/object）
        tools: 可用工具列表

    Returns:
        str: 附加提示文本（可能为空）
    """
    if tool_choice is None:
        return ""

    if isinstance(tool_choice, str):
        if tool_choice == "none":
            return "\n\n**IMPORTANT:** You are prohibited from using any tools in this round. Please respond like a normal chat assistant and answer the user's question directly."
        elif tool_choice == "auto":
            return ""
        elif tool_choice == "required":
            return "\n\n**IMPORTANT:** You MUST call at least one tool in this response. Do not respond without using tools."
        else:
            logger.warning("⚠️ 未知的 tool_choice 值: {}", tool_choice)
            return ""

    # 处理 ToolChoice object: {"type": "function", "function": {"name": "xxx"}}
    if isinstance(tool_choice, dict):
        function_dict = tool_choice.get("function", {})
        required_tool_name = function_dict.get("name") if isinstance(function_dict, dict) else None
    elif hasattr(tool_choice, 'function'):
        function_dict = tool_choice.function
        if isinstance(function_dict, dict):
            required_tool_name = function_dict.get("name")
        else:
            required_tool_name = None
    else:
        logger.warning("⚠️ 不支持的 tool_choice 类型: {}", type(tool_choice))
        return ""

    if required_tool_name and isinstance(required_tool_name, str):
        return f"\n\n**IMPORTANT:** In this round, you must use ONLY the tool named `{required_tool_name}`. Generate the necessary parameters and output in the specified XML format."

    return ""


def process_messages_with_tools(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Any = "auto",
    trigger_signal: str = "",
) -> List[Dict[str, Any]]:
    """
    将 XML 工具定义注入到消息列表中

    Args:
        messages: 原始消息列表
        tools: 工具定义列表
        tool_choice: 工具选择策略
        trigger_signal: XML 触发信号

    Returns:
        List[Dict]: 处理后的消息列表
    """
    tc_mode = tool_choice if isinstance(tool_choice, str) else "auto"
    if not tools or tc_mode == "none":
        return messages

    tools_prompt = generate_tool_prompt(tools, trigger_signal)
    if not tools_prompt:
        return messages

    # 处理 tool_choice 附加指令
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
                    content_str = " ".join([
                        item.get("text", "") if isinstance(item, dict) and item.get("type") == "text" else ""
                        for item in content
                    ])
                else:
                    content_str = str(content)
                new_msg["content"] = content_str + tools_prompt
                processed.append(new_msg)
                system_injected = True
            else:
                processed.append(msg)
    else:
        processed.append({
            "role": "system",
            "content": f"You are a helpful assistant with access to tools.{tools_prompt}"
        })
        processed.extend(messages)

    logger.debug("XML 工具提示已注入到消息列表, 共 {} 条消息", len(processed))
    return processed


# ---------------------------------------------------------------------------
# <think> 块处理
# ---------------------------------------------------------------------------


def _match_known_tag_token(
    text: str,
    pos: int,
    pattern: re.Pattern[str],
) -> Optional[_KnownTagToken]:
    """在指定位置匹配已知标签，忽略 CDATA 片段。"""
    if not text or pos < 0 or pos >= len(text) or text[pos] != "<":
        return None
    if text.startswith("<![CDATA[", pos):
        return None

    match = pattern.match(text, pos)
    if not match:
        return None

    raw = match.group(0)
    return _KnownTagToken(
        name=str(match.group("tag") or "").lower().replace("-", "_"),
        is_close=bool(match.group("close")),
        is_self_closing=(not bool(match.group("close")) and raw.rstrip().endswith("/>")),
        start=pos,
        end=match.end(),
        raw=raw,
    )


def _iter_known_tag_tokens(
    text: str,
    pattern: re.Pattern[str],
):
    """迭代文本中的已知标签，自动跳过 CDATA 内容。"""
    if not text:
        return

    pos = 0
    text_len = len(text)
    while pos < text_len:
        lt_pos = text.find("<", pos)
        if lt_pos == -1:
            break

        if text.startswith("<![CDATA[", lt_pos):
            cdata_end = text.find("]]>", lt_pos + len("<![CDATA["))
            if cdata_end == -1:
                break
            pos = cdata_end + len("]]>")
            continue

        token = _match_known_tag_token(text, lt_pos, pattern)
        if token is not None:
            yield token
            pos = token.end
            continue

        pos = lt_pos + 1


def remove_think_blocks(text: str) -> str:
    """移除所有思维包装块，支持 details/think/reasoning/thought。"""
    if not text:
        return ""

    kept_parts: List[str] = []
    keep_start = 0
    think_stack: List[str] = []

    for token in _iter_known_tag_tokens(text, _THINKING_TAG_TOKEN_RE):
        if token.is_close:
            if think_stack:
                while think_stack and think_stack[-1] != token.name:
                    think_stack.pop()
                if think_stack and think_stack[-1] == token.name:
                    think_stack.pop()
                if not think_stack:
                    keep_start = token.end
            else:
                kept_parts.append(text[keep_start:token.start])
                keep_start = token.end
            continue

        if not think_stack:
            kept_parts.append(text[keep_start:token.start])
        think_stack.append(token.name)

    if not think_stack:
        kept_parts.append(text[keep_start:])

    return "".join(kept_parts)


def find_last_trigger_signal_outside_think(text: str, trigger_signal: str) -> int:
    """查找不在思维包装块内的最后一个触发信号位置。"""
    if not text or not trigger_signal:
        return -1

    i = 0
    last_pos = -1
    think_stack: List[str] = []

    while i < len(text):
        if text.startswith("<![CDATA[", i):
            cdata_end = text.find("]]>", i + len("<![CDATA["))
            if cdata_end == -1:
                break
            i = cdata_end + len("]]>")
            continue

        token = _match_known_tag_token(text, i, _THINKING_TAG_TOKEN_RE)
        if token is not None:
            if token.is_close:
                while think_stack and think_stack[-1] != token.name:
                    think_stack.pop()
                if think_stack and think_stack[-1] == token.name:
                    think_stack.pop()
            else:
                think_stack.append(token.name)
            i = token.end
            continue

        if not think_stack and text.startswith(trigger_signal, i):
            last_pos = i
            i += len(trigger_signal)
            continue

        i += 1

    return last_pos


# ---------------------------------------------------------------------------
# XML / JSON 通配修复 (Wildcard Repair Pipeline)
# ---------------------------------------------------------------------------

# 已知 XML 标签名 → (开标签正则, 闭标签正则, 规范开标签, 规范闭标签)
_KNOWN_TAG_REPAIRS: List[tuple] = [
    # function_calls  (兼容: functioncalls, function-calls, FUNCTION_CALLS 等)
    (
        re.compile(r'<\s*function[\s_-]*calls\s*>', re.IGNORECASE),
        re.compile(r'</\s*function[\s_-]*calls\s*>', re.IGNORECASE),
        '<function_calls>', '</function_calls>',
    ),
    # function_call  (单数, 兼容同上变体)
    (
        re.compile(r'<\s*function[\s_-]*call\s*>(?!\s*s)', re.IGNORECASE),
        re.compile(r'</\s*function[\s_-]*call\s*>(?!\s*s)', re.IGNORECASE),
        '<function_call>', '</function_call>',
    ),
    # tool
    (
        re.compile(r'<\s*tool\s*>', re.IGNORECASE),
        re.compile(r'</\s*tool\s*>', re.IGNORECASE),
        '<tool>', '</tool>',
    ),
    # args_json  (兼容: argsjson, args-json, args_Json 等)
    (
        re.compile(r'<\s*args[\s_-]*json\s*>', re.IGNORECASE),
        re.compile(r'</\s*args[\s_-]*json\s*>', re.IGNORECASE),
        '<args_json>', '</args_json>',
    ),
    # args_kv  (兼容: argskv, args-kv, ARGS_KV 等)
    (
        re.compile(r'<\s*args[\s_-]*kv\s*>', re.IGNORECASE),
        re.compile(r'</\s*args[\s_-]*kv\s*>', re.IGNORECASE),
        '<args_kv>', '</args_kv>',
    ),
    # args  (仅在单独出现时)
    (
        re.compile(r'<\s*args\s*>', re.IGNORECASE),
        re.compile(r'</\s*args\s*>', re.IGNORECASE),
        '<args>', '</args>',
    ),
]

# CDATA 开头泛化匹配:
# 匹配 <! 后面跟若干可选空格/字母(包含CDATA)最终跟 [ 的情况
# 能匹配: <![CDATA[, <![CDATACDATA[, <![ CDATA [, <![CDATA CDATA[, <![CData[
_RE_CDATA_OPEN_FUZZY = re.compile(
    r'<!\s*\[?\s*(?:CDATA\s*)+\[',
    re.IGNORECASE,
)

# CDATA 终止符泛化匹配: ]] > 允许中间空格
_RE_CDATA_CLOSE_FUZZY = re.compile(r'\]\s*\]\s*>')


def normalize_cdata_markers(raw: str) -> str:
    """修复畸形 CDATA 标记。

    泛化匹配: 只要检测到包含 CDATA 和 [ 的开头标记,
    就归一化为标准的 <![CDATA[ 和 ]]>。

    也处理模型输出 ]> 代替 ]]> 的情况（缺少一个 ]）。
    """
    if raw is None:
        return ""

    original = raw

    # 修复开头标记
    raw = _RE_CDATA_OPEN_FUZZY.sub('<![CDATA[', raw)

    # 修复终止符 (仅当存在 CDATA 上下文时)
    if '<![CDATA[' in raw:
        raw = _RE_CDATA_CLOSE_FUZZY.sub(']]>', raw)

        # ★ 修复单括号畸形终止符: ]> → ]]>
        # 模型有时只输出一个 ] 就跟 >,  仅在 CDATA 仍未平衡时修复
        open_count = raw.count('<![CDATA[')
        close_count = raw.count(']]>')
        if open_count > close_count:
            # (?<!\]) 确保不会把已修复的 ]]> 再改一次
            raw = re.sub(r'(?<!\])\]>', ']]>', raw)
            logger.debug(
                "🔧 修复单括号 CDATA 终止符: open={}, close_before={}, close_after={}",
                open_count, close_count, raw.count(']]>'),
            )

    if raw != original:
        logger.debug("🔧 CDATA 标记已修复: {} → {}", repr(original[:60]), repr(raw[:60]))

    return raw


def normalize_xml_tag_names(xml_str: str) -> str:
    """归一化已知 XML 标签名 (大小写/空格/连字符变体)。"""
    if not xml_str:
        return ""

    original = xml_str
    for open_re, close_re, open_canon, close_canon in _KNOWN_TAG_REPAIRS:
        xml_str = open_re.sub(open_canon, xml_str)
        xml_str = close_re.sub(close_canon, xml_str)

    if xml_str != original:
        logger.debug("🔧 XML 标签名已归一化")

    return xml_str


def normalize_xml_structure(xml_str: str) -> str:
    """组合 XML 结构修复: CDATA + 标签名归一化。"""
    xml_str = normalize_cdata_markers(xml_str)
    xml_str = normalize_xml_tag_names(xml_str)
    return xml_str


def scan_xmlfc_structure(
    xml_str: str,
    *,
    final: bool = False,
) -> _XMLFCStructureScan:
    """扫描 xmlfc 标签结构，识别可自动修复的缺失闭合。"""
    if not xml_str:
        return _XMLFCStructureScan(False, False, True)

    normalized = repair_unclosed_cdata(normalize_xml_structure(xml_str))
    stack: List[str] = []
    auto_closed: List[str] = []
    dropped_closing: List[str] = []
    saw_root_open = False
    saw_root_close = False

    for token in _iter_known_tag_tokens(normalized, _XMLFC_TAG_TOKEN_RE):
        if token.is_close:
            if token.name in stack:
                while stack and stack[-1] != token.name:
                    auto_closed.append(stack.pop())
                if stack and stack[-1] == token.name:
                    stack.pop()
            else:
                dropped_closing.append(token.name)
                continue

            if token.name == "function_calls":
                saw_root_close = True
            continue

        while stack and stack[-1] in _XMLFC_LEAF_TAGS:
            auto_closed.append(stack.pop())

        if not token.is_self_closing:
            stack.append(token.name)
            if token.name == "function_calls":
                saw_root_open = True

    remaining_stack = list(stack)
    if final and remaining_stack:
        auto_closed.extend(reversed(remaining_stack))
        remaining_stack = []

    is_complete = saw_root_open and saw_root_close and not remaining_stack
    is_repairable = saw_root_open and (saw_root_close or final)
    fatal = not saw_root_open

    return _XMLFCStructureScan(
        is_complete=is_complete,
        is_repairable=is_repairable,
        fatal=fatal,
        open_stack=remaining_stack,
        auto_closed_tags=auto_closed,
        dropped_closing_tags=dropped_closing,
    )


def repair_xmlfc_structure(
    xml_str: str,
    *,
    final: bool = False,
) -> str:
    """基于 xmlfc 标签栈修复缺失闭合标签，过滤孤立闭合标签。"""
    if not xml_str:
        return ""

    normalized = repair_unclosed_cdata(normalize_xml_structure(xml_str))
    repaired_parts: List[str] = []
    last_end = 0
    stack: List[str] = []

    for token in _iter_known_tag_tokens(normalized, _XMLFC_TAG_TOKEN_RE):
        repaired_parts.append(normalized[last_end:token.start])

        if token.is_close:
            if token.name in stack:
                while stack and stack[-1] != token.name:
                    missing = stack.pop()
                    repaired_parts.append(_XMLFC_CLOSE_TAG_TEXT[missing])
                if stack and stack[-1] == token.name:
                    repaired_parts.append(token.raw)
                    stack.pop()
            else:
                logger.debug("🔧 丢弃孤立闭合标签 </{}>", token.name)

            last_end = token.end
            continue

        while stack and stack[-1] in _XMLFC_LEAF_TAGS:
            missing = stack.pop()
            repaired_parts.append(_XMLFC_CLOSE_TAG_TEXT[missing])

        repaired_parts.append(token.raw)
        if not token.is_self_closing:
            stack.append(token.name)

        last_end = token.end

    repaired_parts.append(normalized[last_end:])

    if final and stack:
        while stack:
            repaired_parts.append(_XMLFC_CLOSE_TAG_TEXT[stack.pop()])

    repaired = "".join(repaired_parts)
    if repaired != normalized:
        logger.debug("🔧 XMLFC 结构已修复 ({} → {} 字符)", len(normalized), len(repaired))

    return repaired


def repair_json_payload(s: str) -> str:
    """修复常见 JSON 畸形: 尾随逗号、Python 布尔值/None。"""
    if not s:
        return s

    original = s

    # Python 布尔值/None → JSON (仅替换不在引号内的)
    # 简单策略: 用词边界替换, 再验证结果是否仍可解析
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)

    # 移除对象/数组最后一个元素后的尾随逗号
    s = re.sub(r',\s*([}\]])', r'\1', s)

    if s != original:
        logger.debug("🔧 JSON payload 已修复: {} → {}", repr(original[:60]), repr(s[:60]))

    return s


def _is_xml_noise(s: str) -> bool:
    """判断字符串是否全为可忽略的 XML 标记噪声 (CDATA 标记残留、空白等)。"""
    if not s:
        return True
    stripped = s.strip()
    if not stripped:
        return True
    # 允许由 CDATA 开闭标记碎片(含畸形)、空白、方括号等组成
    return bool(re.fullmatch(
        r'(?:'
        r'<!\s*\[?\s*(?:CDATA\s*)*\[?'  # 畸形 CDATA 开头碎片
        r'|\]\s*\]?\s*>?'                # CDATA 终止符碎片 (含单括号 ]>)
        r'|\s'                           # 空白
        r')+',
        stripped,
        re.IGNORECASE | re.DOTALL,
    ))


# ---------------------------------------------------------------------------
# XML 解析
# ---------------------------------------------------------------------------


def _repair_unescaped_quotes(s: str) -> Optional[Dict[str, Any]]:
    """迭代修复 JSON 字符串值内的未转义双引号。

    利用 json.JSONDecodeError.pos 定位解析失败点，回溯找到导致字符串
    提前关闭的未转义引号并插入反斜杠，然后重试。最多尝试 20 轮。
    """
    current = s.strip()
    if not current:
        return None
    for _ in range(20):
        try:
            result = json.loads(current)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError as e:
            pos = e.pos
            if pos <= 0 or pos >= len(current):
                return None
            # 错误位置前应有一个未转义的引号导致字符串提前关闭
            quote_pos = current.rfind('"', 0, pos)
            if quote_pos <= 0:
                return None
            # 确认该引号不是已转义的
            num_bs = 0
            check = quote_pos - 1
            while check >= 0 and current[check] == '\\':
                num_bs += 1
                check -= 1
            if num_bs % 2 == 1:
                # 已转义，问题在别处，放弃
                return None
            current = current[:quote_pos] + '\\' + current[quote_pos:]
    return None


def _parse_args_json_payload(payload: str) -> Optional[Dict[str, Any]]:
    """鲁棒的 args_json 解析（合并 PR #8 的改进）

    - 空 payload -> {}
    - 必须解析为 dict
    - 包含 markdown fence 清理和 JSON 对象恢复
    """
    if payload is None:
        return {}
    s = payload.strip()
    if not s:
        return {}

    # 清理意外的 markdown 代码围栏
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

    def _try_parse(x: str):
        try:
            y = json.loads(x)
            return y if isinstance(y, dict) else None
        except Exception:
            return None

    # ★ JSON 层修复: 尾随逗号、Python 布尔值
    s = repair_json_payload(s)

    parsed = _try_parse(s)
    if parsed is not None:
        return parsed

    # 恢复策略 1: 提取平衡的 JSON 对象
    start = s.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:i+1]
                        prefix = s[:start].strip()
                        suffix = s[i+1:].strip()
                        if prefix or suffix:
                            # ★ 放宽: 如果外围内容是 XML 标记噪声则接受
                            if _is_xml_noise(prefix) and _is_xml_noise(suffix):
                                logger.debug("🔧 args_json 外围为 XML 噪声, 予以忽略")
                            else:
                                logger.debug("🔧 args_json 恢复被拒绝: JSON 对象外有额外内容")
                                break
                        parsed = _try_parse(candidate)
                        if parsed is not None:
                            logger.debug("🔧 args_json 通过平衡对象提取恢复成功")
                            return parsed
                        break

    # 恢复策略 2: 迭代修复字符串值内的未转义双引号
    repaired = _repair_unescaped_quotes(s)
    if repaired is not None:
        logger.debug("🔧 args_json 通过未转义引号修复恢复成功")
        return repaired

    logger.debug("🔧 args_json 在所有恢复尝试后仍无效")
    return None


def _extract_cdata_text(raw: str) -> str:
    """提取 CDATA 文本（含畸形 CDATA 修复及流式截断恢复）。

    处理三种情形:
    1. 标准 CDATA: <![CDATA[...]]>  → 正常提取
    2. CDATA 开头存在但 ]]> 不紧跟 → rfind 兜底
    3. 流式截断: <![CDATA[... 末尾无 ]]> → 提取所有后续内容
    """
    if raw is None:
        return ""

    # ★ 先修复畸形 CDATA 标记
    raw = normalize_cdata_markers(raw)

    if "<![CDATA[" not in raw:
        return raw

    # Case 1: 完整 CDATA 段(可多段)
    parts = re.findall(r"<!\[CDATA\[(.*?)\]\]>", raw, flags=re.DOTALL)
    if parts:
        return "".join(parts)

    # Case 2 & 3: CDATA 开头存在但无对应 ]]>
    st = raw.find("<![CDATA[")
    if st != -1:
        content_start = st + len("<![CDATA[")
        ed = raw.rfind("]]>")
        if ed > content_start:
            # Case 2: ]]> 在别处
            return raw[content_start:ed]
        # Case 3: 流式截断 — 取 <![CDATA[ 之后的全部内容
        # (流式边界可能在 ]]> 到达之前就结束了)
        tail = raw[content_start:]
        # 移除可能出现在末尾的不完整终止符碎片
        # 覆盖: ]]>, ]>, ]], ] (从长到短匹配)
        tail = re.sub(r'\]\]?>?$', '', tail)
        logger.debug("🔧 CDATA 流式截断恢复: 提取 {} 字符", len(tail))
        return tail
    return raw


def _parse_args_kv_text(raw: Optional[str], *, from_xml: bool = False) -> str:
    """解析 args_kv 中单个 arg 的文本值。

    - CDATA 内容保持原样
    - 普通文本做轻量 strip
    - 多行普通文本做 dedent 后 strip，避免 XML 排版缩进污染值
    """
    if raw is None:
        return ""

    if not from_xml and "<![CDATA[" in raw:
        return _extract_cdata_text(raw)

    text = str(raw)
    if "\n" in text or "\r" in text:
        return textwrap.dedent(text).strip("\r\n")
    return text.strip()


def _element_inner_xml(el: ET.Element) -> str:
    """提取元素内部原始文本，保留子节点 XML。"""
    parts = [el.text or ""]
    for child in list(el):
        parts.append(ET.tostring(child, encoding="unicode"))
    return "".join(parts)


def _parse_args_kv_elements(args_kv_el: ET.Element) -> Dict[str, Any]:
    """从 ET 节点解析 <args_kv><arg name=\"...\">...</arg></args_kv>。"""
    args: Dict[str, Any] = {}
    for child in list(args_kv_el):
        tag_name = str(getattr(child, "tag", "") or "").lower().replace("-", "_")
        if tag_name != "arg":
            continue
        key = str(child.attrib.get("name") or child.attrib.get("key") or "").strip()
        if not key:
            continue
        args[key] = _parse_args_kv_text(_element_inner_xml(child), from_xml=True)
    return args


_ARG_BLOCK_RE = re.compile(
    r"<\s*arg\b([^>]*)>([\s\S]*?)</\s*arg\s*>",
    re.IGNORECASE,
)
_ARG_NAME_ATTR_RE = re.compile(
    r"""\b(?:name|key)\s*=\s*(?:"([^"]*)"|'([^']*)')""",
    re.IGNORECASE,
)


def _parse_args_kv_block(raw_inner: str) -> Dict[str, Any]:
    """从正则 fallback 路径解析 args_kv 块。"""
    args: Dict[str, Any] = {}
    for match in _ARG_BLOCK_RE.finditer(raw_inner or ""):
        attrs = match.group(1) or ""
        payload = match.group(2)
        name_match = _ARG_NAME_ATTR_RE.search(attrs)
        if not name_match:
            continue
        key = str(name_match.group(1) or name_match.group(2) or "").strip()
        if not key:
            continue
        args[key] = _parse_args_kv_text(payload, from_xml=False)
    return args


def repair_unclosed_cdata(xml_str: str) -> str:
    """为 XML 字符串中所有未闭合的 CDATA 段补充 ]]> 终止符。

    ET 解析器遇到未闭合的 CDATA 会抛出 ParseError: unclosed CDATA section。
    此函数在交给 ET 之前补全缺失的终止符，使 ET 能正常解析。
    会尝试在结构上合理的位置插入（最近的 XML 闭合标签前），
    而非简单追加到末尾。
    """
    if not xml_str or "<![CDATA[" not in xml_str:
        return xml_str

    open_count = xml_str.count("<![CDATA[")
    close_count = xml_str.count("]]>")
    missing = open_count - close_count
    if missing <= 0:
        return xml_str

    logger.debug("🔧 检测到 {} 个未闭合 CDATA, 自动补全终止符", missing)

    # 智能插入: 在最后一个未闭合 <![CDATA[ 之后，
    # 找第一个 </args_json> 或 </function_call> 或任意 </ 标签前插入
    result = xml_str
    for _ in range(missing):
        last_open = result.rfind("<![CDATA[")
        search_start = last_open + len("<![CDATA[")
        # 尝试在最近的 XML 闭合标签前插入
        close_tag = re.search(r'</', result[search_start:])
        if close_tag:
            insert_pos = search_start + close_tag.start()
            result = result[:insert_pos] + "]]>" + result[insert_pos:]
        else:
            result += "]]>"

    return result


def _find_last_function_calls_match(
    text: str,
    *,
    tail_only: bool = False,
) -> Optional[re.Match[str]]:
    """查找最后一个完整的 <function_calls> 块。

    Args:
        text: 已做 think 清理和标签归一化的文本。
        tail_only: 为 True 时要求 </function_calls> 后仅剩空白。
    """
    last_match: Optional[re.Match[str]] = None
    for match in _FUNCTION_CALLS_BLOCK_RE.finditer(text or ""):
        if tail_only and text[match.end():].strip():
            continue
        last_match = match
    return last_match


def locate_function_calls_block(
    xml_string: str,
    trigger_signal: str = "",
    *,
    allow_bare: bool = False,
    bare_tail_only: bool = False,
) -> Optional[Tuple[str, str, str]]:
    """定位当前回复中的工具调用 XML 块。

    Returns:
        (calls_xml, calls_content, source)
        source in {"trigger", "bare"}
    """
    if not xml_string:
        return None

    cleaned_content = remove_think_blocks(xml_string)

    # ★ 先对内容做 XML 通配修复 (标签名/CDATA), 让后续搜索能匹配到畸形标签
    original_cleaned = cleaned_content
    cleaned_content = normalize_xml_structure(cleaned_content)
    if cleaned_content != original_cleaned:
        logger.debug(
            "🔧 XML 结构已修复 ({} → {} 字符)",
            len(original_cleaned),
            len(cleaned_content),
        )

    if trigger_signal and trigger_signal in cleaned_content:
        signal_positions = []
        start_pos = 0
        while True:
            pos = cleaned_content.find(trigger_signal, start_pos)
            if pos == -1:
                break
            signal_positions.append(pos)
            start_pos = pos + 1

        for idx in range(len(signal_positions) - 1, -1, -1):
            pos = signal_positions[idx]
            sub = cleaned_content[pos:]
            match = _FUNCTION_CALLS_BLOCK_RE.search(sub)
            if match:
                logger.debug("🔧 使用触发信号 index {}, pos {}", idx, pos)
                return match.group(0), match.group(1), "trigger"

        logger.debug("🔧 触发信号后未找到 <function_calls> 标签")

    if allow_bare:
        match = _find_last_function_calls_match(
            cleaned_content,
            tail_only=bare_tail_only,
        )
        if match:
            logger.debug(
                "🔧 使用 bare XML 兜底定位 <function_calls>, tail_only={}",
                bare_tail_only,
            )
            return match.group(0), match.group(1), "bare"

        logger.debug("🔧 bare XML 兜底未找到可用的 <function_calls> 块")

    return None


def parse_function_calls_xml(
    xml_string: str,
    trigger_signal: str,
    *,
    allow_bare: bool = False,
    bare_tail_only: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    解析 XML 格式的工具调用

    1. 临时移除 <think> 块
    2. 查找最后一个触发信号
    3. 从触发信号后解析 <function_calls> XML

    Returns:
        解析出的工具调用列表 [{"name": str, "args": dict}, ...] 或 None
    """
    logger.debug(
        "🔧 XML 解析开始, 输入长度: {}, allow_bare={}, tail_only={}",
        len(xml_string) if xml_string else 0,
        allow_bare,
        bare_tail_only,
    )

    if not xml_string:
        logger.debug("🔧 输入为空")
        return None

    located = locate_function_calls_block(
        xml_string,
        trigger_signal,
        allow_bare=allow_bare,
        bare_tail_only=bare_tail_only,
    )
    if located is None:
        if trigger_signal and not allow_bare:
            logger.debug("🔧 输入为空或不包含触发信号")
        return None

    calls_xml, calls_content, source = located
    logger.debug("🔧 XML 块定位成功, source={}", source)

    structure = scan_xmlfc_structure(calls_xml, final=True)
    if structure.auto_closed_tags or structure.dropped_closing_tags:
        logger.debug(
            "🔧 XMLFC 结构扫描: auto_closed={}, dropped_closing={}",
            structure.auto_closed_tags,
            structure.dropped_closing_tags,
        )
    repaired_calls_xml = repair_xmlfc_structure(calls_xml, final=True)
    repaired_match = _FUNCTION_CALLS_BLOCK_RE.search(repaired_calls_xml)
    if repaired_match:
        calls_content = repaired_match.group(1)
    else:
        calls_content = calls_content

    def _coerce_value(v: str):
        try:
            return json.loads(v)
        except Exception:
            return v

    results: List[Dict[str, Any]] = []

    # 主路径: 严格 XML 解析
    # ★ 在交给 ET 之前补全未闭合的 CDATA 终止符,
    #   防止流式截断造成 "unclosed CDATA section" ParseError
    try:
        root = ET.fromstring(repair_unclosed_cdata(repaired_calls_xml))
        for i, fc in enumerate(root.findall("function_call")):
            tool_el = fc.find("tool")
            name = (tool_el.text or "").strip() if tool_el is not None else ""
            if not name:
                continue

            args: Dict[str, Any] = {}
            has_structured_args = False

            args_json_el = fc.find("args_json")
            if args_json_el is not None:
                raw_text = args_json_el.text or ""
                payload = _extract_cdata_text(raw_text)
                parsed_args = _parse_args_json_payload(payload)
                if parsed_args is None:
                    logger.debug("🔧 function_call #{} 的 args_json 无效; 视为解析失败", i + 1)
                    return None
                args = parsed_args
                has_structured_args = True

            args_kv_el = fc.find("args_kv")
            if args_kv_el is not None:
                args.update(_parse_args_kv_elements(args_kv_el))
                has_structured_args = True

            if not has_structured_args:
                # Legacy fallback: <args><k>json</k></args>
                args_el = fc.find("args")
                if args_el is not None:
                    for child in list(args_el):
                        args[child.tag] = _coerce_value(child.text or "")

            results.append({"name": name, "args": args})

        if results:
            logger.debug("🔧 XML 解析结果 (ET): {} 个工具调用", len(results))
            return results
    except Exception as e:
        logger.debug("🔧 XML 库解析失败, 降级到正则: {}: {}", type(e).__name__, e)

    # 降级路径: 正则解析
    results = []
    call_blocks = re.findall(r"<function_call>([\s\S]*?)</function_call>", calls_content)

    for i, block in enumerate(call_blocks):
        tool_match = re.search(r"<tool>(.*?)</tool>", block)
        if not tool_match:
            continue

        name = tool_match.group(1).strip()
        args: Dict[str, Any] = {}
        has_structured_args = False

        # 容错的 args_json 提取（PR #8 改进）
        args_json_open = block.find("<args_json>")
        args_json_close = block.find("</args_json>")
        if args_json_open != -1 and args_json_close != -1 and args_json_close > args_json_open:
            raw_payload = block[args_json_open + len("<args_json>"):args_json_close]
            payload = _extract_cdata_text(raw_payload)
            parsed_args = _parse_args_json_payload(payload)
            if parsed_args is None:
                logger.debug("🔧 function_call #{} (正则) args_json 无效; 视为解析失败", i + 1)
                return None
            args = parsed_args
            has_structured_args = True
        elif args_json_open != -1 and args_json_close == -1:
            # ★ 流式截断: <args_json> 存在但 </args_json> 缺失
            # 取 <args_json> 之后的全部内容尝试解析
            raw_tail = block[args_json_open + len("<args_json>"):]
            payload = _extract_cdata_text(raw_tail)
            parsed_args = _parse_args_json_payload(payload)
            if parsed_args is not None:
                logger.debug(
                    f"🔧 function_call #{i+1} (正则) args_json 流式截断恢复成功: {list(parsed_args.keys())}"
                )
                args = parsed_args
                has_structured_args = True
            else:
                logger.debug(
                    f"🔧 function_call #{i+1} (正则) args_json 流式截断恢复失败, 使用空 args"
                )

        args_kv_match = re.search(r"<args_kv>([\s\S]*?)</args_kv>", block)
        if args_kv_match:
            args.update(_parse_args_kv_block(args_kv_match.group(1)))
            has_structured_args = True

        if not has_structured_args:
            args_block_match = re.search(r"<args>([\s\S]*?)</args>", block)
            if args_block_match:
                args_inner = args_block_match.group(1)
                arg_matches = re.findall(r"<([^\s>/]+)>([\s\S]*?)</\1>", args_inner)
                for k, v in arg_matches:
                    args[k] = _coerce_value(v)

        results.append({"name": name, "args": args})

    logger.debug("🔧 正则解析结果: {} 个工具调用", len(results))
    return results if results else None


# ---------------------------------------------------------------------------
# Schema 验证
# ---------------------------------------------------------------------------

def _schema_type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int) and not isinstance(v, bool):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _validate_value_against_schema(
    value: Any,
    schema: Dict[str, Any],
    path: str = "args",
    depth: int = 0,
) -> List[str]:
    """Best-effort JSON Schema 验证"""
    if schema is None:
        schema = {}
    if depth > 8:
        return []

    errors: List[str] = []

    stype = schema.get("type")
    if stype is None:
        if any(k in schema for k in ("properties", "required", "additionalProperties")):
            stype = "object"

    def _type_ok(t: str) -> bool:
        if t == "object":
            return isinstance(value, dict)
        if t == "array":
            return isinstance(value, list)
        if t == "string":
            return isinstance(value, str)
        if t == "boolean":
            return isinstance(value, bool)
        if t == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if t == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if t == "null":
            return value is None
        return True

    if isinstance(stype, str):
        if not _type_ok(stype):
            errors.append(f"{path}: expected type '{stype}', got '{_schema_type_name(value)}'")
            return errors
    elif isinstance(stype, list):
        if not any(_type_ok(t) for t in stype if isinstance(t, str)):
            errors.append(f"{path}: expected type in {stype!r}, got '{_schema_type_name(value)}'")
            return errors

    # object
    if isinstance(value, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            props = {}
        required = schema.get("required")
        if not isinstance(required, list):
            required = []
        required = [k for k in required if isinstance(k, str)]

        for k in required:
            if k not in value:
                errors.append(f"{path}: missing required property '{k}'")

        for k, v in value.items():
            if k in props:
                errors.extend(_validate_value_against_schema(v, props.get(k) or {}, f"{path}.{k}", depth + 1))

    # array
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, v in enumerate(value):
                errors.extend(_validate_value_against_schema(v, items, f"{path}[{i}]", depth + 1))

    return errors


def validate_parsed_tools(
    parsed_tools: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    """验证解析出的工具调用是否符合声明的工具定义。

    Args:
        parsed_tools: 解析出的工具调用列表 [{"name": str, "args": dict}]
        tools: OpenAI 格式的工具定义列表

    Returns:
        错误信息字符串，验证通过返回 None
    """
    tools = tools or []
    allowed = {}
    for t in tools:
        if not isinstance(t, dict):
            continue
        func = t.get("function", {})
        if isinstance(func, dict) and func.get("name"):
            allowed[func["name"]] = func.get("parameters", {}) or {}
    allowed_names = sorted(list(allowed.keys()))

    for idx, call in enumerate(parsed_tools or []):
        name = (call or {}).get("name")
        args = (call or {}).get("args")

        if not isinstance(name, str) or not name:
            return f"工具调用 #{idx + 1}: 缺少工具名称"

        if name not in allowed:
            return (
                f"工具调用 #{idx + 1}: 未知工具 '{name}'. "
                f"可用工具: {allowed_names}"
            )

        if not isinstance(args, dict):
            return f"工具调用 #{idx + 1} '{name}': 参数必须是 JSON 对象, 得到 {_schema_type_name(args)}"

        schema = allowed[name] or {}
        errs = _validate_value_against_schema(args, schema, path=f"{name}")
        if errs:
            preview = "; ".join(errs[:6])
            more = f" (+{len(errs) - 6} more)" if len(errs) > 6 else ""
            return f"工具调用 #{idx + 1} '{name}': Schema 验证失败: {preview}{more}"

    return None


# ---------------------------------------------------------------------------
# 流式函数调用检测器
# ---------------------------------------------------------------------------

class StreamingFunctionCallDetector:
    """流式工具调用检测器

    核心特性：
    1. 避免在思维包装块内触发工具调用检测
    2. 正常输出思维包装块内容给用户
    3. 支持嵌套 details/think/reasoning/thought 标签
    """

    def __init__(self, trigger_signal: str):
        self.trigger_signal = trigger_signal
        self.reset()

    def reset(self):
        self.content_buffer = ""
        self.state = "detecting"  # detecting, tool_candidate, tool_parsing
        self.in_think_block = False
        self.think_depth = 0
        self.think_stack: List[str] = []
        self.signal = self.trigger_signal
        self.signal_len = len(self.signal)
        self.bare_open_len = len("<function_calls>")

    def process_chunk(self, delta_content: str) -> Tuple[bool, str]:
        """处理流式内容片段

        Returns:
            (is_tool_call_detected, content_to_yield)
        """
        if not delta_content:
            return False, ""

        self.content_buffer += delta_content

        if self.state in ("tool_parsing", "tool_candidate"):
            return False, ""

        buf = self.content_buffer
        buf_len = len(buf)
        parts: list = []
        i = 0

        while i < buf_len:
            # --- 思维标签处理 ---
            skip_chars = self._update_think_state(i)
            if skip_chars > 0:
                end = min(i + skip_chars, buf_len)
                parts.append(buf[i:end])
                i = end
                continue

            if not self.in_think_block:
                # --- 主路径：严格匹配当前会话信号 ---
                if (i + self.signal_len <= buf_len
                        and buf[i:i + self.signal_len] == self.signal):
                    logger.debug("🔧 检测到触发信号 (非思维块内), 切换到工具解析模式")
                    self.state = "tool_parsing"
                    self.content_buffer = buf[i:]
                    return True, "".join(parts)

                # --- 兜底：会话信号漂移时，允许匹配标准 Function 触发器格式 ---
                fallback = _FALLBACK_TRIGGER_RE.match(buf, i)
                if fallback:
                    detected_signal = fallback.group(0)
                    if detected_signal != self.signal:
                        logger.warning(
                            "⚠️ 检测到 trigger_signal 漂移, 使用流内信号: {} -> {}",
                            self.signal,
                            detected_signal,
                        )
                        self.trigger_signal = detected_signal
                        self.signal = detected_signal
                        self.signal_len = len(detected_signal)
                    logger.debug("🔧 检测到触发信号(兜底匹配), 切换到工具解析模式")
                    self.state = "tool_parsing"
                    self.content_buffer = buf[i:]
                    return True, "".join(parts)

                # --- 低置信兜底：允许直接以 <function_calls> 开头 ---
                bare_match = _BARE_FUNCTION_CALLS_OPEN_RE.match(buf, i)
                if bare_match and self._line_prefix_is_whitespace(buf, i):
                    logger.debug("🔧 检测到 bare <function_calls>, 进入候选解析模式")
                    self.state = "tool_candidate"
                    self.content_buffer = buf[i:]
                    return True, "".join(parts)

            # --- look-ahead 保留：尾部可能是不完整的标签/信号 ---
            remaining_len = buf_len - i
            lookahead_len = max(self.signal_len, self.bare_open_len, 32)
            if remaining_len < lookahead_len:
                break

            # --- 批量跳转到下一个感兴趣的位置 ---
            search_start = i + 1
            next_pos = buf.find("<", search_start)

            if next_pos == -1:
                # 没有更多感兴趣位置，批量输出到 look-ahead 边界
                safe_end = buf_len - lookahead_len + 1
                if safe_end > i:
                    parts.append(buf[i:safe_end])
                    i = safe_end
                else:
                    break
            else:
                # 批量输出到目标位置前（但尊重 look-ahead 边界）
                safe_end = min(next_pos, buf_len - lookahead_len + 1)
                if safe_end > i:
                    parts.append(buf[i:safe_end])
                    i = safe_end
                else:
                    # next_pos 在 look-ahead 区域内，逐字符推进
                    parts.append(buf[i])
                    i += 1

        self.content_buffer = buf[i:]
        return False, "".join(parts)

    def _update_think_state(self, pos: int) -> int:
        buf = self.content_buffer
        token = _match_known_tag_token(buf, pos, _THINKING_TAG_TOKEN_RE)
        if token is None:
            return 0

        if token.is_close:
            while self.think_stack and self.think_stack[-1] != token.name:
                self.think_stack.pop()
            if self.think_stack and self.think_stack[-1] == token.name:
                self.think_stack.pop()
        else:
            self.think_stack.append(token.name)

        self.think_depth = len(self.think_stack)
        self.in_think_block = self.think_depth > 0
        return token.end - token.start

    @staticmethod
    def _line_prefix_is_whitespace(buf: str, pos: int) -> bool:
        """检查当前位置是否位于新行起始处（允许前导空白）。"""
        line_start = buf.rfind("\n", 0, pos)
        if line_start == -1:
            prefix = buf[:pos]
        else:
            prefix = buf[line_start + 1:pos]
        return not prefix.strip()

    def flush(self) -> str:
        """流结束时刷出被 look-ahead 保留的剩余内容。

        process_chunk 会保留最后 signal_len 个字符用于触发信号匹配，
        当流正常结束时必须调用此方法将其释放。
        """
        remaining = self.content_buffer
        self.content_buffer = ""
        return remaining

    def reject_candidate(self) -> str:
        """候选 bare XML 解析失败时，将缓冲区恢复为普通文本。"""
        remaining = self.content_buffer
        self.reset()
        return remaining

    def finalize(self) -> Optional[List[Dict[str, Any]]]:
        """流结束时的最终处理"""
        if self.state == "tool_parsing":
            return parse_function_calls_xml(self.content_buffer, self.trigger_signal)
        if self.state == "tool_candidate":
            return parse_function_calls_xml(
                self.content_buffer,
                self.trigger_signal,
                allow_bare=True,
                bare_tail_only=True,
            )
        return None


# ---------------------------------------------------------------------------
# 完整性守卫（来自 PR #8）
# ---------------------------------------------------------------------------

def looks_like_complete_function_calls(buf: str) -> bool:
    """检查缓冲区是否包含完整的 <function_calls> XML 结构。

    防止流式传输中因 HTTP 分块边界导致的 XML 片段被过早解析。
    使用结构扫描 + 自动补闭合标签，而不是简单计数。
    """
    if not buf:
        return False

    buf = normalize_xml_structure(buf)
    start = buf.find("<function_calls>")
    end = buf.rfind("</function_calls>")
    if start == -1 or end == -1 or end < start:
        return False

    candidate = buf[start : end + len("</function_calls>")]
    structure = scan_xmlfc_structure(candidate, final=False)
    if not structure.is_repairable or not structure.is_complete:
        return False

    repaired = repair_unclosed_cdata(repair_xmlfc_structure(candidate, final=False))
    try:
        ET.fromstring(repaired)
    except ET.ParseError:
        return False
    return True


# ---------------------------------------------------------------------------
# 工具调用结果格式化
# ---------------------------------------------------------------------------


def _wrap_cdata(text: str) -> str:
    safe = (text or "").replace("]]>", "]]]]><![CDATA[>")
    return f"<![CDATA[{safe}]]>"


def _normalize_tool_arguments_dict(arguments_val: Any) -> Dict[str, Any]:
    try:
        if isinstance(arguments_val, dict):
            return arguments_val
        if isinstance(arguments_val, str):
            parsed = json.loads(arguments_val or "{}")
            return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
    return {}


def _should_emit_arg_via_args_kv(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if "\n" in value or "\r" in value:
        return True
    if "<" in value or ">" in value:
        return True
    if len(value) > 200 and any(ch in value for ch in ('"', "'", "{", "}", "[", "]")):
        return True
    return False


def _split_xmlfc_arguments(args_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    args_json: Dict[str, Any] = {}
    args_kv: Dict[str, str] = {}

    for key, value in (args_dict or {}).items():
        if _should_emit_arg_via_args_kv(value):
            args_kv[str(key)] = str(value)
        else:
            args_json[str(key)] = value

    return args_json, args_kv


def _build_function_call_xml(name: str, args_dict: Dict[str, Any]) -> str:
    args_json, args_kv = _split_xmlfc_arguments(args_dict)
    parts = [
        "<function_call>",
        f"<tool>{name}</tool>",
    ]

    if args_json or not args_kv:
        parts.append(
            f"<args_json>{_wrap_cdata(json.dumps(args_json, ensure_ascii=False))}</args_json>"
        )

    if args_kv:
        parts.append("<args_kv>")
        for key, value in args_kv.items():
            key_attr = html.escape(key, quote=True)
            parts.append(
                f'<arg name="{key_attr}">{_wrap_cdata(value)}</arg>'
            )
        parts.append("</args_kv>")

    parts.append("</function_call>")
    return "\n".join(parts)


def format_assistant_tool_calls_for_ai(
    tool_calls: List[Dict[str, Any]],
    trigger_signal: str,
) -> str:
    """将历史 assistant tool_calls 格式化为 AI 可理解的文本。"""

    xml_calls_parts = []
    for tool_call in tool_calls:
        function_info = tool_call.get("function", {})
        name = function_info.get("name", "")
        arguments_val = function_info.get("arguments", "{}")

        if not name:
            continue

        args_dict = _normalize_tool_arguments_dict(arguments_val)
        xml_calls_parts.append(_build_function_call_xml(name, args_dict))

    all_calls = "\n".join(xml_calls_parts)
    prefix = f"{trigger_signal}\n" if trigger_signal else ""
    return f"{prefix}<function_calls>\n{all_calls}\n</function_calls>"


# ---------------------------------------------------------------------------
# 兼容性: 保留旧的 JSON 解析作为降级方案
# ---------------------------------------------------------------------------

def parse_and_extract_tool_calls(content: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    从响应内容中提取 tool_calls JSON（降级兼容方案）

    Args:
        content: 模型返回的文本内容

    Returns:
        Tuple[Optional[List], str]: (提取的 tool_calls 列表, 清理后的内容)
    """
    if not content or not content.strip():
        return None, content

    tool_calls = None
    cleaned_content = content

    # 方法1: 尝试解析 JSON 代码块中的 tool_calls
    json_block_pattern = r'```(?:json)?\s*\n?(\{[\s\S]*?\})\s*\n?```'
    json_blocks = re.findall(json_block_pattern, content)

    for json_str in json_blocks:
        try:
            parsed_data = json.loads(json_str)
            if "tool_calls" in parsed_data:
                tool_calls = parsed_data["tool_calls"]
                if tool_calls and isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if tc.get("function"):
                            func = tc["function"]
                            if func.get("arguments"):
                                if isinstance(func["arguments"], dict):
                                    func["arguments"] = json.dumps(func["arguments"], ensure_ascii=False)
                                elif not isinstance(func["arguments"], str):
                                    func["arguments"] = str(func["arguments"])
                    break
        except json.JSONDecodeError:
            continue

    # 方法2: 内联 JSON 搜索
    if not tool_calls:
        i = 0
        while i < len(content):
            if content[i] == '{':
                brace_count = 1
                j = i + 1
                in_string = False
                escape_next = False

                while j < len(content) and brace_count > 0:
                    if escape_next:
                        escape_next = False
                    elif content[j] == '\\':
                        escape_next = True
                    elif content[j] == '"':
                        in_string = not in_string
                    elif not in_string:
                        if content[j] == '{':
                            brace_count += 1
                        elif content[j] == '}':
                            brace_count -= 1
                    j += 1

                if brace_count == 0:
                    json_candidate = content[i:j]
                    try:
                        parsed_data = json.loads(json_candidate)
                        if "tool_calls" in parsed_data:
                            tool_calls = parsed_data["tool_calls"]
                            if tool_calls and isinstance(tool_calls, list):
                                for tc in tool_calls:
                                    if tc.get("function"):
                                        func = tc["function"]
                                        if func.get("arguments"):
                                            if isinstance(func["arguments"], dict):
                                                func["arguments"] = json.dumps(func["arguments"], ensure_ascii=False)
                                            elif not isinstance(func["arguments"], str):
                                                func["arguments"] = str(func["arguments"])
                                break
                    except json.JSONDecodeError:
                        pass
                i = j
            else:
                i += 1

    if tool_calls:
        cleaned_content = remove_tool_json_content(content)

    return tool_calls, cleaned_content


def remove_tool_json_content(content: str) -> str:
    """从响应内容中移除工具调用 JSON"""
    if not content:
        return content

    cleaned_text = content

    def replace_json_block(match):
        json_content = match.group(1)
        try:
            parsed_data = json.loads(json_content)
            if "tool_calls" in parsed_data:
                return ""
        except json.JSONDecodeError:
            pass
        return match.group(0)

    json_block_pattern = r'```(?:json)?\s*\n?(\{[\s\S]*?\})\s*\n?```'
    cleaned_text = re.sub(json_block_pattern, replace_json_block, cleaned_text)

    result = []
    i = 0
    while i < len(cleaned_text):
        if cleaned_text[i] == '{':
            brace_count = 1
            j = i + 1
            in_string = False
            escape_next = False

            while j < len(cleaned_text) and brace_count > 0:
                if escape_next:
                    escape_next = False
                elif cleaned_text[j] == '\\':
                    escape_next = True
                elif cleaned_text[j] == '"':
                    in_string = not in_string
                elif not in_string:
                    if cleaned_text[j] == '{':
                        brace_count += 1
                    elif cleaned_text[j] == '}':
                        brace_count -= 1
                j += 1

            if brace_count == 0:
                json_candidate = cleaned_text[i:j]
                try:
                    parsed = json.loads(json_candidate)
                    if "tool_calls" in parsed:
                        i = j
                        continue
                except json.JSONDecodeError:
                    pass

            result.append(cleaned_text[i])
            i += 1
        else:
            result.append(cleaned_text[i])
            i += 1

    cleaned_result = "".join(result).strip()
    cleaned_result = re.sub(r'\n{3,}', '\n\n', cleaned_result)
    return cleaned_result


def content_to_string(content: Any) -> str:
    """将消息内容转换为字符串"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)
    else:
        return str(content)
