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
import re
import secrets
import string
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from app.utils.logger import get_logger

logger = get_logger()


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
4. When tool execution results are present in the context, they will be formatted with XML tags like <tool_result>...</tool_result> for easy identification.
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
- The <args_json> tag must contain a single JSON object with all required arguments for that tool.
- You MAY wrap the JSON content inside <![CDATA[...]]> to avoid XML escaping issues.

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

Now please be ready to strictly follow the above specifications.
"""

    logger.debug(f"生成 XML 工具提示词, 包含 {len(tools_list_str)} 个工具定义, 触发信号: {trigger_signal[:20]}...")
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
            logger.warning(f"⚠️ 未知的 tool_choice 值: {tool_choice}")
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
        logger.warning(f"⚠️ 不支持的 tool_choice 类型: {type(tool_choice)}")
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

    logger.debug(f"XML 工具提示已注入到消息列表, 共 {len(processed)} 条消息")
    return processed


# ---------------------------------------------------------------------------
# <think> 块处理
# ---------------------------------------------------------------------------

def remove_think_blocks(text: str) -> str:
    """临时移除所有 <think>...</think> 块用于 XML 解析 (支持嵌套)"""
    while '<think>' in text and '</think>' in text:
        start_pos = text.find('<think>')
        if start_pos == -1:
            break

        pos = start_pos + 7
        depth = 1

        while pos < len(text) and depth > 0:
            if text[pos:pos+7] == '<think>':
                depth += 1
                pos += 7
            elif text[pos:pos+8] == '</think>':
                depth -= 1
                pos += 8
            else:
                pos += 1

        if depth == 0:
            text = text[:start_pos] + text[pos:]
        else:
            break

    return text


def find_last_trigger_signal_outside_think(text: str, trigger_signal: str) -> int:
    """查找不在 <think> 块内的最后一个触发信号位置。找不到返回 -1。"""
    if not text or not trigger_signal:
        return -1

    i = 0
    think_depth = 0
    last_pos = -1

    while i < len(text):
        if text.startswith("<think>", i):
            think_depth += 1
            i += 7
            continue

        if text.startswith("</think>", i):
            think_depth = max(0, think_depth - 1)
            i += 8
            continue

        if think_depth == 0 and text.startswith(trigger_signal, i):
            last_pos = i
            i += 1
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
    """
    if raw is None:
        return ""

    original = raw

    # 修复开头标记
    raw = _RE_CDATA_OPEN_FUZZY.sub('<![CDATA[', raw)

    # 修复终止符 (仅当存在 CDATA 上下文时)
    if '<![CDATA[' in raw:
        raw = _RE_CDATA_CLOSE_FUZZY.sub(']]>', raw)

    if raw != original:
        logger.debug(f"🔧 CDATA 标记已修复: {repr(original[:60])} → {repr(raw[:60])}")

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
        logger.debug(f"🔧 JSON payload 已修复: {repr(original[:60])} → {repr(s[:60])}")

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
        r'|\]\s*\]\s*>?'                 # CDATA 终止符碎片
        r'|\s'                           # 空白
        r')+',
        stripped,
        re.IGNORECASE | re.DOTALL,
    ))


# ---------------------------------------------------------------------------
# XML 解析
# ---------------------------------------------------------------------------


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

    # 恢复策略: 提取平衡的 JSON 对象
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

    logger.debug("🔧 args_json 在所有恢复尝试后仍无效")
    return None


def _extract_cdata_text(raw: str) -> str:
    """提取 CDATA 文本（含畸形 CDATA 修复）"""
    if raw is None:
        return ""

    # ★ 先修复畸形 CDATA 标记
    raw = normalize_cdata_markers(raw)

    if "<![CDATA[" not in raw:
        return raw

    # 拼接分割的 CDATA 段
    parts = re.findall(r"<!\[CDATA\[(.*?)\]\]>", raw, flags=re.DOTALL)
    if parts:
        return "".join(parts)

    # 未终止的 CDATA: 尝试提取内容
    st = raw.find("<![CDATA[")
    if st != -1:
        st += len("<![CDATA[")
        ed = raw.rfind("]]>")
        if ed > st:
            return raw[st:ed]
        return ""
    return raw


def parse_function_calls_xml(
    xml_string: str,
    trigger_signal: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    解析 XML 格式的工具调用

    1. 临时移除 <think> 块
    2. 查找最后一个触发信号
    3. 从触发信号后解析 <function_calls> XML

    Returns:
        解析出的工具调用列表 [{"name": str, "args": dict}, ...] 或 None
    """
    logger.debug(f"🔧 XML 解析开始, 输入长度: {len(xml_string) if xml_string else 0}")

    if not xml_string or trigger_signal not in xml_string:
        logger.debug("🔧 输入为空或不包含触发信号")
        return None

    cleaned_content = remove_think_blocks(xml_string)

    # ★ 先对内容做 XML 通配修复 (标签名/CDATA), 让后续搜索能匹配到畸形标签
    original_cleaned = cleaned_content
    cleaned_content = normalize_xml_structure(cleaned_content)
    if cleaned_content != original_cleaned:
        logger.debug(f"🔧 XML 结构已修复 ({len(original_cleaned)} → {len(cleaned_content)} 字符)")

    # 查找所有触发信号位置
    signal_positions = []
    start_pos = 0
    while True:
        pos = cleaned_content.find(trigger_signal, start_pos)
        if pos == -1:
            break
        signal_positions.append(pos)
        start_pos = pos + 1

    if not signal_positions:
        return None

    # 从最后一个触发信号开始找 <function_calls>
    calls_content_match = None
    for idx in range(len(signal_positions) - 1, -1, -1):
        pos = signal_positions[idx]
        sub = cleaned_content[pos:]
        m = re.search(r"<function_calls>([\s\S]*?)</function_calls>", sub)
        if m:
            calls_content_match = m
            logger.debug(f"🔧 使用触发信号 index {idx}, pos {pos}")
            break

    if calls_content_match is None:
        logger.debug("🔧 未找到 <function_calls> 标签")
        return None

    calls_xml = calls_content_match.group(0)
    calls_content = calls_content_match.group(1)

    def _coerce_value(v: str):
        try:
            return json.loads(v)
        except Exception:
            return v

    results: List[Dict[str, Any]] = []

    # 主路径: 严格 XML 解析
    try:
        root = ET.fromstring(calls_xml)
        for i, fc in enumerate(root.findall("function_call")):
            tool_el = fc.find("tool")
            name = (tool_el.text or "").strip() if tool_el is not None else ""
            if not name:
                continue

            args: Dict[str, Any] = {}

            args_json_el = fc.find("args_json")
            if args_json_el is not None:
                raw_text = args_json_el.text or ""
                payload = _extract_cdata_text(raw_text)
                parsed_args = _parse_args_json_payload(payload)
                if parsed_args is None:
                    logger.debug(f"🔧 function_call #{i+1} 的 args_json 无效; 视为解析失败")
                    return None
                args = parsed_args
            else:
                # Legacy fallback: <args><k>json</k></args>
                args_el = fc.find("args")
                if args_el is not None:
                    for child in list(args_el):
                        args[child.tag] = _coerce_value(child.text or "")

            results.append({"name": name, "args": args})

        if results:
            logger.debug(f"🔧 XML 解析结果 (ET): {len(results)} 个工具调用")
            return results
    except Exception as e:
        logger.debug(f"🔧 XML 库解析失败, 降级到正则: {type(e).__name__}: {e}")

    # 降级路径: 正则解析
    results = []
    call_blocks = re.findall(r"<function_call>([\s\S]*?)</function_call>", calls_content)

    for i, block in enumerate(call_blocks):
        tool_match = re.search(r"<tool>(.*?)</tool>", block)
        if not tool_match:
            continue

        name = tool_match.group(1).strip()
        args: Dict[str, Any] = {}

        # 容错的 args_json 提取（PR #8 改进）
        args_json_open = block.find("<args_json>")
        args_json_close = block.find("</args_json>")
        if args_json_open != -1 and args_json_close != -1 and args_json_close > args_json_open:
            raw_payload = block[args_json_open + len("<args_json>"):args_json_close]
            payload = _extract_cdata_text(raw_payload)
            parsed_args = _parse_args_json_payload(payload)
            if parsed_args is None:
                logger.debug(f"🔧 function_call #{i+1} (正则) args_json 无效; 视为解析失败")
                return None
            args = parsed_args
        else:
            args_block_match = re.search(r"<args>([\s\S]*?)</args>", block)
            if args_block_match:
                args_inner = args_block_match.group(1)
                arg_matches = re.findall(r"<([^\s>/]+)>([\s\S]*?)</\1>", args_inner)
                for k, v in arg_matches:
                    args[k] = _coerce_value(v)

        results.append({"name": name, "args": args})

    logger.debug(f"🔧 正则解析结果: {len(results)} 个工具调用")
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
    1. 避免在 <think> 块内触发工具调用检测
    2. 正常输出 <think> 块内容给用户
    3. 支持嵌套 think 标签
    """

    def __init__(self, trigger_signal: str):
        self.trigger_signal = trigger_signal
        self.reset()

    def reset(self):
        self.content_buffer = ""
        self.state = "detecting"  # detecting, tool_parsing
        self.in_think_block = False
        self.think_depth = 0
        self.signal = self.trigger_signal
        self.signal_len = len(self.signal)

    def process_chunk(self, delta_content: str) -> Tuple[bool, str]:
        """处理流式内容片段

        Returns:
            (is_tool_call_detected, content_to_yield)
        """
        if not delta_content:
            return False, ""

        self.content_buffer += delta_content
        content_to_yield = ""

        if self.state == "tool_parsing":
            return False, ""

        i = 0
        while i < len(self.content_buffer):
            skip_chars = self._update_think_state(i)
            if skip_chars > 0:
                for j in range(skip_chars):
                    if i + j < len(self.content_buffer):
                        content_to_yield += self.content_buffer[i + j]
                i += skip_chars
                continue

            if not self.in_think_block and self._can_detect_signal_at(i):
                if self.content_buffer[i:i+self.signal_len] == self.signal:
                    logger.debug(f"🔧 检测到触发信号 (非 think 块内), 切换到工具解析模式")
                    self.state = "tool_parsing"
                    self.content_buffer = self.content_buffer[i:]
                    return True, content_to_yield

            remaining_len = len(self.content_buffer) - i
            if remaining_len < self.signal_len or remaining_len < 8:
                break

            content_to_yield += self.content_buffer[i]
            i += 1

        self.content_buffer = self.content_buffer[i:]
        return False, content_to_yield

    def _update_think_state(self, pos: int) -> int:
        remaining = self.content_buffer[pos:]
        if remaining.startswith('<think>'):
            self.think_depth += 1
            self.in_think_block = True
            return 7
        elif remaining.startswith('</think>'):
            self.think_depth = max(0, self.think_depth - 1)
            self.in_think_block = self.think_depth > 0
            return 8
        return 0

    def _can_detect_signal_at(self, pos: int) -> bool:
        return (pos + self.signal_len <= len(self.content_buffer) and
                not self.in_think_block)

    def flush(self) -> str:
        """流结束时刷出被 look-ahead 保留的剩余内容。

        process_chunk 会保留最后 signal_len 个字符用于触发信号匹配，
        当流正常结束时必须调用此方法将其释放。
        """
        remaining = self.content_buffer
        self.content_buffer = ""
        return remaining

    def finalize(self) -> Optional[List[Dict[str, Any]]]:
        """流结束时的最终处理"""
        if self.state == "tool_parsing":
            return parse_function_calls_xml(self.content_buffer, self.trigger_signal)
        return None


# ---------------------------------------------------------------------------
# 完整性守卫（来自 PR #8）
# ---------------------------------------------------------------------------

def looks_like_complete_function_calls(buf: str) -> bool:
    """检查缓冲区是否包含完整的 <function_calls> XML 结构。

    防止流式传输中因 HTTP 分块边界导致的 XML 片段被过早解析。
    先对缓冲区做标签归一化, 以便畸形标签也能正确计数。
    """
    if not buf:
        return False

    # ★ 对缓冲区做轻量归一化以匹配畸形标签
    buf = normalize_xml_tag_names(buf)
    buf = normalize_cdata_markers(buf)

    if "<function_calls>" not in buf or "</function_calls>" not in buf:
        return False
    if buf.count("<function_call>") != buf.count("</function_call>"):
        return False
    if ("<args_json>" in buf or "</args_json>" in buf) and buf.count("<args_json>") != buf.count("</args_json>"):
        return False
    if ("<![CDATA[" in buf or "]]>" in buf) and buf.count("<![CDATA[") != buf.count("]]>"):
        return False
    return True


# ---------------------------------------------------------------------------
# 工具调用结果格式化
# ---------------------------------------------------------------------------

def format_tool_result_for_ai(
    tool_name: str,
    tool_arguments: str,
    result_content: str,
) -> str:
    """格式化工具执行结果供上游模型理解。"""
    return (
        f"Tool execution result:\n"
        f"- Tool name: {tool_name}\n"
        f"- Tool arguments: {tool_arguments}\n"
        f"- Execution result:\n"
        f"<tool_result>\n"
        f"{result_content}\n"
        f"</tool_result>"
    )


def format_assistant_tool_calls_for_ai(
    tool_calls: List[Dict[str, Any]],
    trigger_signal: str,
) -> str:
    """将历史 assistant tool_calls 格式化为 AI 可理解的文本。"""

    def _wrap_cdata(text: str) -> str:
        safe = (text or "").replace("]]>", "]]]]><![CDATA[>")
        return f"<![CDATA[{safe}]]>"

    xml_calls_parts = []
    for tool_call in tool_calls:
        function_info = tool_call.get("function", {})
        name = function_info.get("name", "")
        arguments_val = function_info.get("arguments", "{}")

        try:
            if isinstance(arguments_val, dict):
                args_dict = arguments_val
            elif isinstance(arguments_val, str):
                parsed = json.loads(arguments_val or "{}")
                if not isinstance(parsed, dict):
                    args_dict = {}
                else:
                    args_dict = parsed
            else:
                args_dict = {}
        except Exception:
            args_dict = {}

        args_payload = json.dumps(args_dict, ensure_ascii=False)
        xml_call = (
            f"<function_call>\n"
            f"<tool>{name}</tool>\n"
            f"<args_json>{_wrap_cdata(args_payload)}</args_json>\n"
            f"</function_call>"
        )
        xml_calls_parts.append(xml_call)

    all_calls = "\n".join(xml_calls_parts)
    return f"{trigger_signal}\n<function_calls>\n{all_calls}\n</function_calls>"


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
