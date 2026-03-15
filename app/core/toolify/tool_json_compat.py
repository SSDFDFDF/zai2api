#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Legacy JSON tool-call compatibility helpers."""

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(\{[\s\S]*?\})\s*\n?```")


def _normalize_tool_call_arguments(tool_calls: Any) -> Optional[List[Dict[str, Any]]]:
    """规范化 tool_calls 中的 function.arguments。"""
    if not isinstance(tool_calls, list):
        return None

    normalized: List[Dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        if tool_call.get("function") and isinstance(tool_call["function"], dict):
            func = tool_call["function"]
            if func.get("arguments"):
                if isinstance(func["arguments"], dict):
                    func["arguments"] = json.dumps(func["arguments"], ensure_ascii=False)
                elif not isinstance(func["arguments"], str):
                    func["arguments"] = str(func["arguments"])
        normalized.append(tool_call)

    return normalized


def _extract_tool_calls_from_json_payload(payload: str) -> Optional[List[Dict[str, Any]]]:
    """尝试从单个 JSON payload 中提取 tool_calls。"""
    try:
        parsed_data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if "tool_calls" not in parsed_data:
        return None
    return _normalize_tool_call_arguments(parsed_data["tool_calls"])


def _iter_balanced_json_objects(text: str) -> Iterator[Tuple[int, int, str]]:
    """扫描文本中的平衡 JSON 对象候选。"""
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue

        brace_count = 1
        j = i + 1
        in_string = False
        escape_next = False

        while j < len(text) and brace_count > 0:
            if escape_next:
                escape_next = False
            elif text[j] == "\\":
                escape_next = True
            elif text[j] == '"':
                in_string = not in_string
            elif not in_string:
                if text[j] == "{":
                    brace_count += 1
                elif text[j] == "}":
                    brace_count -= 1
            j += 1

        if brace_count == 0:
            yield i, j, text[i:j]
            i = j
            continue

        i += 1


def parse_and_extract_tool_calls(content: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """从响应内容中提取 tool_calls JSON（降级兼容方案）。"""
    if not content or not content.strip():
        return None, content

    tool_calls = None
    cleaned_content = content

    for json_str in _JSON_BLOCK_RE.findall(content):
        tool_calls = _extract_tool_calls_from_json_payload(json_str)
        if tool_calls:
            break

    if not tool_calls:
        for _, _, json_candidate in _iter_balanced_json_objects(content):
            tool_calls = _extract_tool_calls_from_json_payload(json_candidate)
            if tool_calls:
                break

    if tool_calls:
        cleaned_content = remove_tool_json_content(content)

    return tool_calls, cleaned_content


def remove_tool_json_content(content: str) -> str:
    """从响应内容中移除工具调用 JSON。"""
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

    cleaned_text = _JSON_BLOCK_RE.sub(replace_json_block, cleaned_text)

    result = []
    i = 0
    while i < len(cleaned_text):
        if cleaned_text[i] != "{":
            result.append(cleaned_text[i])
            i += 1
            continue

        consumed = False
        for start, end, json_candidate in _iter_balanced_json_objects(cleaned_text[i:]):
            if start != 0:
                break
            if _extract_tool_calls_from_json_payload(json_candidate) is not None:
                i += end
                consumed = True
            break

        if consumed:
            continue

        result.append(cleaned_text[i])
        i += 1

    cleaned_result = "".join(result).strip()
    cleaned_result = re.sub(r"\n{3,}", "\n\n", cleaned_result)
    return cleaned_result


def content_to_string(content: Any) -> str:
    """将消息内容转换为字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)
    return str(content)
