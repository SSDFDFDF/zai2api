#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""XMLFC structural rules, normalization, and repair helpers."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

from app.utils.logger import get_logger

logger = get_logger()

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


_KNOWN_TAG_REPAIRS = [
    (
        re.compile(r"<\s*function[\s_-]*calls\s*>", re.IGNORECASE),
        re.compile(r"</\s*function[\s_-]*calls\s*>", re.IGNORECASE),
        "<function_calls>",
        "</function_calls>",
    ),
    (
        re.compile(r"<\s*function[\s_-]*call\s*>(?!\s*s)", re.IGNORECASE),
        re.compile(r"</\s*function[\s_-]*call\s*>(?!\s*s)", re.IGNORECASE),
        "<function_call>",
        "</function_call>",
    ),
    (
        re.compile(r"<\s*tool\s*>", re.IGNORECASE),
        re.compile(r"</\s*tool\s*>", re.IGNORECASE),
        "<tool>",
        "</tool>",
    ),
    (
        re.compile(r"<\s*args[\s_-]*json\s*>", re.IGNORECASE),
        re.compile(r"</\s*args[\s_-]*json\s*>", re.IGNORECASE),
        "<args_json>",
        "</args_json>",
    ),
    (
        re.compile(r"<\s*args[\s_-]*kv\s*>", re.IGNORECASE),
        re.compile(r"</\s*args[\s_-]*kv\s*>", re.IGNORECASE),
        "<args_kv>",
        "</args_kv>",
    ),
    (
        re.compile(r"<\s*args\s*>", re.IGNORECASE),
        re.compile(r"</\s*args\s*>", re.IGNORECASE),
        "<args>",
        "</args>",
    ),
]

_RE_CDATA_OPEN_FUZZY = re.compile(
    r"<!\s*\[?\s*(?:CDATA\s*)+\[",
    re.IGNORECASE,
)
_RE_CDATA_CLOSE_FUZZY = re.compile(r"\]\s*\]\s*>")


def normalize_cdata_markers(raw: str) -> str:
    """修复畸形 CDATA 标记。"""
    if raw is None:
        return ""

    original = raw
    raw = _RE_CDATA_OPEN_FUZZY.sub("<![CDATA[", raw)

    if "<![CDATA[" in raw:
        raw = _RE_CDATA_CLOSE_FUZZY.sub("]]>", raw)
        open_count = raw.count("<![CDATA[")
        close_count = raw.count("]]>")
        if open_count > close_count:
            raw = re.sub(r"(?<!\])\]>", "]]>", raw)
            logger.debug(
                "🔧 修复单括号 CDATA 终止符: open={}, close_before={}, close_after={}",
                open_count,
                close_count,
                raw.count("]]>"),
            )

    if raw != original:
        logger.debug("🔧 CDATA 标记已修复: {} → {}", repr(original[:60]), repr(raw[:60]))

    return raw


def normalize_xml_tag_names(xml_str: str) -> str:
    """归一化已知 XML 标签名。"""
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


def repair_unclosed_cdata(xml_str: str) -> str:
    """为 XML 字符串中所有未闭合的 CDATA 段补充 ]]> 终止符。"""
    if not xml_str or "<![CDATA[" not in xml_str:
        return xml_str

    open_count = xml_str.count("<![CDATA[")
    close_count = xml_str.count("]]>")
    missing = open_count - close_count
    if missing <= 0:
        return xml_str

    logger.debug("🔧 检测到 {} 个未闭合 CDATA, 自动补全终止符", missing)

    result = xml_str
    for _ in range(missing):
        last_open = result.rfind("<![CDATA[")
        search_start = last_open + len("<![CDATA[")
        close_tag = re.search(r"</", result[search_start:])
        if close_tag:
            insert_pos = search_start + close_tag.start()
            result = result[:insert_pos] + "]]>" + result[insert_pos:]
        else:
            result += "]]>"

    return result


def repair_json_payload(s: str) -> str:
    """修复常见 JSON 畸形: 尾随逗号、Python 布尔值/None。"""
    if not s:
        return s

    original = s
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)

    if s != original:
        logger.debug("🔧 JSON payload 已修复: {} → {}", repr(original[:60]), repr(s[:60]))

    return s


def _is_xml_noise(s: str) -> bool:
    """判断字符串是否全为可忽略的 XML 标记噪声。"""
    if not s:
        return True

    stripped = s.strip()
    if not stripped:
        return True

    return bool(
        re.fullmatch(
            r"(?:"
            r"<!\s*\[?\s*(?:CDATA\s*)*\[?"
            r"|\]\s*\]?\s*>?"
            r"|\s"
            r")+",
            stripped,
            re.IGNORECASE | re.DOTALL,
        )
    )


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


def looks_like_complete_function_calls(buf: str) -> bool:
    """检查缓冲区是否包含完整的 <function_calls> XML 结构。"""
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
