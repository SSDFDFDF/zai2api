#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""XMLFC parsing, validation, and serialization helpers."""

import html
import json
import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.utils.logger import get_logger
from .xmlfc_structure import (
    _is_xml_noise,
    normalize_cdata_markers,
    normalize_xml_structure,
    remove_think_blocks,
    repair_json_payload,
    repair_unclosed_cdata,
    repair_xmlfc_structure,
    scan_xmlfc_structure,
)

logger = get_logger()

_FUNCTION_CALLS_BLOCK_RE = re.compile(r"<function_calls>([\s\S]*?)</function_calls>")
_ARG_BLOCK_RE = re.compile(
    r"<\s*arg\b([^>]*)>([\s\S]*?)</\s*arg\s*>",
    re.IGNORECASE,
)
_ARG_NAME_ATTR_RE = re.compile(
    r"""\b(?:name|key)\s*=\s*(?:"([^"]*)"|'([^']*)')""",
    re.IGNORECASE,
)


@dataclass
class FunctionCallsBlockInspection:
    """调试用的 <function_calls> 候选块诊断结果。"""

    has_candidate: bool = False
    source: str = ""
    reason: str = ""
    candidate_length: int = 0
    trigger_present: bool = False
    trailing_text_length: int = 0
    trailing_preview: str = ""
    candidate_preview: str = ""
    open_stack: List[str] = field(default_factory=list)
    auto_closed_tags: List[str] = field(default_factory=list)
    dropped_closing_tags: List[str] = field(default_factory=list)
    is_repairable: bool = False
    is_complete: bool = False
    etree_ok: bool = False
    etree_error: str = ""

    def to_log_string(self) -> str:
        parts = [
            f"candidate={self.has_candidate}",
            f"source={self.source or '-'}",
            f"reason={self.reason or '-'}",
            f"len={self.candidate_length}",
            f"trigger={self.trigger_present}",
            f"repairable={self.is_repairable}",
            f"complete={self.is_complete}",
            f"etree_ok={self.etree_ok}",
            f"open_stack={self.open_stack}",
        ]
        if self.auto_closed_tags:
            parts.append(f"auto_closed={self.auto_closed_tags}")
        if self.dropped_closing_tags:
            parts.append(f"dropped_closing={self.dropped_closing_tags}")
        if self.trailing_text_length:
            parts.append(f"trailing_len={self.trailing_text_length}")
        if self.etree_error:
            parts.append(f"etree_error={self.etree_error}")
        if self.trailing_preview:
            parts.append(f"trailing_preview={self.trailing_preview!r}")
        if self.candidate_preview:
            parts.append(f"candidate_preview={self.candidate_preview!r}")
        return ", ".join(parts)


def _preview_debug_text(text: str, limit: int = 160) -> str:
    """压缩调试预览，避免日志刷出整段大文本。"""
    if not text:
        return ""
    normalized = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


def _repair_unescaped_quotes(s: str) -> Optional[Dict[str, Any]]:
    """迭代修复 JSON 字符串值内的未转义双引号。"""
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
            quote_pos = current.rfind('"', 0, pos)
            if quote_pos <= 0:
                return None

            num_bs = 0
            check = quote_pos - 1
            while check >= 0 and current[check] == "\\":
                num_bs += 1
                check -= 1
            if num_bs % 2 == 1:
                return None
            current = current[:quote_pos] + "\\" + current[quote_pos:]
    return None


def _parse_args_json_payload(payload: str) -> Optional[Dict[str, Any]]:
    """鲁棒的 args_json 解析。"""
    if payload is None:
        return {}

    s = payload.strip()
    if not s:
        return {}

    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

    def _try_parse(x: str):
        try:
            y = json.loads(x)
            return y if isinstance(y, dict) else None
        except Exception:
            return None

    s = repair_json_payload(s)

    parsed = _try_parse(s)
    if parsed is not None:
        return parsed

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
                        candidate = s[start : i + 1]
                        prefix = s[:start].strip()
                        suffix = s[i + 1 :].strip()
                        if prefix or suffix:
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

    repaired = _repair_unescaped_quotes(s)
    if repaired is not None:
        logger.debug("🔧 args_json 通过未转义引号修复恢复成功")
        return repaired

    logger.debug("🔧 args_json 在所有恢复尝试后仍无效")
    return None


def _extract_cdata_text(raw: str) -> str:
    """提取 CDATA 文本。"""
    if raw is None:
        return ""

    raw = normalize_cdata_markers(raw)
    if "<![CDATA[" not in raw:
        return raw

    parts = re.findall(r"<!\[CDATA\[(.*?)\]\]>", raw, flags=re.DOTALL)
    if parts:
        return "".join(parts)

    st = raw.find("<![CDATA[")
    if st != -1:
        content_start = st + len("<![CDATA[")
        ed = raw.rfind("]]>")
        if ed > content_start:
            return raw[content_start:ed]
        tail = raw[content_start:]
        tail = re.sub(r"\]\]?>?$", "", tail)
        logger.debug("🔧 CDATA 流式截断恢复: 提取 {} 字符", len(tail))
        return tail
    return raw


def _parse_args_kv_text(raw: Optional[str], *, from_xml: bool = False) -> str:
    """解析 args_kv 中单个 arg 的文本值。"""
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
    """从 ET 节点解析 <args_kv><arg ...>...</arg></args_kv>。"""
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


def _find_last_function_calls_match(
    text: str,
    *,
    tail_only: bool = False,
) -> Optional[re.Match[str]]:
    """查找最后一个完整的 <function_calls> 块。"""
    last_match: Optional[re.Match[str]] = None
    for match in _FUNCTION_CALLS_BLOCK_RE.finditer(text or ""):
        if tail_only and text[match.end() :].strip():
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
    """定位当前回复中的工具调用 XML 块。"""
    if not xml_string:
        return None

    cleaned_content = remove_think_blocks(xml_string)

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


def inspect_function_calls_block(
    xml_string: str,
    trigger_signal: str,
    *,
    allow_bare: bool = False,
    bare_tail_only: bool = False,
) -> FunctionCallsBlockInspection:
    """诊断当前缓冲区中是否存在可解析的 <function_calls> 候选块。"""
    if not xml_string:
        return FunctionCallsBlockInspection(reason="empty_input")

    cleaned_content = normalize_xml_structure(remove_think_blocks(xml_string))
    trigger_present = bool(
        trigger_signal and trigger_signal in cleaned_content
    )

    located = locate_function_calls_block(
        xml_string,
        trigger_signal,
        allow_bare=allow_bare,
        bare_tail_only=bare_tail_only,
    )
    if located is None:
        inspection = FunctionCallsBlockInspection(
            reason="no_locatable_block",
            trigger_present=trigger_present,
            candidate_preview=_preview_debug_text(cleaned_content[-200:]),
        )

        if "</function_calls>" not in cleaned_content:
            inspection.reason = "root_close_not_seen"
            return inspection

        if trigger_signal and trigger_present:
            trigger_pos = cleaned_content.rfind(trigger_signal)
            after_trigger = cleaned_content[
                trigger_pos + len(trigger_signal) :
            ]
            if "<function_calls>" not in after_trigger:
                inspection.reason = "no_function_calls_after_trigger"
                return inspection

        if allow_bare:
            bare_match = _find_last_function_calls_match(
                cleaned_content,
                tail_only=False,
            )
            if bare_match is not None:
                trailing = cleaned_content[bare_match.end() :]
                if bare_tail_only and trailing.strip():
                    inspection.reason = "bare_xml_has_trailing_text"
                    inspection.trailing_text_length = len(trailing)
                    inspection.trailing_preview = _preview_debug_text(trailing)
                    inspection.candidate_length = len(bare_match.group(0))
                    inspection.candidate_preview = _preview_debug_text(
                        bare_match.group(0)[-200:]
                    )
                    return inspection

        return inspection

    calls_xml, _, source = located
    structure = scan_xmlfc_structure(calls_xml, final=False)
    repaired_calls_xml = repair_unclosed_cdata(
        repair_xmlfc_structure(calls_xml, final=False)
    )

    inspection = FunctionCallsBlockInspection(
        has_candidate=True,
        source=source,
        reason="ok",
        candidate_length=len(calls_xml),
        trigger_present=trigger_present,
        candidate_preview=_preview_debug_text(calls_xml[-200:]),
        open_stack=list(structure.open_stack),
        auto_closed_tags=list(structure.auto_closed_tags),
        dropped_closing_tags=list(structure.dropped_closing_tags),
        is_repairable=structure.is_repairable,
        is_complete=structure.is_complete,
    )

    try:
        ET.fromstring(repaired_calls_xml)
        inspection.etree_ok = True
    except ET.ParseError as exc:
        inspection.etree_error = f"{type(exc).__name__}: {exc}"
        if structure.is_complete:
            inspection.reason = "etree_parse_failed"
        else:
            inspection.reason = "structure_incomplete"
    else:
        if not structure.is_complete:
            inspection.reason = "structure_incomplete"

    return inspection


def parse_function_calls_xml(
    xml_string: str,
    trigger_signal: str,
    *,
    allow_bare: bool = False,
    bare_tail_only: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """解析 XML 格式的工具调用。"""
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
    logger.debug(
        "🔧 XML 块定位成功, source={}, len={}, tail={}",
        source,
        len(calls_xml),
        _preview_debug_text(calls_xml[-160:]),
    )

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

    def _coerce_value(v: str):
        try:
            return json.loads(v)
        except Exception:
            return v

    results: List[Dict[str, Any]] = []

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
                args_el = fc.find("args")
                if args_el is not None:
                    for child in list(args_el):
                        args[child.tag] = _coerce_value(child.text or "")

            results.append({"name": name, "args": args})

        if results:
            logger.debug("🔧 XML 解析结果 (ET): {} 个工具调用", len(results))
            return results
    except Exception as e:
        logger.debug(
            "🔧 XML 库解析失败, 降级到正则: {}: {}, repaired_tail={}",
            type(e).__name__,
            e,
            _preview_debug_text(repaired_calls_xml[-200:]),
        )

    results = []
    call_blocks = re.findall(r"<function_call>([\s\S]*?)</function_call>", calls_content)

    for i, block in enumerate(call_blocks):
        tool_match = re.search(r"<tool>(.*?)</tool>", block)
        if not tool_match:
            continue

        name = tool_match.group(1).strip()
        args: Dict[str, Any] = {}
        has_structured_args = False

        args_json_open = block.find("<args_json>")
        args_json_close = block.find("</args_json>")
        if (
            args_json_open != -1
            and args_json_close != -1
            and args_json_close > args_json_open
        ):
            raw_payload = block[args_json_open + len("<args_json>") : args_json_close]
            payload = _extract_cdata_text(raw_payload)
            parsed_args = _parse_args_json_payload(payload)
            if parsed_args is None:
                logger.debug("🔧 function_call #{} (正则) args_json 无效; 视为解析失败", i + 1)
                return None
            args = parsed_args
            has_structured_args = True
        elif args_json_open != -1 and args_json_close == -1:
            raw_tail = block[args_json_open + len("<args_json>") :]
            payload = _extract_cdata_text(raw_tail)
            parsed_args = _parse_args_json_payload(payload)
            if parsed_args is not None:
                logger.debug(
                    "🔧 function_call #{} (正则) args_json 流式截断恢复成功: {}",
                    i + 1,
                    list(parsed_args.keys()),
                )
                args = parsed_args
                has_structured_args = True
            else:
                logger.debug(
                    "🔧 function_call #{} (正则) args_json 流式截断恢复失败, 使用空 args",
                    i + 1,
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
    """Best-effort JSON Schema 验证。"""
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
            errors.append(
                f"{path}: expected type '{stype}', got '{_schema_type_name(value)}'"
            )
            return errors
    elif isinstance(stype, list):
        if not any(_type_ok(t) for t in stype if isinstance(t, str)):
            errors.append(
                f"{path}: expected type in {stype!r}, got '{_schema_type_name(value)}'"
            )
            return errors

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
                errors.extend(
                    _validate_value_against_schema(
                        v,
                        props.get(k) or {},
                        f"{path}.{k}",
                        depth + 1,
                    )
                )

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, v in enumerate(value):
                errors.extend(
                    _validate_value_against_schema(v, items, f"{path}[{i}]", depth + 1)
                )

    return errors


def validate_parsed_tools(
    parsed_tools: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    """验证解析出的工具调用是否符合声明的工具定义。"""
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
            return (
                f"工具调用 #{idx + 1} '{name}': 参数必须是 JSON 对象, "
                f"得到 {_schema_type_name(args)}"
            )

        schema = allowed[name] or {}
        errs = _validate_value_against_schema(args, schema, path=f"{name}")
        if errs:
            preview = "; ".join(errs[:6])
            more = f" (+{len(errs) - 6} more)" if len(errs) > 6 else ""
            return (
                f"工具调用 #{idx + 1} '{name}': Schema 验证失败: {preview}{more}"
            )

    return None


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


def _split_xmlfc_arguments(
    args_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
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
            parts.append(f'<arg name="{key_attr}">{_wrap_cdata(value)}</arg>')
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
