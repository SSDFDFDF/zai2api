#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""XMLFC protocol compatibility facade.

历史上 ``xml_protocol.py`` 同时承载 prompt 生成、结构修复、XML 解析、
流式 detector、Schema 校验和旧 JSON 降级兼容。

现在这些实现已按职责拆分到平铺文件：
- ``xmlfc_prompt.py``
- ``xmlfc_structure.py``
- ``xmlfc_codec.py``
- ``xmlfc_stream.py``
- ``tool_json_compat.py``

当前文件只负责维持稳定导出，避免外部调用方在重构期间感知到模块拆分。
"""

from .tool_json_compat import (
    content_to_string,
    parse_and_extract_tool_calls,
    remove_tool_json_content,
)
from .xmlfc_codec import (
    _extract_cdata_text,
    _parse_args_json_payload,
    format_assistant_tool_calls_for_ai,
    inspect_function_calls_block,
    locate_function_calls_block,
    parse_function_calls_xml,
    validate_parsed_tools,
)
from .xmlfc_prompt import (
    generate_tool_prompt,
    generate_trigger_signal,
    process_messages_with_tools,
    process_tool_choice,
)
from .xmlfc_stream import DetectorState, StreamingFunctionCallDetector
from .xmlfc_structure import (
    THINKING_XML_TAGS,
    _KnownTagToken,
    _XMLFCStructureScan,
    _is_xml_noise,
    _iter_known_tag_tokens,
    _match_known_tag_token,
    find_last_trigger_signal_outside_think,
    looks_like_complete_function_calls,
    normalize_cdata_markers,
    normalize_xml_structure,
    normalize_xml_tag_names,
    remove_think_blocks,
    repair_json_payload,
    repair_unclosed_cdata,
    repair_xmlfc_structure,
    scan_xmlfc_structure,
)

__all__ = [
    "THINKING_XML_TAGS",
    "DetectorState",
    "StreamingFunctionCallDetector",
    "_KnownTagToken",
    "_XMLFCStructureScan",
    "_extract_cdata_text",
    "_is_xml_noise",
    "_iter_known_tag_tokens",
    "_match_known_tag_token",
    "_parse_args_json_payload",
    "content_to_string",
    "find_last_trigger_signal_outside_think",
    "format_assistant_tool_calls_for_ai",
    "generate_tool_prompt",
    "generate_trigger_signal",
    "inspect_function_calls_block",
    "locate_function_calls_block",
    "looks_like_complete_function_calls",
    "normalize_cdata_markers",
    "normalize_xml_structure",
    "normalize_xml_tag_names",
    "parse_and_extract_tool_calls",
    "parse_function_calls_xml",
    "process_messages_with_tools",
    "process_tool_choice",
    "remove_think_blocks",
    "remove_tool_json_content",
    "repair_json_payload",
    "repair_unclosed_cdata",
    "repair_xmlfc_structure",
    "scan_xmlfc_structure",
    "validate_parsed_tools",
]
