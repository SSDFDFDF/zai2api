#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Toolify 工具调用处理包。"""

from app.core.toolify.glm_handler import GLMToolHandler
from app.core.toolify.handler import ToolifyHandler
from app.core.toolify.message import extract_last_user_text, preprocess_openai_messages
from app.core.toolify.request_handler import (
    ToolifyPreparedRequest,
    ToolifyRequestHandler,
)
from app.core.toolify.xml_protocol import (
    StreamingFunctionCallDetector,
    generate_trigger_signal,
    looks_like_complete_function_calls,
    parse_and_extract_tool_calls,
    parse_function_calls_xml,
    validate_parsed_tools,
)

__all__ = [
    "GLMToolHandler",
    "ToolifyHandler",
    "ToolifyPreparedRequest",
    "ToolifyRequestHandler",
    "extract_last_user_text",
    "preprocess_openai_messages",
    "StreamingFunctionCallDetector",
    "generate_trigger_signal",
    "looks_like_complete_function_calls",
    "parse_and_extract_tool_calls",
    "parse_function_calls_xml",
    "validate_parsed_tools",
]
