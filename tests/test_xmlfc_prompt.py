#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.toolify.xmlfc_prompt import (
    generate_tool_prompt,
    process_messages_with_tools,
    process_tool_choice,
)


READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file from disk",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path"},
            },
            "required": ["file_path"],
        },
    },
}


def test_generate_tool_prompt_enforces_pure_tool_turn():
    prompt = generate_tool_prompt([READ_TOOL], "<Function_TEST_Start/>")

    assert "this assistant turn MUST be a pure tool turn" in prompt
    assert "No text before the trigger." in prompt
    assert "No text after </function_calls>." in prompt
    assert "Never mix tool XML with natural-language text" in prompt
    assert "...response content (optional)..." not in prompt
    assert "I will call a tool now." in prompt
    assert "Done." in prompt


def test_generate_tool_prompt_includes_compact_schema_block():
    prompt = generate_tool_prompt([READ_TOOL], "<Function_TEST_Start/>")

    assert "JSON schema:" in prompt
    assert '"file_path"' in prompt
    assert '"required": [' in prompt
    assert "Parameters summary:" not in prompt
    assert "Parameter details:" not in prompt


def test_process_tool_choice_named_tool_requires_pure_tool_turn():
    prompt = process_tool_choice(
        {"type": "function", "function": {"name": "Read"}},
        [READ_TOOL],
    )

    assert "ONLY the tool named `Read`" in prompt
    assert "pure tool turn" in prompt
    assert "no natural-language text" in prompt


def test_process_messages_with_tools_adds_separated_protocol_block():
    messages = [{"role": "system", "content": "Base system instructions."}]

    processed = process_messages_with_tools(
        messages,
        [READ_TOOL],
        tool_choice="auto",
        trigger_signal="<Function_TEST_Start/>",
    )

    assert len(processed) == 1
    content = processed[0]["content"]
    assert content.startswith("Base system instructions.\n\n[XML TOOL CALL PROTOCOL]")
    assert "<Function_TEST_Start/>" in content

