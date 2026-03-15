#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Translate OpenAI Responses requests into the internal chat request model."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from app.models.openai_responses import OpenAIResponsesRequest
from app.models.schemas import Message, OpenAIRequest


def _stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _extract_image_url(part: Dict[str, Any]) -> str:
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        return str(image_url.get("url") or "")
    return ""


def _normalize_content_part(part: Any) -> Optional[Dict[str, Any]]:
    if isinstance(part, str):
        return {"type": "text", "text": part}

    if not isinstance(part, dict):
        raise ValueError(f"Unsupported content part type: {type(part)!r}")

    part_type = str(part.get("type") or "").strip()
    if part_type in ("text", "input_text", "output_text"):
        return {"type": "text", "text": str(part.get("text") or "")}

    if part_type in ("image_url", "input_image"):
        image_url = _extract_image_url(part)
        if not image_url:
            raise ValueError("input_image requires image_url")
        return {"type": "image_url", "image_url": {"url": image_url}}

    if part_type in ("input_file", "file"):
        raise ValueError("input_file is not supported yet on /v1/responses")

    raise ValueError(f"Unsupported content part type: {part_type or '<empty>'}")


def _normalize_message_content(content: Any) -> Union[str, List[Dict[str, Any]]]:
    if isinstance(content, str):
        return content

    if content is None:
        return ""

    if not isinstance(content, list):
        return _stringify_value(content)

    normalized_parts: List[Dict[str, Any]] = []
    text_only = True
    text_parts: List[str] = []

    for part in content:
        normalized = _normalize_content_part(part)
        if not normalized:
            continue
        normalized_parts.append(normalized)
        if normalized["type"] == "text":
            text_parts.append(str(normalized.get("text") or ""))
        else:
            text_only = False

    if text_only:
        return " ".join(part for part in text_parts if part).strip()
    return normalized_parts


def _normalize_instruction_content(content: Any) -> str:
    normalized = _normalize_message_content(content)
    if isinstance(normalized, list):
        raise ValueError("instructions only supports text in phase 1")
    return normalized


def _responses_tool_to_openai_tool(tool: Any) -> Dict[str, Any]:
    if not isinstance(tool, dict):
        raise ValueError("tools entries must be objects")

    tool_type = str(tool.get("type") or "").strip() or "function"
    if tool_type != "function":
        raise ValueError(f"Unsupported tool type on /v1/responses: {tool_type}")

    function_payload = tool.get("function")
    if isinstance(function_payload, dict):
        name = str(function_payload.get("name") or "").strip()
        description = function_payload.get("description")
        parameters = function_payload.get("parameters")
    else:
        name = str(tool.get("name") or "").strip()
        description = tool.get("description")
        parameters = tool.get("parameters")

    if not name:
        raise ValueError("function tool requires name")

    normalized_function: Dict[str, Any] = {"name": name}
    if description is not None:
        normalized_function["description"] = description
    if parameters is not None:
        normalized_function["parameters"] = parameters

    return {
        "type": "function",
        "function": normalized_function,
    }


def _normalize_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    return [_responses_tool_to_openai_tool(tool) for tool in tools]


def _normalize_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return tool_choice

    choice_type = str(tool_choice.get("type") or "").strip()
    if choice_type in ("auto", "none", "required"):
        return choice_type
    if choice_type == "function":
        name = str(
            tool_choice.get("name")
            or ((tool_choice.get("function") or {}).get("name"))
            or ""
        ).strip()
        if not name:
            raise ValueError("tool_choice.function requires name")
        return {
            "type": "function",
            "function": {"name": name},
        }
    return tool_choice


def _normalize_input_items(raw_input: Any) -> List[Any]:
    if raw_input is None:
        return []
    if isinstance(raw_input, list):
        return raw_input
    return [raw_input]


def _input_item_to_messages(item: Any) -> List[Message]:
    if isinstance(item, str):
        return [Message(role="user", content=item)]

    if not isinstance(item, dict):
        raise ValueError(f"Unsupported input item type: {type(item)!r}")

    item_type = str(item.get("type") or "message").strip() or "message"

    if item_type == "message":
        role = str(item.get("role") or "user").strip() or "user"
        return [
            Message(
                role=role,
                content=_normalize_message_content(item.get("content")),
            )
        ]

    if item_type == "function_call_output":
        call_id = str(item.get("call_id") or item.get("tool_call_id") or "").strip()
        if not call_id:
            raise ValueError("function_call_output requires call_id")
        return [
            Message(
                role="tool",
                tool_call_id=call_id,
                content=_stringify_value(item.get("output")),
            )
        ]

    if item_type == "function_call":
        call_id = str(item.get("call_id") or item.get("id") or "").strip()
        name = str(item.get("name") or "").strip()
        arguments = item.get("arguments")
        if not call_id or not name:
            raise ValueError("function_call requires call_id and name")
        return [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _stringify_value(arguments or {}),
                        },
                    }
                ],
            )
        ]

    raise ValueError(f"Unsupported input item type: {item_type}")


def responses_request_to_openai_request(
    body: OpenAIResponsesRequest,
) -> OpenAIRequest:
    if body.previous_response_id:
        raise ValueError("previous_response_id is not supported yet on /v1/responses")

    messages: List[Message] = []

    if body.instructions is not None:
        instructions = _normalize_instruction_content(body.instructions)
        if instructions:
            messages.append(Message(role="developer", content=instructions))

    for item in _normalize_input_items(body.input):
        messages.extend(_input_item_to_messages(item))

    if not messages:
        raise ValueError("input is required")

    return OpenAIRequest(
        model=body.model,
        messages=messages,
        stream=bool(body.stream),
        temperature=body.temperature,
        max_tokens=body.max_output_tokens,
        tools=_normalize_tools(body.tools),
        tool_choice=_normalize_tool_choice(body.tool_choice),
        enable_thinking=bool(body.reasoning) if body.reasoning is not None else None,
    )
