#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel


class ImageUrl(BaseModel):
    """Image URL model for vision content"""
    url: str
    detail: Optional[str] = None

    model_config = {"extra": "allow"}


class InputAudio(BaseModel):
    """Input audio payload for multimodal requests."""

    data: str
    format: str

    model_config = {"extra": "allow"}


class FilePayload(BaseModel):
    """File payload for chat-compatible requests."""

    file_id: Optional[str] = None
    file_data: Optional[str] = None
    file_url: Optional[str] = None
    filename: Optional[str] = None

    model_config = {"extra": "allow"}


class ContentPart(BaseModel):
    """Content part model for OpenAI's new content format"""

    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None
    input_audio: Optional[InputAudio] = None
    file: Optional[FilePayload] = None
    file_id: Optional[str] = None
    file_data: Optional[str] = None
    file_url: Optional[str] = None
    filename: Optional[str] = None

    model_config = {"extra": "allow"}


class Message(BaseModel):
    """Chat message model"""

    role: str
    content: Optional[Union[str, List[ContentPart]]] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    is_error: Optional[bool] = None


class OpenAIRequest(BaseModel):
    """OpenAI-compatible request model"""

    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    enable_thinking: Optional[bool] = None
    web_search: Optional[bool] = None

    # 场景透传字段（客户端可直接指定上游场景配置）
    mcp_servers: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None
    flags: Optional[List[str]] = None
    started_at: Optional[float] = None

class Model(BaseModel):
    """Model information for listing"""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    name: Optional[str] = None
    is_active: Optional[bool] = None
    updated_at: Optional[int] = None
    capabilities: Optional[Dict[str, Any]] = None
    mcpServerIds: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class ModelsResponse(BaseModel):
    """Models list response model"""

    object: str = "list"
    data: List[Model]
