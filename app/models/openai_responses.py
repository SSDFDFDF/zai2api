#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class OpenAIResponsesRequest(BaseModel):
    """Minimal OpenAI Responses API request model.

    Phase 1 only validates the fields we actively use. Remaining fields are
    preserved via ``extra=allow`` so clients can send newer optional fields
    without immediate hard failures.
    """

    model: str
    input: Optional[Any] = None
    instructions: Optional[Any] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    previous_response_id: Optional[str] = None
    store: Optional[bool] = None
    reasoning: Optional[Any] = None

    model_config = {"extra": "allow"}
