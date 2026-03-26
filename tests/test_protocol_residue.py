#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.core.message_utils import preprocess_openai_messages


def test_preprocess_passes_through_messages():
    """Messages pass through unchanged (except developer -> system)."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    normalized = preprocess_openai_messages(messages)
    assert normalized == messages


def test_preprocess_developer_to_system():
    messages = [{"role": "developer", "content": "Be concise."}]
    normalized = preprocess_openai_messages(messages)
    assert normalized == [{"role": "system", "content": "Be concise."}]
