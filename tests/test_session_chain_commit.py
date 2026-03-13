#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Regression tests for session chain commit timing."""

import asyncio

from app.core.session.session_manager import SessionManager


def run(coro):
    return asyncio.run(coro)


def _session_key(mgr: SessionManager, model: str, chat_id: str) -> str:
    client_fp = mgr._fp.generate_client_fingerprint(model, model)
    return f"session:{client_fp}:{chat_id}"


def test_reuse_session_does_not_advance_parent_before_commit():
    mgr = SessionManager(session_ttl=3600, max_sessions_per_client=5)

    msg1 = {"role": "user", "content": "Q1"}
    msg2 = {"role": "assistant", "content": "A1"}
    msg3 = {"role": "user", "content": "Q2"}
    msg4 = {"role": "assistant", "content": "A2"}
    msg5 = {"role": "user", "content": "Q3"}

    round1 = [msg1]
    round2 = [msg1, msg2, msg3]
    round3 = [msg1, msg2, msg3, msg4, msg5]

    run(
        mgr.create_session(
            auth_token="token_a",
            model="glm-5",
            messages=round1,
            chat_id="chat-1",
            message_id="user-msg-1",
        )
    )

    first_retry = run(mgr.find_session(model="glm-5", messages=round2))
    assert first_retry is not None
    assert first_retry.parent_id == "user-msg-1"

    stored = run(mgr._store.get(_session_key(mgr, "glm-5", "chat-1")))
    assert stored["last_message_id"] == "user-msg-1"

    second_retry = run(mgr.find_session(model="glm-5", messages=round2))
    assert second_retry is not None
    assert second_retry.parent_id == "user-msg-1"

    run(
        mgr.commit_session_turn(
            model="glm-5",
            messages=round2,
            chat_id="chat-1",
            message_id=first_retry.message_id,
        )
    )

    stored = run(mgr._store.get(_session_key(mgr, "glm-5", "chat-1")))
    assert stored["last_message_id"] == first_retry.message_id

    next_turn = run(mgr.find_session(model="glm-5", messages=round3))
    assert next_turn is not None
    assert next_turn.parent_id == first_retry.message_id


def test_create_session_tracks_current_user_message_id():
    mgr = SessionManager(session_ttl=3600, max_sessions_per_client=5)

    run(
        mgr.create_session(
            auth_token="token_a",
            model="glm-5",
            messages=[{"role": "user", "content": "hello"}],
            chat_id="chat-2",
            message_id="user-msg-42",
        )
    )

    stored = run(mgr._store.get(_session_key(mgr, "glm-5", "chat-2")))
    assert stored["last_message_id"] == "user-msg-42"
