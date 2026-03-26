#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the session management mechanism.
"""

import asyncio
import time
import pytest

from app.core.session.session_fingerprint import SessionFingerprint, _fast_hash
from app.core.session.session_store import SessionStore
from app.core.session.session_manager import SessionManager


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_messages(*texts, roles=None):
    """Helper: build a simple messages list."""
    if roles is None:
        roles = ["user", "assistant"] * 10
    result = []
    for i, text in enumerate(texts):
        result.append({"role": roles[i % len(roles)], "content": text})
    return result


def run(coro):
    """Run async coroutine synchronously."""
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────────────────────
# SessionFingerprint tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionFingerprint:
    def test_fast_hash_returns_string(self):
        h = _fast_hash("hello world")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_fast_hash_deterministic(self):
        assert _fast_hash("abc") == _fast_hash("abc")

    def test_fast_hash_different_inputs(self):
        assert _fast_hash("abc") != _fast_hash("xyz")

    def test_client_fingerprint_stable(self):
        fp1 = SessionFingerprint.generate_client_fingerprint("token_abc", "GLM-5")
        fp2 = SessionFingerprint.generate_client_fingerprint("token_abc", "GLM-5")
        assert fp1 == fp2

    def test_client_fingerprint_different_model(self):
        fp1 = SessionFingerprint.generate_client_fingerprint("token_abc", "GLM-5")
        fp2 = SessionFingerprint.generate_client_fingerprint("token_abc", "GLM-4.7")
        assert fp1 != fp2

    def test_message_fingerprint_stable(self):
        msg = {"role": "user", "content": "Hello!"}
        assert SessionFingerprint.message_fingerprint(msg) == SessionFingerprint.message_fingerprint(msg)

    def test_message_fingerprint_different(self):
        a = SessionFingerprint.message_fingerprint({"role": "user", "content": "a"})
        b = SessionFingerprint.message_fingerprint({"role": "user", "content": "b"})
        assert a != b

    def test_collect_fingerprints_cap(self):
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
        fps = SessionFingerprint.collect_fingerprints(msgs)
        assert len(fps) == SessionFingerprint.MAX_CACHED_FINGERPRINTS

    def test_is_continuous_single_message(self):
        msgs = make_messages("Hello")
        assert not SessionFingerprint.is_continuous_session(msgs, ["fp1"])

    def test_is_continuous_no_cached(self):
        msgs = make_messages("Hello", "World", "Again")
        assert not SessionFingerprint.is_continuous_session(msgs, [])

    def test_is_continuous_3_messages_match(self):
        msg1 = {"role": "user", "content": "First question"}
        msg2 = {"role": "assistant", "content": "First answer"}
        msg3 = {"role": "user", "content": "Second question"}
        fp1 = SessionFingerprint.message_fingerprint(msg1)
        # Simulate: cache from round 1 has fp of msg1
        cached = [fp1]
        new_msgs = [msg1, msg2, msg3]
        assert SessionFingerprint.is_continuous_session(new_msgs, cached)

    def test_is_continuous_3_messages_no_match(self):
        msg1 = {"role": "user", "content": "First question"}
        msg2 = {"role": "assistant", "content": "First answer"}
        msg3 = {"role": "user", "content": "Second question"}
        cached = ["totally_different_fingerprint"]
        new_msgs = [msg1, msg2, msg3]
        assert not SessionFingerprint.is_continuous_session(new_msgs, cached)

    def test_is_continuous_5_messages_match(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
        ]
        # Cached fingerprints from Q1, A1, Q2 (indices 0,1,2 => maps to -5,-4,-3)
        cached = [SessionFingerprint.message_fingerprint(m) for m in msgs[:3]]
        assert SessionFingerprint.is_continuous_session(msgs, cached)


# ──────────────────────────────────────────────────────────────────────────────
# SessionStore tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionStore:
    def setup_method(self):
        self.store = SessionStore()

    def test_set_get(self):
        run(self.store.set("key1", {"a": 1}, ttl=60))
        result = run(self.store.get("key1"))
        assert result == {"a": 1}

    def test_get_missing(self):
        result = run(self.store.get("nonexistent"))
        assert result is None

    def test_delete(self):
        run(self.store.set("k", {"v": 1}, ttl=60))
        run(self.store.delete("k"))
        assert run(self.store.get("k")) is None

    def test_ttl_expiry(self):
        run(self.store.set("expiring", {"data": "x"}, ttl=1))
        key = "expiring"
        val, _ = self.store._data[key]
        self.store._data[key] = (val, time.monotonic() - 1)  # already expired
        assert run(self.store.get(key)) is None

    def test_keys_prefix(self):
        run(self.store.set("session:fp1:chat1", {}, ttl=60))
        run(self.store.set("session:fp1:chat2", {}, ttl=60))
        run(self.store.set("fp_index:fp1", {}, ttl=60))
        keys = run(self.store.keys("session:"))
        assert "session:fp1:chat1" in keys
        assert "session:fp1:chat2" in keys
        assert "fp_index:fp1" not in keys

    def test_cleanup_expired(self):
        run(self.store.set("a", {}, ttl=60))
        run(self.store.set("b", {}, ttl=60))
        self.store._data["b"] = ({}, time.monotonic() - 1)
        removed = run(self.store.cleanup_expired())
        assert removed == 1
        assert run(self.store.get("a")) is not None


# ──────────────────────────────────────────────────────────────────────────────
# SessionManager tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionManager:
    def setup_method(self):
        self.mgr = SessionManager(session_ttl=3600, max_sessions_per_client=5)

    def test_new_session_created(self):
        msgs = make_messages("Hello")
        result = run(self.mgr.find_or_create_session("token_x", "GLM-5", msgs))
        assert result.is_new is True
        assert result.parent_id is None
        assert result.chat_id
        assert result.message_id

    def test_single_message_always_new(self):
        """Single-message requests should never match (new conversation)."""
        msgs = make_messages("Hello")
        r1 = run(self.mgr.find_or_create_session("token_x", "GLM-5", msgs))
        r2 = run(self.mgr.find_or_create_session("token_x", "GLM-5", msgs))
        assert r1.is_new is True
        assert r2.is_new is True

    def test_continuous_session_reuses_chat_id(self):
        """3-message round should reuse chat_id from round 1."""
        msg1 = {"role": "user", "content": "What is Python?"}
        msgs_round1 = [msg1]
        r1 = run(self.mgr.find_or_create_session("token_a", "GLM-5", msgs_round1))

        msg2 = {"role": "assistant", "content": "Python is a language."}
        msg3 = {"role": "user", "content": "Tell me more."}
        msgs_round2 = [msg1, msg2, msg3]
        r2 = run(self.mgr.find_or_create_session("token_a", "GLM-5", msgs_round2))

        assert r2.is_new is False
        assert r2.chat_id == r1.chat_id
        assert r2.parent_id == r1.message_id

    def test_bound_token_returned_on_reuse(self):
        """Session reuse should return the original creation token as bound_token."""
        msg1 = {"role": "user", "content": "What is Python?"}
        msgs_round1 = [msg1]
        # Round 1: create session with token_a
        r1 = run(self.mgr.find_or_create_session("token_a", "GLM-5", msgs_round1))
        assert r1.is_new is True
        assert r1.bound_token is None  # new session has no bound_token

        msg2 = {"role": "assistant", "content": "Python is a language."}
        msg3 = {"role": "user", "content": "Tell me more."}
        msgs_round2 = [msg1, msg2, msg3]
        # Round 2: pool gives a different token (token_b) — simulates round-robin
        # With model-keyed indexing, this WILL find the session from round 1
        r2 = run(self.mgr.find_or_create_session("token_b", "GLM-5", msgs_round2))

        assert r2.is_new is False
        assert r2.chat_id == r1.chat_id
        # bound_token must be the ORIGINAL token (token_a), not token_b
        assert r2.bound_token == "token_a"

    def test_different_tokens_isolated(self):
        """Different content history creates different sessions, even with same model."""
        msgs_a = [{"role": "user", "content": "Hello from conversation A"}]
        r1 = run(self.mgr.find_or_create_session("token_a", "GLM-5", msgs_a))

        # Totally different content — different conversation, should create new session
        msg2 = {"role": "assistant", "content": "Hi"}
        msg3 = {"role": "user", "content": "Unrelated topic"}
        msgs_b = [{"role": "user", "content": "Completely different start"}, msg2, msg3]
        r2 = run(self.mgr.find_or_create_session("token_b", "GLM-5", msgs_b))

        assert r1.chat_id != r2.chat_id

    def test_different_models_isolated(self):
        """Same token, different models → different sessions."""
        msg1 = {"role": "user", "content": "Hello world"}
        msgs1 = [msg1]
        r1 = run(self.mgr.find_or_create_session("token_x", "GLM-5", msgs1))

        msg2 = {"role": "assistant", "content": "Hi"}
        msg3 = {"role": "user", "content": "Again"}
        msgs_cont = [msg1, msg2, msg3]
        r2 = run(self.mgr.find_or_create_session("token_x", "GLM-4.7", msgs_cont))

        assert r1.chat_id != r2.chat_id

    def test_max_sessions_eviction(self):
        """When max sessions exceeded, oldest is evicted."""
        mgr = SessionManager(session_ttl=3600, max_sessions_per_client=2)
        token = "token_evict"
        for i in range(3):
            msgs = [{"role": "user", "content": f"unique_{i}"}]
            run(mgr.find_or_create_session(token, "GLM-5", msgs))

        stats = run(mgr.get_stats())
        assert stats["total_sessions"] <= 2

    def test_stats(self):
        run(self.mgr.find_or_create_session("tok", "GLM-5", make_messages("hi")))
        stats = run(self.mgr.get_stats())
        assert stats["total_sessions"] >= 1
        assert stats["total_clients"] >= 1
