#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Message content preparation helpers.

Provides functions to flatten message content, concatenate conversation
history, choose the correct turn content, inject system prompts, and
build the final messages list used in session / direct modes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.session.session_manager import SessionResult


# ---------------------------------------------------------------------------
# Low-level content helpers
# ---------------------------------------------------------------------------

def content_to_text(content: Any) -> str:
    """Best-effort flatten message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return " ".join(parts)
    if content is None:
        return ""
    return str(content)


def concat_history(messages: List[Dict[str, Any]]) -> str:
    """Concatenate non-system messages into a single text block.

    Used when a session is lost / new and we need to replay history into
    a single upstream user message.  System messages are skipped because
    the upstream model configuration handles them separately.
    """
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        if content:
            parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


def extract_turn_content(
    raw_messages: List[Dict[str, Any]],
    normalized_messages: List[Dict[str, Any]],
    fallback_user_text: str,
) -> str:
    """Choose the appropriate content for the current turn.

    - last raw role == user  -> send that user text
    - last raw role == tool  -> send the converted ``<tool_result>`` user block
    - otherwise              -> fallback to *fallback_user_text*
    """
    if raw_messages:
        last_raw = raw_messages[-1]
        if isinstance(last_raw, dict):
            last_role = str(last_raw.get("role") or "")
            if last_role == "user":
                text = content_to_text(last_raw.get("content"))
                if text.strip():
                    return text
            elif last_role == "tool":
                # tool messages are converted to user + <tool_result> during
                # normalisation; find the matching block.
                for msg in reversed(normalized_messages):
                    if not isinstance(msg, dict) or msg.get("role") != "user":
                        continue
                    content = content_to_text(msg.get("content"))
                    if content and "<tool_result>" in content:
                        return content
                # Fallback: no tool_result found, use last user text
                for msg in reversed(normalized_messages):
                    if not isinstance(msg, dict) or msg.get("role") != "user":
                        continue
                    content = content_to_text(msg.get("content"))
                    if content.strip():
                        return content
    return fallback_user_text


# ---------------------------------------------------------------------------
# System prompt injection (works for both session and direct modes)
# ---------------------------------------------------------------------------

def inject_system_prompt(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge all system messages into the first user message.

    Extracts every ``role=system`` entry, concatenates their text, and
    prepends it to the first ``role=user`` message's content.  The
    system entries are removed from the returned list.

    If there are no system messages or no user messages, the list is
    returned unchanged (shallow copy).

    This is used in **both** session mode and direct mode when
    ``SESSION_SYSTEM_INJECT`` is enabled.
    """
    # Collect system text
    system_parts: List[str] = []
    for msg in messages:
        if msg.get("role") == "system":
            text = content_to_text(msg.get("content"))
            if text.strip():
                system_parts.append(text.strip())

    if not system_parts:
        return list(messages)

    system_text = "\n\n".join(system_parts)

    # Build new list: skip system, inject into first user message
    result: List[Dict[str, Any]] = []
    injected = False
    for msg in messages:
        if msg.get("role") == "system":
            continue
        if not injected and msg.get("role") == "user":
            user_content = content_to_text(msg.get("content"))
            result.append({
                **msg,
                "content": system_text + "\n\n" + user_content,
            })
            injected = True
        else:
            result.append(msg)

    return result


# ---------------------------------------------------------------------------
# High-level session body builders
# ---------------------------------------------------------------------------

def build_session_body_messages(
    normalized_messages: List[Dict[str, Any]],
    session_turn_content: str,
    is_new_session: bool,
    inject_system: bool = True,
) -> List[Dict[str, Any]]:
    """Build the messages list for session mode body.

    Args:
        normalized_messages: Full preprocessed message list (for extracting
            system prompts and history).
        session_turn_content: The current turn's content (from
            ``extract_turn_content``).
        is_new_session: ``True`` if no existing session was found.
        inject_system: If ``True``, concatenate the system prompt into the
            user message content.  If ``False``, pass the system prompt as
            a separate ``system`` message.

    Returns:
        Messages list for the upstream body.  In session mode this is
        typically 1 message (inject) or 2 messages (pass-through with
        system).
    """
    if not is_new_session:
        # Reuse session: upstream already has the system prompt from turn 1
        return [{"role": "user", "content": session_turn_content}]

    # ---- New session: handle system prompt ----
    system_parts: List[str] = []
    for msg in normalized_messages:
        if msg.get("role") == "system":
            c = msg.get("content", "")
            if isinstance(c, str) and c.strip():
                system_parts.append(c.strip())
    system_text = "\n\n".join(system_parts)

    # Build history content (for multi-message recovery)
    raw_msg_count = sum(1 for m in normalized_messages if m.get("role") != "system")
    if raw_msg_count > 1:
        # Multiple non-system messages: concatenate history
        history_text = concat_history(normalized_messages)
    else:
        history_text = session_turn_content

    if not system_text:
        # No system prompt: just send user content
        return [{"role": "user", "content": history_text}]

    if inject_system:
        # Inject mode: concatenate system + content into single user message
        return [{"role": "user", "content": system_text + "\n\n" + history_text}]
    else:
        # Pass-through mode: system as separate message
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": history_text},
        ]


def resolve_trigger_signal(
    session_result: SessionResult,
    current_signal: Optional[str],
    logger: Any = None,
) -> str:
    """Resolve the Toolify trigger signal for a reused session.

    If the session carries a stored trigger signal, use it (preserving
    consistency across turns).  Otherwise fall back to *current_signal*.

    Args:
        session_result: The matched session result.
        current_signal: Trigger signal generated for the current request.
        logger: Optional logger for debug/warning messages.

    Returns:
        The resolved trigger signal string.
    """
    session_trigger = str(session_result.trigger_signal or "").strip()
    if session_trigger:
        if logger and current_signal and session_trigger != current_signal:
            logger.debug(
                "♻️ 复用会话触发信号: {} -> {}",
                current_signal,
                session_trigger,
            )
        return session_trigger

    if logger:
        logger.warning(
            "⚠️ 复用会话未命中历史 trigger_signal, 当前请求使用新信号: {}",
            current_signal,
        )
    return current_signal or ""


def get_precreate_content(
    session_body_messages: List[Dict[str, Any]],
) -> str:
    """Extract the text content to pass to ``_precreate_chat`` from session body messages.

    Simply concatenates all message contents into a single string.
    """
    parts: List[str] = []
    for msg in session_body_messages:
        c = msg.get("content", "")
        if c:
            parts.append(c)
    return "\n\n".join(parts)
