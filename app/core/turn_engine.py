#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared assistant turn state machine.

This module is protocol-agnostic. It decides whether a turn should commit as
text or tool output and returns actions for serializers to materialize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from app.utils.logger import get_logger

logger = get_logger()

TurnState = Literal["undecided", "text_turn", "tool_turn", "completed"]
TurnActionKind = Literal[
    "emit_text",
    "emit_tool_calls",
    "drop_pending_text",
    "ignore_tool_calls",
    "ignore_text",
]


@dataclass
class TurnEngineConfig:
    has_tools: bool
    strict_tool_turn: bool = True
    pending_text_char_threshold: int = 256
    pending_text_chunk_threshold: int = 3
    debug_label: str = "turn"


@dataclass
class TurnEngineAction:
    kind: TurnActionKind
    text: str = ""
    tool_calls: List[Dict] = field(default_factory=list)
    dropped_chars: int = 0
    reason: str = ""


class TurnEngine:
    """Strict text/tool turn boundary controller."""

    def __init__(self, config: TurnEngineConfig) -> None:
        self.config = config
        self.state: TurnState = "undecided"
        self.pending_text_parts: List[str] = []
        self.pending_text_chunks = 0

    @property
    def pending_text(self) -> str:
        return "".join(self.pending_text_parts)

    def buffer_text(
        self,
        text: str,
        *,
        eager: bool = True,
    ) -> List[TurnEngineAction]:
        if not text:
            return []

        if self.state == "tool_turn":
            logger.debug(
                "[turn-engine][{}] ignore late text after tool_turn: {} chars",
                self.config.debug_label,
                len(text),
            )
            return [
                TurnEngineAction(
                    kind="ignore_text",
                    text=text,
                    reason="late_text_after_tool_turn",
                )
            ]

        if self.state == "text_turn":
            return [
                TurnEngineAction(
                    kind="emit_text",
                    text=text,
                    reason="already_committed_text_turn",
                )
            ]

        self.pending_text_parts.append(text)
        self.pending_text_chunks += 1
        logger.debug(
            "[turn-engine][{}] buffered text while undecided: chunks={}, chars={}",
            self.config.debug_label,
            self.pending_text_chunks,
            len(self.pending_text),
        )

        if eager and self._should_commit_text():
            return self.flush_text(force=True, reason="buffer_threshold")
        return []

    def flush_text(
        self,
        *,
        force: bool = False,
        reason: str = "manual_flush",
    ) -> List[TurnEngineAction]:
        if self.state == "tool_turn":
            dropped_chars = len(self.pending_text)
            self.pending_text_parts = []
            self.pending_text_chunks = 0
            if dropped_chars:
                logger.debug(
                    "[turn-engine][{}] drop pending text while in tool_turn: {} chars",
                    self.config.debug_label,
                    dropped_chars,
                )
                return [
                    TurnEngineAction(
                        kind="drop_pending_text",
                        dropped_chars=dropped_chars,
                        reason="tool_turn_already_committed",
                    )
                ]
            return []

        if not self.pending_text_parts:
            return []

        if not force and not self._should_commit_text():
            return []

        text = self.pending_text
        self.pending_text_parts = []
        self.pending_text_chunks = 0
        self._set_state("text_turn", reason)
        return [
            TurnEngineAction(
                kind="emit_text",
                text=text,
                reason=reason,
            )
        ]

    def commit_tool_calls(
        self,
        tool_calls: List[Dict],
        *,
        reason: str = "tool_calls_detected",
    ) -> List[TurnEngineAction]:
        if not tool_calls:
            return []

        if self.state == "text_turn" and self.config.strict_tool_turn:
            logger.warning(
                "[turn-engine][{}] ignore tool calls after text_turn commit: {}",
                self.config.debug_label,
                len(tool_calls),
            )
            return [
                TurnEngineAction(
                    kind="ignore_tool_calls",
                    tool_calls=tool_calls,
                    reason="late_tool_calls_after_text_turn",
                )
            ]

        actions: List[TurnEngineAction] = []
        dropped_chars = len(self.pending_text)
        if dropped_chars:
            logger.debug(
                "[turn-engine][{}] drop pending text before tool_turn commit: {} chars",
                self.config.debug_label,
                dropped_chars,
            )
            actions.append(
                TurnEngineAction(
                    kind="drop_pending_text",
                    dropped_chars=dropped_chars,
                    reason="tool_turn_committed",
                )
            )
        self.pending_text_parts = []
        self.pending_text_chunks = 0
        self._set_state("tool_turn", reason)
        actions.append(
            TurnEngineAction(
                kind="emit_tool_calls",
                tool_calls=tool_calls,
                reason=reason,
            )
        )
        return actions

    def mark_completed(self) -> None:
        self._set_state("completed", "stream_completed")

    def _should_commit_text(self) -> bool:
        if self.state != "undecided":
            return False

        if not self.pending_text_parts:
            return False

        if not self.config.has_tools:
            return True

        return (
            len(self.pending_text) >= self.config.pending_text_char_threshold
            or self.pending_text_chunks >= self.config.pending_text_chunk_threshold
        )

    def _set_state(self, new_state: TurnState, reason: str) -> None:
        if self.state == new_state:
            return
        logger.debug(
            "[turn-engine][{}] state {} -> {} ({})",
            self.config.debug_label,
            self.state,
            new_state,
            reason,
        )
        self.state = new_state
