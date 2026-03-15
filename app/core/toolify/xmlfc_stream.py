#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""XMLFC streaming detection state machine."""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

from app.utils.logger import get_logger
from .xmlfc_codec import parse_function_calls_xml
from .xmlfc_structure import _THINKING_TAG_TOKEN_RE, _match_known_tag_token

logger = get_logger()

_FALLBACK_TRIGGER_RE = re.compile(r"<Function_[A-Za-z0-9]{4}_Start/>")
_BARE_FUNCTION_CALLS_OPEN_RE = re.compile(
    r"<\s*function[\s_-]*calls\s*>",
    re.IGNORECASE,
)


class DetectorState(str, Enum):
    DETECTING = "detecting"
    TOOL_CANDIDATE = "tool_candidate"
    TOOL_PARSING = "tool_parsing"


class StreamingFunctionCallDetector:
    """流式工具调用检测器。"""

    def __init__(self, trigger_signal: str):
        self.trigger_signal = trigger_signal
        self.reset()

    def reset(self):
        self.content_buffer = ""
        self.state = DetectorState.DETECTING
        self.in_think_block = False
        self.think_depth = 0
        self.think_stack: List[str] = []
        self.signal = self.trigger_signal
        self.signal_len = len(self.signal)
        self.bare_open_len = len("<function_calls>")

    def process_chunk(self, delta_content: str) -> Tuple[bool, str]:
        """处理流式内容片段。"""
        if not delta_content:
            return False, ""

        self.content_buffer += delta_content

        if self.state in (DetectorState.TOOL_PARSING, DetectorState.TOOL_CANDIDATE):
            return False, ""

        buf = self.content_buffer
        buf_len = len(buf)
        parts: list[str] = []
        i = 0

        while i < buf_len:
            skip_chars = self._update_think_state(i)
            if skip_chars > 0:
                end = min(i + skip_chars, buf_len)
                parts.append(buf[i:end])
                i = end
                continue

            if not self.in_think_block:
                if i + self.signal_len <= buf_len and buf[i : i + self.signal_len] == self.signal:
                    logger.debug("🔧 检测到触发信号 (非思维块内), 切换到工具解析模式")
                    self.state = DetectorState.TOOL_PARSING
                    self.content_buffer = buf[i:]
                    return True, "".join(parts)

                fallback = _FALLBACK_TRIGGER_RE.match(buf, i)
                if fallback:
                    detected_signal = fallback.group(0)
                    if detected_signal != self.signal:
                        logger.warning(
                            "⚠️ 检测到 trigger_signal 漂移, 使用流内信号: {} -> {}",
                            self.signal,
                            detected_signal,
                        )
                        self.trigger_signal = detected_signal
                        self.signal = detected_signal
                        self.signal_len = len(detected_signal)
                    logger.debug("🔧 检测到触发信号(兜底匹配), 切换到工具解析模式")
                    self.state = DetectorState.TOOL_PARSING
                    self.content_buffer = buf[i:]
                    return True, "".join(parts)

                bare_match = _BARE_FUNCTION_CALLS_OPEN_RE.match(buf, i)
                if bare_match and self._line_prefix_is_whitespace(buf, i):
                    logger.debug("🔧 检测到 bare <function_calls>, 进入候选解析模式")
                    self.state = DetectorState.TOOL_CANDIDATE
                    self.content_buffer = buf[i:]
                    return True, "".join(parts)

            remaining_len = buf_len - i
            lookahead_len = max(self.signal_len, self.bare_open_len, 32)
            if remaining_len < lookahead_len:
                break

            search_start = i + 1
            next_pos = buf.find("<", search_start)

            if next_pos == -1:
                safe_end = buf_len - lookahead_len + 1
                if safe_end > i:
                    parts.append(buf[i:safe_end])
                    i = safe_end
                else:
                    break
            else:
                safe_end = min(next_pos, buf_len - lookahead_len + 1)
                if safe_end > i:
                    parts.append(buf[i:safe_end])
                    i = safe_end
                else:
                    parts.append(buf[i])
                    i += 1

        self.content_buffer = buf[i:]
        return False, "".join(parts)

    def _update_think_state(self, pos: int) -> int:
        buf = self.content_buffer
        token = _match_known_tag_token(buf, pos, _THINKING_TAG_TOKEN_RE)
        if token is None:
            return 0

        if token.is_close:
            while self.think_stack and self.think_stack[-1] != token.name:
                self.think_stack.pop()
            if self.think_stack and self.think_stack[-1] == token.name:
                self.think_stack.pop()
        else:
            self.think_stack.append(token.name)

        self.think_depth = len(self.think_stack)
        self.in_think_block = self.think_depth > 0
        return token.end - token.start

    @staticmethod
    def _line_prefix_is_whitespace(buf: str, pos: int) -> bool:
        """检查当前位置是否位于新行起始处。"""
        line_start = buf.rfind("\n", 0, pos)
        if line_start == -1:
            prefix = buf[:pos]
        else:
            prefix = buf[line_start + 1 : pos]
        return not prefix.strip()

    def flush(self) -> str:
        """流结束时刷出被 look-ahead 保留的剩余内容。"""
        remaining = self.content_buffer
        self.content_buffer = ""
        return remaining

    def reject_candidate(self) -> str:
        """候选 bare XML 解析失败时，将缓冲区恢复为普通文本。"""
        remaining = self.content_buffer
        self.reset()
        return remaining

    def finalize(self) -> Optional[List[Dict[str, Any]]]:
        """流结束时的最终处理。"""
        if self.state == DetectorState.TOOL_PARSING:
            return parse_function_calls_xml(self.content_buffer, self.trigger_signal)
        if self.state == DetectorState.TOOL_CANDIDATE:
            return parse_function_calls_xml(
                self.content_buffer,
                self.trigger_signal,
                allow_bare=True,
                bare_tail_only=True,
            )
        return None
