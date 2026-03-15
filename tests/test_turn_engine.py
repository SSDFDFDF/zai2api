from app.core.turn_engine import TurnEngine, TurnEngineConfig


def test_turn_engine_buffers_text_until_threshold_when_tools_enabled():
    engine = TurnEngine(
        TurnEngineConfig(
            has_tools=True,
            pending_text_char_threshold=10,
            pending_text_chunk_threshold=2,
            debug_label="test",
        )
    )

    actions = engine.buffer_text("hello")
    assert actions == []
    assert engine.state == "undecided"

    actions = engine.buffer_text(" world")
    assert len(actions) == 1
    assert actions[0].kind == "emit_text"
    assert actions[0].text == "hello world"
    assert engine.state == "text_turn"


def test_turn_engine_commits_tool_turn_and_drops_pending_text():
    engine = TurnEngine(
        TurnEngineConfig(
            has_tools=True,
            pending_text_char_threshold=100,
            pending_text_chunk_threshold=100,
            debug_label="test",
        )
    )

    engine.buffer_text("draft answer")
    actions = engine.commit_tool_calls(
        [{"id": "call_1", "function": {"name": "ls", "arguments": "{}"}}]
    )

    assert [action.kind for action in actions] == [
        "drop_pending_text",
        "emit_tool_calls",
    ]
    assert actions[0].dropped_chars == len("draft answer")
    assert engine.state == "tool_turn"


def test_turn_engine_can_stage_text_without_eager_commit():
    engine = TurnEngine(
        TurnEngineConfig(
            has_tools=True,
            pending_text_char_threshold=1,
            pending_text_chunk_threshold=1,
            debug_label="test",
        )
    )

    actions = engine.buffer_text("draft", eager=False)
    assert actions == []
    assert engine.state == "undecided"

    actions = engine.commit_tool_calls(
        [{"id": "call_1", "function": {"name": "ls", "arguments": "{}"}}]
    )
    assert actions[0].kind == "drop_pending_text"
    assert actions[1].kind == "emit_tool_calls"


def test_turn_engine_ignores_late_tool_calls_after_text_turn():
    engine = TurnEngine(
        TurnEngineConfig(
            has_tools=False,
            debug_label="test",
        )
    )

    actions = engine.buffer_text("plain answer")
    assert actions[0].kind == "emit_text"
    assert engine.state == "text_turn"

    actions = engine.commit_tool_calls(
        [{"id": "call_1", "function": {"name": "ls", "arguments": "{}"}}]
    )
    assert len(actions) == 1
    assert actions[0].kind == "ignore_tool_calls"
    assert engine.state == "text_turn"


def test_turn_engine_ignores_late_text_after_tool_turn():
    engine = TurnEngine(
        TurnEngineConfig(
            has_tools=True,
            debug_label="test",
        )
    )

    engine.commit_tool_calls(
        [{"id": "call_1", "function": {"name": "ls", "arguments": "{}"}}]
    )
    actions = engine.buffer_text("too late")

    assert len(actions) == 1
    assert actions[0].kind == "ignore_text"
    assert engine.state == "tool_turn"
