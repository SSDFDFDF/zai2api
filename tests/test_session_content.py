from app.core.session.session_content import (
    build_session_body_messages,
    get_precreate_content,
)


def test_get_precreate_content_flattens_multimodal_message_content():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "text", "text": "second"},
            ],
        },
        {"role": "assistant", "content": "done"},
    ]

    assert get_precreate_content(messages) == "first second\n\ndone"


def test_build_session_body_messages_flattens_system_content_lists():
    normalized_messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are concise."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        },
    ]

    body_messages = build_session_body_messages(
        normalized_messages=normalized_messages,
        session_turn_content="hello",
        is_new_session=True,
        inject_system=False,
    )

    assert body_messages == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "hello"},
    ]
