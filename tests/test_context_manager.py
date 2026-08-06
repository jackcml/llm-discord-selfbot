from collections import deque

from context_manager import (
    CONTEXT_CLOSE_TAG,
    CONTEXT_MESSAGE_NAME,
    CONTEXT_OPEN_TAG,
    TARGET_CLOSE_TAG,
    TARGET_MESSAGE_NAME,
    TARGET_OPEN_TAG,
    ContextManager,
)


def _manager() -> ContextManager:
    return ContextManager(
        {
            "channels": {
                "max_messages": 20,
                "max_message_length": 500,
                "max_conversation_turns": 10,
            },
            "dms": {"max_messages": 20, "max_message_length": 500},
        }
    )


def _entry(
    message_id: int,
    author: str,
    content: str,
    *,
    is_self: bool = False,
    images: list[dict] | None = None,
) -> dict:
    return {
        "author": author,
        "author_id": 999 if is_self else message_id,
        "content": content,
        "id": message_id,
        "is_self": is_self,
        "images": images or [],
    }


def test_context_and_exact_target_are_separate_named_blocks():
    manager = _manager()
    manager.buffers[1] = deque(
        [
            _entry(10, "Alice", "can you review my old request?"),
            _entry(20, "Helper", "the old answer", is_self=True),
            _entry(30, "Bob", "unrelated channel chatter"),
            _entry(40, "Carol", "this is the message to answer"),
            _entry(50, "Dave", "arrived after the trigger"),
        ]
    )

    messages = manager.get_conversation(1, target_message_id=40)

    assert [message["role"] for message in messages] == ["user", "user"]
    assert [message["name"] for message in messages] == [
        CONTEXT_MESSAGE_NAME,
        TARGET_MESSAGE_NAME,
    ]
    context, target = messages
    assert context["content"].startswith(CONTEXT_OPEN_TAG)
    assert context["content"].endswith(CONTEXT_CLOSE_TAG)
    assert "can you review my old request?" in context["content"]
    assert "Helper (you): the old answer" in context["content"]
    assert "unrelated channel chatter" in context["content"]
    assert "this is the message to answer" not in context["content"]
    assert "arrived after the trigger" not in context["content"]

    assert target["content"] == (
        f"{TARGET_OPEN_TAG}\n"
        "Carol: this is the message to answer\n"
        f"{TARGET_CLOSE_TAG}"
    )
    assert "old request" not in target["content"]


def test_only_target_is_returned_when_there_is_no_history():
    manager = _manager()
    manager.buffers[1] = deque([_entry(10, "Alice", "hello")])

    messages = manager.get_conversation(1, target_message_id=10)

    assert len(messages) == 1
    assert messages[0]["name"] == TARGET_MESSAGE_NAME
    assert messages[0]["content"] == (
        f"{TARGET_OPEN_TAG}\nAlice: hello\n{TARGET_CLOSE_TAG}"
    )


def test_missing_or_self_target_does_not_generate_a_request():
    manager = _manager()
    manager.buffers[1] = deque(
        [
            _entry(10, "Alice", "hello"),
            _entry(20, "Helper", "my reply", is_self=True),
        ]
    )

    assert manager.get_conversation(1, target_message_id=15) == []
    assert manager.get_conversation(1, target_message_id=20) == []


def test_remembered_conversation_is_merged_deduped_and_sorted_as_context():
    manager = _manager()
    manager.buffers[1] = deque(
        [
            _entry(20, "Ambient", "channel message"),
            _entry(40, "Alice", "new target"),
        ]
    )
    manager.conversations[(1, 7)] = deque(
        [
            _entry(10, "Alice", "remembered question"),
            _entry(20, "Wrong duplicate", "must not replace channel entry"),
            _entry(30, "Helper", "remembered answer", is_self=True),
        ]
    )

    messages = manager.get_conversation(1, target_message_id=40, target_user_id=7)

    context = messages[0]["content"]
    assert context.index("remembered question") < context.index("channel message")
    assert context.index("channel message") < context.index("remembered answer")
    assert "must not replace channel entry" not in context
    assert messages[-1]["name"] == TARGET_MESSAGE_NAME
    assert "new target" in messages[-1]["content"]


def _image_urls(content: list[dict]) -> list[str]:
    return [part["image_url"]["url"] for part in content if part["type"] == "image_url"]


def test_vision_images_stay_with_their_context_or_target_block():
    manager = _manager()
    manager.buffers[1] = deque(
        [
            _entry(
                10,
                "Alice",
                "old image",
                images=[{"url": "https://example.com/context.png"}],
            ),
            _entry(
                20,
                "Bob",
                "what is in this?",
                images=[{"url": "https://example.com/target.png"}],
            ),
        ]
    )

    context, target = manager.get_conversation(
        1, target_message_id=20, vision_enabled=True
    )

    assert isinstance(context["content"], list)
    assert isinstance(target["content"], list)
    assert _image_urls(context["content"]) == ["https://example.com/context.png"]
    assert _image_urls(target["content"]) == ["https://example.com/target.png"]
    assert context["content"][0]["text"].startswith(CONTEXT_OPEN_TAG)
    assert context["content"][-1]["text"].endswith(CONTEXT_CLOSE_TAG)
    assert target["content"][0]["text"].startswith(TARGET_OPEN_TAG)
    assert target["content"][-1]["text"].endswith(TARGET_CLOSE_TAG)

    plain_context, plain_target = manager.get_conversation(
        1, target_message_id=20, vision_enabled=False
    )
    assert isinstance(plain_context["content"], str)
    assert isinstance(plain_target["content"], str)
