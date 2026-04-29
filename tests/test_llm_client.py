from types import SimpleNamespace

from llm_client import LLMClient


def _tool_call():
    return SimpleNamespace(
        id="call_123",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query": "deepseek"}'),
    )


def test_assistant_tool_call_message_preserves_reasoning_content():
    client = object.__new__(LLMClient)
    message = SimpleNamespace(
        content="",
        reasoning_content="Need current information, so I should search.",
        tool_calls=[_tool_call()],
    )

    payload = client._assistant_tool_call_message(message)

    assert payload == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Need current information, so I should search.",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "deepseek"}',
                },
            }
        ],
    }


def test_assistant_tool_call_message_reads_reasoning_content_from_model_extra():
    client = object.__new__(LLMClient)
    message = SimpleNamespace(
        content="",
        model_extra={"reasoning_content": "Preserved extra field."},
        tool_calls=[_tool_call()],
    )

    payload = client._assistant_tool_call_message(message)

    assert payload["reasoning_content"] == "Preserved extra field."
