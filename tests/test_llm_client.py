import asyncio
from types import SimpleNamespace

import llm_client as llm_client_module
from llm_client import LLMClient, _web_fetch_tool


def _tool_call(name="web_search", arguments='{"query": "deepseek"}'):
    return SimpleNamespace(
        id="call_123",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
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


def test_run_tool_call_dispatches_web_fetch(monkeypatch):
    client = object.__new__(LLMClient)
    client.web_fetch_max_chars = 12000
    client.web_fetch_hard_max_chars = 100000
    calls = []

    async def fake_web_fetch(url, max_chars=4000):
        calls.append((url, max_chars))
        return '{"text": "page"}'

    monkeypatch.setattr(llm_client_module, "web_fetch", fake_web_fetch)

    payload = asyncio.run(
        client._run_tool_call(
            _tool_call("web_fetch", '{"url": "https://example.com", "max_chars": 900}')
        )
    )

    assert calls == [("https://example.com", 900)]
    assert payload == {
        "role": "tool",
        "tool_call_id": "call_123",
        "name": "web_fetch",
        "content": '{"text": "page"}',
    }


def test_run_tool_call_clamps_web_fetch_to_configured_hard_max(monkeypatch):
    client = object.__new__(LLMClient)
    client.web_fetch_max_chars = 12000
    client.web_fetch_hard_max_chars = 50000
    calls = []

    async def fake_web_fetch(url, max_chars=4000):
        calls.append((url, max_chars))
        return '{"text": "page"}'

    monkeypatch.setattr(llm_client_module, "web_fetch", fake_web_fetch)

    asyncio.run(
        client._run_tool_call(
            _tool_call(
                "web_fetch",
                '{"url": "https://example.com/log", "max_chars": 999999}',
            )
        )
    )

    assert calls == [("https://example.com/log", 50000)]


def test_web_fetch_tool_schema_uses_configured_limits():
    tool = _web_fetch_tool(max_chars=12000, hard_max_chars=100000)

    max_chars_schema = tool["function"]["parameters"]["properties"]["max_chars"]
    assert max_chars_schema["default"] == 12000
    assert max_chars_schema["maximum"] == 100000
