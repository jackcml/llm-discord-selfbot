import json

from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIStatusError

from web_search import web_search


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web for current or factual information. Use this when "
            "the answer may depend on recent events, prices, releases, schedules, or "
            "other information not guaranteed to be in the conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of search results to return, from 1 to 10.",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


class LLMClient:
    def __init__(self, llm_config: dict):
        self.client = AsyncOpenAI(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
        )
        self.model = llm_config["model"]
        self.max_tokens = llm_config["max_tokens"]
        self.temperature = llm_config.get("temperature", 1.0)
        self.system_prompt = llm_config["system_prompt"]
        web_search_config = llm_config.get("web_search", {})
        self.web_search_enabled = web_search_config.get("enabled", False)
        self.web_search_max_results = web_search_config.get("max_results", 5)
        self.web_search_max_rounds = web_search_config.get("max_tool_rounds", 2)

    async def reply(self, conversation: list[dict]) -> str | None:
        """Send conversation history and return the response text.

        Returns None on any API error.
        """
        if not conversation:
            return None

        messages = [{"role": "system", "content": self.system_prompt}] + conversation

        return await self._chat(messages, allow_tools=True)

    async def _chat(self, messages: list[dict], allow_tools: bool = False) -> str | None:
        """Send messages to Chat Completions, optionally handling tool calls."""
        tools = [WEB_SEARCH_TOOL] if allow_tools and self.web_search_enabled else None

        try:
            for _ in range(max(1, self.web_search_max_rounds + 1)):
                kwargs = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": messages,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = await self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                tool_calls = getattr(message, "tool_calls", None)

                if not tool_calls:
                    return message.content

                messages.append(self._assistant_tool_call_message(message))
                for tool_call in tool_calls:
                    messages.append(await self._run_tool_call(tool_call))

            print("[llm] Tool call limit reached, skipping reply")
            return None
        except RateLimitError:
            print("[llm] Rate limited, skipping reply")
            return None
        except APIConnectionError:
            print("[llm] Connection error, skipping reply")
            return None
        except APIStatusError as e:
            print(f"[llm] API error {e.status_code}: {e.message}")
            return None

    def _assistant_tool_call_message(self, message) -> dict:
        tool_calls = []
        for tool_call in message.tool_calls:
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )
        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": tool_calls,
        }
        reasoning_content = self._message_field(message, "reasoning_content")
        if reasoning_content is not None:
            assistant_message["reasoning_content"] = reasoning_content
        return assistant_message

    def _message_field(self, message, field_name: str):
        value = getattr(message, field_name, None)
        if value is not None:
            return value

        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            return model_extra.get(field_name)

        if isinstance(message, dict):
            return message.get(field_name)

        return None

    async def _run_tool_call(self, tool_call) -> dict:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        if name == "web_search":
            query = str(args.get("query", ""))
            requested = args.get("max_results", self.web_search_max_results)
            try:
                max_results = int(requested)
            except (TypeError, ValueError):
                max_results = self.web_search_max_results
            max_results = min(max_results, self.web_search_max_results)
            content = await web_search(query, max_results=max_results)
        else:
            content = json.dumps({"error": f"unknown tool: {name}"})

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": name,
            "content": content,
        }

    async def pick_interesting(self, messages_summary: str) -> str | None:
        """Ask the LLM to pick the most interesting message to reply to.

        Returns the raw response text (caller parses REPLY_TO/RESPONSE/SKIP),
        or None on error.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Here are recent Discord messages. Pick the ONE most interesting "
                    "message to reply to. Respond in this exact format:\n"
                    "REPLY_TO: username: message content\n"
                    "RESPONSE: your reply\n\n"
                    "If nothing is interesting enough to warrant a reply, respond "
                    "with just: SKIP\n\n"
                    f"{messages_summary}"
                ),
            },
        ]
        try:
            return await self._chat(messages, allow_tools=False)
        except Exception as e:
            print(f"[llm] Error in pick_interesting: {e}")
            return None
