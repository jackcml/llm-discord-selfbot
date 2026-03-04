from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIStatusError


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

    async def reply(self, conversation: list[dict]) -> str | None:
        """Send conversation history and return the response text.

        Returns None on any API error.
        """
        if not conversation:
            return None

        messages = [{"role": "system", "content": self.system_prompt}] + conversation

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
            )
            return response.choices[0].message.content
        except RateLimitError:
            print("[llm] Rate limited, skipping reply")
            return None
        except APIConnectionError:
            print("[llm] Connection error, skipping reply")
            return None
        except APIStatusError as e:
            print(f"[llm] API error {e.status_code}: {e.message}")
            return None

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
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[llm] Error in pick_interesting: {e}")
            return None
