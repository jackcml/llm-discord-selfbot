import asyncio
from types import SimpleNamespace

import interesting_picker as interesting_picker_module
from interesting_picker import InterestingPicker


class _Emoji:
    name = "party"
    id = 123

    @staticmethod
    def is_usable():
        return True

    def __str__(self):
        return "<:party:123>"


class _LLM:
    def __init__(self):
        self.aliases = None

    async def pick_interesting(self, summary, *, custom_emoji_aliases):
        self.aliases = custom_emoji_aliases
        return "REPLY_TO: Alice: check :party:\nRESPONSE: that is great :party:"


class _Context:
    def __init__(self):
        self.snippet = None

    def get_recent_summary(self, channel_id, *, limit):
        return "Alice: check :party:"

    def find_message_id_by_content(self, channel_id, snippet):
        self.snippet = snippet
        return 42


class _Target:
    def __init__(self):
        self.replies = []

    async def reply(self, content, *, mention_author):
        self.replies.append((content, mention_author))


class _Channel:
    id = 10

    def __init__(self):
        self.guild = SimpleNamespace(emojis=[_Emoji()])
        self.target = _Target()

    async def fetch_message(self, message_id):
        assert message_id == 42
        return self.target


def test_interesting_picker_resolves_only_response_aliases(monkeypatch):
    llm = _LLM()
    context = _Context()
    channel = _Channel()
    picker = InterestingPicker(
        SimpleNamespace(),
        llm,
        context,
        {
            "reply_modes": {
                "interesting": {
                    "interval_minutes": 15,
                    "channel_ids": [],
                    "lookback_messages": 50,
                }
            }
        },
    )
    monkeypatch.setattr(picker, "_get_target_channels", lambda: [channel])

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(interesting_picker_module.asyncio, "sleep", no_sleep)

    asyncio.run(picker._run_check())

    assert llm.aliases == (":party:",)
    assert context.snippet == "Alice: check :party:"
    assert channel.target.replies == [("that is great <:party:123>", False)]
