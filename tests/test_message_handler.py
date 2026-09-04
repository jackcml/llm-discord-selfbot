import asyncio
from types import SimpleNamespace

import message_handler as message_handler_module
from message_handler import MessageHandler


class _Emoji:
    name = "party"
    id = 123

    @staticmethod
    def is_usable():
        return True

    def __str__(self):
        return "<:party:123>"


class _GuildMessage:
    def __init__(self):
        self.id = 50
        self.guild = SimpleNamespace(emojis=[_Emoji()])
        self.channel = SimpleNamespace(id=10)
        self.author = SimpleNamespace(id=20)
        self.replies = []

    async def reply(self, content, *, mention_author):
        self.replies.append((content, mention_author))
        return SimpleNamespace(id=60)


class _ReplyLLM:
    def __init__(self, response="sounds good :party:"):
        self.response = response
        self.aliases = None

    async def reply(
        self,
        conversation,
        *,
        tool_activity_context,
        custom_emoji_aliases,
    ):
        self.aliases = custom_emoji_aliases
        return self.response


class _ReplyContext:
    channel_max_length = 500

    def __init__(self):
        self.exchange = None

    def get_conversation(self, *args, **kwargs):
        return [{"role": "user", "content": "hello"}]

    def get_entry_by_message_id(self, channel_id, message_id):
        return {"id": message_id, "content": "hello"}

    def record_exchange(self, channel_id, user_id, user_entry, bot_entry):
        self.exchange = (channel_id, user_id, user_entry, bot_entry)


class _DMChannel:
    id = 11

    def __init__(self):
        self.sent = []

    async def send(self, content):
        self.sent.append(content)
        return SimpleNamespace(id=61)


def _handler(
    *, channel_ids=None, user_ids=None, role_ids=None, guilds=None
):
    bot_user = SimpleNamespace(mentioned_in=lambda message: message.mentions_bot)
    if guilds is None:
        guilds = {
            "default": {
                "channel_ids": channel_ids or [],
                "user_ids": user_ids or [],
                "role_ids": role_ids or [],
            }
        }
    config = {
        "reply_modes": {
            "mention": {
                "enabled": True,
                "guilds": guilds,
            }
        }
    }
    return MessageHandler(
        SimpleNamespace(user=bot_user),
        SimpleNamespace(),
        SimpleNamespace(),
        config,
    )


def _message(
    *, guild_id=1, channel_id=10, user_id=20, role_ids=(), mentions_bot=True
):
    return SimpleNamespace(
        guild=SimpleNamespace(id=guild_id),
        channel=SimpleNamespace(id=channel_id),
        author=SimpleNamespace(
            id=user_id,
            roles=[SimpleNamespace(id=role_id) for role_id in role_ids],
        ),
        mentions_bot=mentions_bot,
    )


def test_mention_role_filter_allows_member_with_any_configured_role():
    handler = _handler(role_ids=["100", "200"])

    assert handler._is_mentioned(_message(role_ids=[50, 200])) is True


def test_mention_role_filter_rejects_member_without_configured_role():
    handler = _handler(role_ids=[100, 200])

    assert handler._is_mentioned(_message(role_ids=[50, 60])) is False


def test_empty_mention_role_filter_allows_members_without_roles():
    handler = _handler(role_ids=[])

    assert handler._is_mentioned(_message(role_ids=[])) is True


def test_mention_role_filter_rejects_author_without_roles_attribute():
    handler = _handler(role_ids=[100])
    message = _message()
    del message.author.roles

    assert handler._is_mentioned(message) is False


def test_mention_user_and_role_filters_are_alternative_grants():
    handler = _handler(channel_ids=[10], user_ids=[20], role_ids=[30])

    assert handler._is_mentioned(_message(role_ids=[30])) is True
    assert handler._is_mentioned(_message(role_ids=[31])) is True
    assert handler._is_mentioned(_message(user_id=21, role_ids=[30])) is True
    assert handler._is_mentioned(_message(user_id=21, role_ids=[31])) is False
    assert handler._is_mentioned(_message(channel_id=11, role_ids=[30])) is False


def test_mention_explicit_user_is_allowed_without_roles_attribute():
    handler = _handler(user_ids=[20], role_ids=[30])
    message = _message()
    del message.author.roles

    assert handler._is_mentioned(message) is True


def test_mention_user_filter_still_restricts_when_no_role_filter_is_configured():
    handler = _handler(user_ids=[20], role_ids=[])

    assert handler._is_mentioned(_message(user_id=20)) is True
    assert handler._is_mentioned(_message(user_id=21)) is False


def test_guild_rules_can_allow_all_channels_in_one_guild_and_restrict_another():
    handler = _handler(
        guilds={
            "default": {
                "channel_ids": [10],
                "user_ids": [],
                "role_ids": [999],
            },
            "1": {"channel_ids": [], "role_ids": []},
            "2": {"channel_ids": [20], "role_ids": [200]},
        },
    )

    assert handler._is_mentioned(_message(guild_id=1, channel_id=99)) is True
    assert handler._is_mentioned(
        _message(guild_id=2, channel_id=20, role_ids=[200])
    ) is True
    assert handler._is_mentioned(
        _message(guild_id=2, channel_id=21, role_ids=[200])
    ) is False
    assert handler._is_mentioned(
        _message(guild_id=2, channel_id=20, role_ids=[201])
    ) is False


def test_unlisted_guild_uses_default_rules():
    handler = _handler(
        guilds={
            "default": {
                "channel_ids": [10],
                "user_ids": [],
                "role_ids": [999],
            },
            1: {"channel_ids": [], "role_ids": []},
        }
    )

    assert handler._is_mentioned(_message(guild_id=1, channel_id=99)) is True
    assert handler._is_mentioned(_message(guild_id=3, role_ids=[999])) is True
    assert handler._is_mentioned(_message(guild_id=3, role_ids=[200])) is False
    assert handler._is_mentioned(
        _message(guild_id=3, channel_id=11, role_ids=[999])
    ) is False


def test_guild_rules_can_override_user_grants():
    handler = _handler(
        guilds={
            "default": {"channel_ids": [], "user_ids": [20], "role_ids": []},
            "2": {"user_ids": [21]},
        }
    )

    assert handler._is_mentioned(_message(guild_id=1, user_id=20)) is True
    assert handler._is_mentioned(_message(guild_id=2, user_id=20)) is False
    assert handler._is_mentioned(_message(guild_id=2, user_id=21)) is True


def test_legacy_flat_mention_rules_remain_supported():
    handler = _handler()
    handler.config["reply_modes"]["mention"] = {
        "enabled": True,
        "channel_ids": [10],
        "user_ids": [20],
        "role_ids": [30],
        "role_ids_by_guild": {"2": []},
    }

    assert handler._is_mentioned(_message(guild_id=1, user_id=20)) is True
    assert handler._is_mentioned(_message(guild_id=1, user_id=21)) is False
    assert handler._is_mentioned(_message(guild_id=2, user_id=20)) is True


def test_guild_reply_injects_aliases_renders_markup_and_stores_alias_form():
    llm = _ReplyLLM()
    context = _ReplyContext()
    bot_user = SimpleNamespace(id=99, display_name="Helper")
    handler = MessageHandler(
        SimpleNamespace(user=bot_user),
        llm,
        context,
        {
            "llm": {"vision": False},
            "behavior": {
                "simulated_typing_delay": False,
                "typing_indicator": False,
            },
        },
    )
    message = _GuildMessage()

    asyncio.run(handler._reply_to(message))

    assert llm.aliases == (":party:",)
    assert message.replies == [("sounds good <:party:123>", True)]
    assert context.exchange[3]["content"] == "sounds good :party:"


def test_dm_reply_does_not_inject_or_render_guild_emoji(monkeypatch):
    monkeypatch.setattr(message_handler_module.discord, "DMChannel", _DMChannel)
    llm = _ReplyLLM()
    context = _ReplyContext()
    handler = MessageHandler(
        SimpleNamespace(user=SimpleNamespace(id=99, display_name="Helper")),
        llm,
        context,
        {
            "llm": {"vision": False},
            "behavior": {
                "simulated_typing_delay": False,
                "typing_indicator": False,
            },
        },
    )
    channel = _DMChannel()
    message = SimpleNamespace(
        id=51,
        guild=SimpleNamespace(emojis=[_Emoji()]),
        channel=channel,
        author=SimpleNamespace(id=21),
    )

    asyncio.run(handler._reply_to(message))

    assert llm.aliases == ()
    assert channel.sent == ["sounds good :party:"]
    assert context.exchange is None
