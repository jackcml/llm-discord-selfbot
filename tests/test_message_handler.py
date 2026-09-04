from types import SimpleNamespace

from message_handler import MessageHandler


def _handler(
    *, channel_ids=None, user_ids=None, role_ids=None, role_ids_by_guild=None
):
    bot_user = SimpleNamespace(mentioned_in=lambda message: message.mentions_bot)
    config = {
        "reply_modes": {
            "mention": {
                "enabled": True,
                "channel_ids": channel_ids or [],
                "user_ids": user_ids or [],
                "role_ids": role_ids or [],
                "role_ids_by_guild": role_ids_by_guild or {},
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


def test_mention_must_pass_channel_user_and_role_filters():
    handler = _handler(channel_ids=[10], user_ids=[20], role_ids=[30])

    assert handler._is_mentioned(_message(role_ids=[30])) is True
    assert handler._is_mentioned(_message(channel_id=11, role_ids=[30])) is False
    assert handler._is_mentioned(_message(user_id=21, role_ids=[30])) is False
    assert handler._is_mentioned(_message(role_ids=[31])) is False


def test_guild_role_rules_can_allow_everyone_in_one_guild_and_restrict_another():
    handler = _handler(
        role_ids=[999],
        role_ids_by_guild={
            "1": [],
            "2": [200],
        },
    )

    assert handler._is_mentioned(_message(guild_id=1, role_ids=[])) is True
    assert handler._is_mentioned(_message(guild_id=2, role_ids=[200])) is True
    assert handler._is_mentioned(_message(guild_id=2, role_ids=[201])) is False


def test_unlisted_guild_uses_global_role_rule():
    handler = _handler(
        role_ids=[999],
        role_ids_by_guild={1: []},
    )

    assert handler._is_mentioned(_message(guild_id=3, role_ids=[999])) is True
    assert handler._is_mentioned(_message(guild_id=3, role_ids=[200])) is False
