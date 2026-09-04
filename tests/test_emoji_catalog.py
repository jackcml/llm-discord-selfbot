from types import SimpleNamespace

from emoji_catalog import GuildEmojiCatalog
from utils import split_message


class _Emoji:
    def __init__(
        self,
        name: str,
        emoji_id: int,
        *,
        animated: bool = False,
        usable: bool = True,
    ):
        self.name = name
        self.id = emoji_id
        self.animated = animated
        self._usable = usable

    def is_usable(self) -> bool:
        return self._usable

    def __str__(self) -> str:
        prefix = "a" if self.animated else ""
        return f"<{prefix}:{self.name}:{self.id}>"


class _UncachedRestrictedEmoji(_Emoji):
    def is_usable(self) -> bool:
        raise AttributeError("guild member is not cached")


def test_catalog_uses_full_usable_guild_cache_in_deterministic_order():
    guild = SimpleNamespace(
        emojis=[
            _Emoji("zebra", 30),
            _Emoji("Dance", 20, animated=True),
            _Emoji("blocked", 10, usable=False),
            _UncachedRestrictedEmoji("uncached", 50),
            _Emoji("apple", 40),
        ]
    )

    catalog = GuildEmojiCatalog.from_guild(guild)

    assert catalog.aliases == (":apple:", ":Dance:", ":zebra:")
    assert catalog.render(":apple: :Dance: :blocked:") == (
        "<:apple:40> <a:Dance:20> :blocked:"
    )


def test_catalog_leaves_unknown_aliases_and_existing_markup_unchanged():
    catalog = GuildEmojiCatalog(((":party:", "<:party:123>"),))

    assert (
        catalog.render("yes :party: :unknown: <:party:123> <a:party:456>")
        == "yes <:party:123> :unknown: <:party:123> <a:party:456>"
    )


def test_catalog_without_a_guild_is_empty_and_does_not_change_text():
    catalog = GuildEmojiCatalog.from_guild(None)

    assert catalog.aliases == ()
    assert catalog.render("hello :party:") == "hello :party:"


def test_rendered_markup_is_not_broken_across_discord_message_chunks():
    catalog = GuildEmojiCatalog(((":party:", "<:party:123>"),))
    rendered = catalog.render("x" * 1995 + ":party:")

    assert split_message(rendered) == ["x" * 1995, "<:party:123>"]
