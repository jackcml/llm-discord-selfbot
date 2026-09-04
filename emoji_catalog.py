import re
from collections.abc import Sequence

import discord

CUSTOM_EMOJI_ALIAS_RE = re.compile(r"(?<!<)(?<!<a):(?P<name>[A-Za-z0-9_]{2,32}):")


class GuildEmojiCatalog:
    """Usable custom emoji aliases for one Discord guild."""

    def __init__(self, entries: Sequence[tuple[str, str]] = ()):
        self._markup_by_alias = dict(entries)

    @classmethod
    def from_guild(cls, guild: discord.Guild | None) -> "GuildEmojiCatalog":
        """Build a catalog from discord.py-self's gateway-managed guild cache."""
        if guild is None:
            return cls()

        usable_emojis = [emoji for emoji in guild.emojis if cls._is_usable(emoji)]
        usable_emojis.sort(
            key=lambda emoji: (emoji.name.casefold(), emoji.name, emoji.id)
        )

        # Emoji names should be unique within a guild. setdefault still gives us a
        # deterministic result if Discord ever supplies duplicate names.
        entries: dict[str, str] = {}
        for emoji in usable_emojis:
            entries.setdefault(f":{emoji.name}:", str(emoji))
        return cls(tuple(entries.items()))

    @staticmethod
    def _is_usable(emoji: discord.Emoji) -> bool:
        try:
            return emoji.is_usable()
        except AttributeError:
            # Role-restricted emoji need the current guild member in cache. If
            # that member is unexpectedly absent, exclude the emoji rather than
            # failing the whole reply.
            return False

    @property
    def aliases(self) -> tuple[str, ...]:
        """Return aliases in deterministic prompt order."""
        return tuple(self._markup_by_alias)

    def render(self, text: str) -> str:
        """Replace known aliases with sendable Discord custom emoji markup."""

        def replace_alias(match: re.Match[str]) -> str:
            return self._markup_by_alias.get(match.group(0), match.group(0))

        return CUSTOM_EMOJI_ALIAS_RE.sub(replace_alias, text)
