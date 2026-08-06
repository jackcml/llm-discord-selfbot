from collections import deque
from typing import Any

import discord

from utils import clean_message_content, extract_image_urls

CONTEXT_MESSAGE_NAME = "discord_context"
TARGET_MESSAGE_NAME = "discord_target"
CONTEXT_OPEN_TAG = "<recent_discord_context>"
CONTEXT_CLOSE_TAG = "</recent_discord_context>"
TARGET_OPEN_TAG = "<discord_message_to_reply_to>"
TARGET_CLOSE_TAG = "</discord_message_to_reply_to>"


class ContextManager:
    def __init__(self, context_config: dict):
        # Support both new split config and legacy flat config
        if "channels" in context_config:
            ch = context_config["channels"]
            dm = context_config["dms"]
            self.channel_max_messages = ch["max_messages"]
            self.channel_max_length = ch["max_message_length"]
            self.channel_max_conv_turns = ch.get("max_conversation_turns", 20)
            self.dm_max_messages = dm["max_messages"]
            self.dm_max_length = dm["max_message_length"]
        else:
            # Legacy flat config
            self.channel_max_messages = context_config["max_messages_per_channel"]
            self.channel_max_length = context_config["max_message_length"]
            self.channel_max_conv_turns = context_config.get(
                "max_conversation_turns", 20
            )
            self.dm_max_messages = self.channel_max_messages
            self.dm_max_length = self.channel_max_length

        # channel_id -> deque of message records (created lazily with correct maxlen)
        self.buffers: dict[int, deque] = {}
        # (channel_id, user_id) -> deque of message records
        # Tracks direct exchanges: their messages that the bot replied to + bot's replies
        self.conversations: dict[tuple[int, int], deque] = {}
        # Track which channel IDs are DMs so we use the right limits
        self._dm_channel_ids: set[int] = set()
        self._bot_user_id: int | None = None

    def set_bot_user(self, user_id: int):
        """Set the bot's own user ID so we can tag messages correctly."""
        self._bot_user_id = user_id

    def _is_dm(self, channel: discord.abc.Messageable) -> bool:
        return isinstance(channel, (discord.DMChannel, discord.GroupChannel))

    def _get_buffer(self, channel_id: int, is_dm: bool) -> deque:
        """Get or create the buffer for a channel with the correct maxlen."""
        if channel_id not in self.buffers:
            maxlen = self.dm_max_messages if is_dm else self.channel_max_messages
            self.buffers[channel_id] = deque(maxlen=maxlen)
            if is_dm:
                self._dm_channel_ids.add(channel_id)
        return self.buffers[channel_id]

    def _get_conv_buffer(self, channel_id: int, user_id: int) -> deque:
        """Get or create the conversation tracker for a channel+user pair."""
        key = (channel_id, user_id)
        if key not in self.conversations:
            self.conversations[key] = deque(maxlen=self.channel_max_conv_turns)
        return self.conversations[key]

    def _max_length_for(self, channel_id: int) -> int:
        return (
            self.dm_max_length
            if channel_id in self._dm_channel_ids
            else self.channel_max_length
        )

    def add_message(self, message: discord.Message):
        """Record a message into the channel's context buffer."""
        content = clean_message_content(message)
        if not content and not message.attachments:
            return
        is_dm = self._is_dm(message.channel)
        max_len = self.dm_max_length if is_dm else self.channel_max_length
        entry = {
            "author": message.author.display_name,
            "author_id": message.author.id,
            "content": (content or "")[:max_len],
            "id": message.id,
            "is_self": message.author.id == self._bot_user_id,
            "images": extract_image_urls(message),
        }
        buf = self._get_buffer(message.channel.id, is_dm)
        buf.append(entry)

    def record_exchange(
        self, channel_id: int, user_id: int, user_entry: dict, bot_entry: dict
    ):
        """Record a direct exchange (user message + bot reply) in the conversation tracker.

        Called by MessageHandler after successfully replying to someone.
        """
        conv = self._get_conv_buffer(channel_id, user_id)
        # Avoid duplicating the user message if it's already the last entry
        if not conv or conv[-1]["id"] != user_entry["id"]:
            conv.append(user_entry)
        conv.append(bot_entry)

    @staticmethod
    def _format_entry(entry: dict) -> str:
        """Format one buffered Discord message for a labelled transcript block."""
        author = entry["author"]
        if entry["is_self"]:
            author = f"{author} (you)"
        content = entry["content"]
        return f"{author}: {content}" if content else f"{author}:"

    def _build_message_block(
        self,
        entries: list[dict],
        open_tag: str,
        close_tag: str,
        vision_enabled: bool,
    ) -> str | list[dict[str, Any]]:
        """Build a labelled text or multimodal block from Discord entries."""
        has_images = vision_enabled and any(e.get("images") for e in entries)
        if not has_images:
            lines = [open_tag]
            lines.extend(self._format_entry(entry) for entry in entries)
            lines.append(close_tag)
            return "\n".join(lines)

        parts: list[dict[str, Any]] = []
        pending_lines = [open_tag]
        for entry in entries:
            pending_lines.append(self._format_entry(entry))
            images = entry.get("images", [])
            if images:
                parts.append({"type": "text", "text": "\n".join(pending_lines)})
                pending_lines = []
                for image in images:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image["url"]},
                        }
                    )

        pending_lines.append(close_tag)
        parts.append({"type": "text", "text": "\n".join(pending_lines)})
        return parts

    def get_conversation(
        self,
        channel_id: int,
        target_message_id: int,
        target_user_id: int | None = None,
        vision_enabled: bool = False,
    ) -> list[dict]:
        """Build background context plus the one Discord message to answer.

        The exact ``target_message_id`` is emitted alone in a final, named user
        message. All earlier channel and per-user conversation entries are emitted
        in a separate, labelled context message. This prevents ambient chat or old
        requests from being mistaken for additional messages the bot should answer.

        Context and target content remain user-role data so untrusted Discord text is
        not promoted to system instructions and vision-capable APIs can receive image
        parts. ``LLMClient.reply`` supplies the system-level handling contract.

        Entries are deduplicated by ID, sorted chronologically, and truncated at the
        target. If the exact target is unavailable, no request is sent to the model.
        """
        buf = self.buffers.get(channel_id)
        channel_entries = [
            entry
            for entry in (list(buf) if buf else [])
            if entry["id"] <= target_message_id
        ]

        if target_user_id is not None:
            conv = self.conversations.get((channel_id, target_user_id))
            if conv:
                seen_ids = {entry["id"] for entry in channel_entries}
                for entry in conv:
                    if entry["id"] <= target_message_id and entry["id"] not in seen_ids:
                        channel_entries.append(entry)
                        seen_ids.add(entry["id"])

        channel_entries.sort(key=lambda entry: entry["id"])
        target_entry = next(
            (
                entry
                for entry in channel_entries
                if entry["id"] == target_message_id and not entry["is_self"]
            ),
            None,
        )
        if target_entry is None:
            return []

        context_entries = [
            entry for entry in channel_entries if entry["id"] != target_message_id
        ]
        messages: list[dict] = []
        if context_entries:
            messages.append(
                {
                    "role": "user",
                    "name": CONTEXT_MESSAGE_NAME,
                    "content": self._build_message_block(
                        context_entries,
                        CONTEXT_OPEN_TAG,
                        CONTEXT_CLOSE_TAG,
                        vision_enabled,
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "name": TARGET_MESSAGE_NAME,
                "content": self._build_message_block(
                    [target_entry],
                    TARGET_OPEN_TAG,
                    TARGET_CLOSE_TAG,
                    vision_enabled,
                ),
            }
        )
        return messages

    def get_recent_summary(self, channel_id: int, limit: int = 50) -> str:
        """Return plain-text summary of recent messages for the interesting-picker."""
        buf = self.buffers.get(channel_id)
        if not buf:
            return ""
        entries = list(buf)[-limit:]
        lines = []
        for e in entries:
            if e["is_self"]:
                continue
            line = f"{e['author']}: {e['content']}"
            if e.get("images"):
                line += f" [{len(e['images'])} image(s)]"
            lines.append(line)
        return "\n".join(lines)

    def find_message_id_by_content(self, channel_id: int, snippet: str) -> int | None:
        """Look up a message ID from the buffer by matching content substring."""
        buf = self.buffers.get(channel_id)
        if not buf:
            return None
        snippet_lower = snippet.lower().strip()
        # Try exact-ish match first: "username: content"
        for entry in reversed(buf):
            candidate = f"{entry['author']}: {entry['content']}".lower()
            if snippet_lower in candidate or candidate in snippet_lower:
                return entry["id"]
        # Fallback: match content portion only
        for entry in reversed(buf):
            if snippet_lower in entry["content"].lower():
                return entry["id"]
        return None

    def get_entry_by_message_id(self, channel_id: int, message_id: int) -> dict | None:
        """Look up a stored entry by its message ID."""
        buf = self.buffers.get(channel_id)
        if not buf:
            return None
        for entry in reversed(buf):
            if entry["id"] == message_id:
                return entry
        return None
