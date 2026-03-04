from collections import deque

import discord

from utils import clean_message_content, extract_image_urls


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
            self.channel_max_conv_turns = context_config.get("max_conversation_turns", 20)
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
        return self.dm_max_length if channel_id in self._dm_channel_ids else self.channel_max_length

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

    def record_exchange(self, channel_id: int, user_id: int, user_entry: dict, bot_entry: dict):
        """Record a direct exchange (user message + bot reply) in the conversation tracker.

        Called by MessageHandler after successfully replying to someone.
        """
        conv = self._get_conv_buffer(channel_id, user_id)
        # Avoid duplicating the user message if it's already the last entry
        if not conv or conv[-1]["id"] != user_entry["id"]:
            conv.append(user_entry)
        conv.append(bot_entry)

    def get_conversation(
        self,
        channel_id: int,
        target_user_id: int | None = None,
        up_to_message_id: int | None = None,
        vision_enabled: bool = False,
    ) -> list[dict]:
        """Convert context into OpenAI-compatible messages format.

        If target_user_id is provided, merges:
          1. The stored conversation history with that user (so the LLM remembers
             past exchanges even if they've scrolled off the channel buffer)
          2. Recent channel messages for general awareness

        If up_to_message_id is provided, only includes messages up to and
        including that message ID. This prevents the LLM from seeing messages
        that arrived after the trigger and trying to address them all.

        If vision_enabled is True and entries have images, emits multi-part
        content arrays with image_url blocks.

        Messages are deduped by ID and sorted chronologically.
        """
        # Collect entries from channel buffer
        buf = self.buffers.get(channel_id)
        channel_entries = list(buf) if buf else []

        # Truncate to only messages up to the trigger message
        if up_to_message_id is not None:
            channel_entries = [e for e in channel_entries if e["id"] <= up_to_message_id]

        if target_user_id is not None:
            # Merge in conversation-specific history
            key = (channel_id, target_user_id)
            conv = self.conversations.get(key)
            if conv:
                conv_entries = list(conv)
                # Also truncate conversation history to the trigger
                if up_to_message_id is not None:
                    conv_entries = [e for e in conv_entries if e["id"] <= up_to_message_id]
                # Dedupe by message ID
                seen_ids = {e["id"] for e in channel_entries}
                for entry in conv_entries:
                    if entry["id"] not in seen_ids:
                        channel_entries.append(entry)
                # Sort by message ID (snowflake IDs are chronological)
                channel_entries.sort(key=lambda e: e["id"])

        if not channel_entries:
            return []

        # Convert to OpenAI-compatible format
        messages = []
        for entry in channel_entries:
            if entry["is_self"]:
                role = "assistant"
                text = entry["content"]
            else:
                role = "user"
                text = f"{entry['author']}: {entry['content']}" if entry["content"] else f"{entry['author']}:"

            images = entry.get("images", [])

            # Build content: multi-part if vision + images, plain string otherwise
            if vision_enabled and images:
                content = [{"type": "text", "text": text}]
                for img in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img["url"]},
                    })
            else:
                content = text

            # Merge with previous if same role
            if messages and messages[-1]["role"] == role:
                prev_content = messages[-1]["content"]
                if isinstance(prev_content, list):
                    # Previous is already multi-part — append text + images
                    if isinstance(content, list):
                        prev_content.extend(content)
                    else:
                        prev_content.append({"type": "text", "text": content})
                else:
                    # Previous is a plain string
                    if isinstance(content, list):
                        # Upgrade previous to multi-part, then append new parts
                        messages[-1]["content"] = [{"type": "text", "text": prev_content}] + content
                    else:
                        messages[-1]["content"] = f"{prev_content}\n{content}"
            else:
                messages.append({"role": role, "content": content})

        # API requires the first message to be "user" role
        if messages and messages[0]["role"] == "assistant":
            messages = messages[1:]

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

    def find_message_id_by_content(
        self, channel_id: int, snippet: str
    ) -> int | None:
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
