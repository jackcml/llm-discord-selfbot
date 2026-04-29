import asyncio
import random

import discord
from discord.ext import commands

from llm_client import LLMClient
from context_manager import ContextManager
from utils import clean_response, split_message


class MessageHandler:
    def __init__(
        self,
        bot: commands.Bot,
        llm: LLMClient,
        ctx_manager: ContextManager,
        config: dict,
    ):
        self.bot = bot
        self.llm = llm
        self.ctx = ctx_manager
        self.config = config

    async def handle(self, message: discord.Message):
        """Process an incoming message and decide whether to reply."""
        # Never reply to ourselves (prevent infinite loops)
        if message.author.id == self.bot.user.id:
            return

        # Optionally ignore bots
        if self.config["behavior"]["ignore_bots"] and message.author.bot:
            return

        # Skip empty messages
        if not message.content and not message.attachments:
            return

        # Determine channel type
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_group_dm = isinstance(message.channel, discord.GroupChannel)

        if is_dm:
            # 1-on-1 DMs: always reply (if enabled)
            if self.config["behavior"].get("reply_to_dms", True):
                await self._reply_to(message)
            return

        if is_group_dm:
            # Group DMs: only reply when mentioned (if enabled)
            if self.config["behavior"].get("reply_to_group_dms", False):
                if self.bot.user.mentioned_in(message):
                    await self._reply_to(message)
            return

        # Guild channels: check reply modes
        if self._is_mentioned(message):
            await self._reply_to(message)
        elif self._should_random_reply(message.channel.id):
            await self._reply_to(message)

    @staticmethod
    def _id_in_list(target_id: int, id_list: list) -> bool:
        """Check if a Discord snowflake ID is in a config list.

        Handles the case where YAML may parse IDs as strings (quoted)
        or ints (unquoted) by coercing both sides to int.
        """
        return any(int(x) == target_id for x in id_list)

    def _is_mentioned(self, message: discord.Message) -> bool:
        """Check if the bot user is mentioned in the message."""
        mc = self.config["reply_modes"]["mention"]
        if not mc["enabled"]:
            return False
        # Channel filter: empty list = all channels
        allowed_channels = mc.get("channel_ids", [])
        if allowed_channels and not self._id_in_list(
            message.channel.id, allowed_channels
        ):
            return False
        # User filter: empty list = all users
        allowed_users = mc.get("user_ids", [])
        if allowed_users and not self._id_in_list(message.author.id, allowed_users):
            return False
        return self.bot.user.mentioned_in(message)

    def _should_random_reply(self, channel_id: int) -> bool:
        """Roll the dice for random reply mode."""
        rc = self.config["reply_modes"]["random"]
        if not rc["enabled"]:
            return False
        # Channel filter: empty list = all channels
        allowed = rc.get("channel_ids", [])
        if allowed and not self._id_in_list(channel_id, allowed):
            return False
        return random.randint(1, rc["one_in"]) == 1

    def _typing_delay(self, text: str) -> float:
        """Calculate a human-like delay based on response length.

        Simulates reading the message + typing the response.
        Base rate: ~120 WPM typing, with random jitter.
        """
        words = len(text.split())
        # ~120 WPM = 1s per word, but cap it so long messages don't wait forever
        base = min(words * 0.5, 3.0)
        # Add a small "reading" delay + random jitter
        reading = random.uniform(1.0, 3.0)
        jitter = random.uniform(-0.5, 1.5)
        return max(2.0, base + reading + jitter)

    async def _reply_to(self, message: discord.Message):
        """Generate a response and send it."""
        is_dm = isinstance(message.channel, (discord.DMChannel, discord.GroupChannel))

        # In DMs, no need for per-user conversation tracking — just use channel context
        # Scope context up to this message so Claude only sees what came before,
        # not later messages (which would cause it to address multiple conversations)
        conversation = self.ctx.get_conversation(
            message.channel.id,
            target_user_id=None if is_dm else message.author.id,
            up_to_message_id=message.id,
            vision_enabled=self.config["llm"].get("vision", False),
        )
        if not conversation:
            return

        # Generate the response first (not visible to others yet)
        response = await self.llm.reply(conversation)
        if response is None:
            return
        response = clean_response(response)

        # Simulate human typing delay based on response length
        use_delay = self.config["behavior"].get("simulated_typing_delay", False)
        if use_delay:
            delay = self._typing_delay(response)
            if self.config["behavior"]["typing_indicator"]:
                async with message.channel.typing():
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(delay)

        # Split long responses into multiple messages
        chunks = split_message(response)
        sent_message = None
        for i, chunk in enumerate(chunks):
            if i == 0:
                # In DMs, send a plain message; in guilds, use reply threading
                if is_dm:
                    sent_message = await message.channel.send(chunk)
                else:
                    sent_message = await message.reply(chunk, mention_author=True)
            else:
                # Brief pause between chunks
                if use_delay:
                    if self.config["behavior"]["typing_indicator"]:
                        async with message.channel.typing():
                            await asyncio.sleep(self._typing_delay(chunk))
                    else:
                        await asyncio.sleep(self._typing_delay(chunk))
                else:
                    await asyncio.sleep(0.5)
                sent_message = await message.channel.send(chunk)

        # Record this exchange in the conversation tracker so it persists
        # even after the channel buffer scrolls past it (guild channels only —
        # DMs don't need this since there's no noise from other users)
        if sent_message is not None and not is_dm:
            user_entry = self.ctx.get_entry_by_message_id(
                message.channel.id, message.id
            )
            if user_entry:
                bot_entry = {
                    "author": self.bot.user.display_name,
                    "author_id": self.bot.user.id,
                    "content": response[: self.ctx.channel_max_length],
                    "id": sent_message.id,
                    "is_self": True,
                }
                self.ctx.record_exchange(
                    message.channel.id,
                    message.author.id,
                    user_entry,
                    bot_entry,
                )
