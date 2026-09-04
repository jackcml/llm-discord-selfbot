import asyncio

import discord
from discord.ext import commands, tasks

from emoji_catalog import GuildEmojiCatalog
from llm_client import LLMClient
from context_manager import ContextManager
from utils import split_message


class InterestingPicker:
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
        ic = config["reply_modes"]["interesting"]
        self.interval = ic["interval_minutes"]
        self.channel_ids: list[int] = ic["channel_ids"]
        self.lookback = ic["lookback_messages"]
        self._task = self._create_task()

    def _create_task(self):
        @tasks.loop(minutes=self.interval)
        async def interesting_check():
            await self._run_check()

        @interesting_check.before_loop
        async def before_check():
            await self.bot.wait_until_ready()

        return interesting_check

    def start(self):
        if not self._task.is_running():
            self._task.start()

    def stop(self):
        if self._task.is_running():
            self._task.cancel()

    async def _run_check(self):
        """Review channels and pick interesting messages to reply to."""
        channels = self._get_target_channels()

        for channel in channels:
            summary = self.ctx.get_recent_summary(channel.id, limit=self.lookback)
            if not summary:
                continue

            emoji_catalog = GuildEmojiCatalog.from_guild(channel.guild)
            result = await self.llm.pick_interesting(
                summary,
                custom_emoji_aliases=emoji_catalog.aliases,
            )
            if result is None or result.strip() == "SKIP":
                continue

            reply_to_snippet, response_text = self._parse_pick_result(result)
            if not reply_to_snippet or not response_text:
                continue
            response_text = emoji_catalog.render(response_text)

            # Find the original message to reply to
            msg_id = self.ctx.find_message_id_by_content(channel.id, reply_to_snippet)
            if msg_id is None:
                continue

            try:
                target = await channel.fetch_message(msg_id)
                chunks = split_message(response_text)
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        await target.reply(chunk, mention_author=False)
                    else:
                        await channel.send(chunk)
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.5)
            except discord.NotFound:
                continue
            except discord.Forbidden:
                continue
            except discord.HTTPException as e:
                print(f"[interesting] HTTP error replying in #{channel}: {e}")
                continue

            # Small delay between channels
            await asyncio.sleep(1)

    def _get_target_channels(self) -> list[discord.TextChannel]:
        """Get channels to monitor. Empty channel_ids = all text channels."""
        if self.channel_ids:
            return [
                ch
                for cid in self.channel_ids
                if (ch := self.bot.get_channel(cid)) is not None
                and isinstance(ch, discord.TextChannel)
            ]
        channels = []
        for guild in self.bot.guilds:
            for channel in guild.text_channels:
                channels.append(channel)
        return channels

    @staticmethod
    def _parse_pick_result(result: str) -> tuple[str | None, str | None]:
        """Parse the LLM's structured response.

        Expected format:
            REPLY_TO: username: message content
            RESPONSE: the reply text

        Returns (reply_to_snippet, response_text) or (None, None).
        """
        reply_to = None
        response_lines = []
        in_response = False

        for line in result.strip().split("\n"):
            if line.startswith("REPLY_TO:"):
                reply_to = line[len("REPLY_TO:") :].strip()
                in_response = False
            elif line.startswith("RESPONSE:"):
                response_lines.append(line[len("RESPONSE:") :].strip())
                in_response = True
            elif in_response:
                response_lines.append(line)

        response = "\n".join(response_lines).strip() if response_lines else None
        return reply_to, response
