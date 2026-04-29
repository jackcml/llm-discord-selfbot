import traceback

import discord
import yaml
from discord.ext import commands
from dotenv import load_dotenv

from llm_client import LLMClient
from context_manager import ContextManager
from interesting_picker import InterestingPicker
from message_handler import MessageHandler

load_dotenv()

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize bot
bot = commands.Bot(
    command_prefix=config["behavior"]["command_prefix"],
    self_bot=True,
)

# Initialize components
llm = LLMClient(config["llm"])
ctx_manager = ContextManager(config["context"])
handler = MessageHandler(bot, llm, ctx_manager, config)
picker = InterestingPicker(bot, llm, ctx_manager, config)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    ctx_manager.set_bot_user(bot.user.id)
    if config["reply_modes"]["interesting"]["enabled"]:
        picker.start()
        print("[interesting] Background picker started")
    print("Ready.")


@bot.event
async def on_message(message):
    # Feed every message into context
    ctx_manager.add_message(message)
    # Process reply logic
    await handler.handle(message)
    # Process admin commands
    await bot.process_commands(message)


@bot.event
async def on_error(event, *args, **kwargs):
    traceback.print_exc()


# --- Admin commands (only respond to the bot's own user) ---


@bot.command(name="toggle")
async def toggle_mode(ctx: commands.Context[commands.Bot], mode: str):
    """Toggle a reply mode on/off. Usage: >toggle random | >toggle interesting"""
    mode = mode.lower()
    if mode == "random":
        current = config["reply_modes"]["random"]["enabled"]
        config["reply_modes"]["random"]["enabled"] = not current
        state = "enabled" if not current else "disabled"
        await ctx.message.edit(content=f"Random reply mode **{state}**.")
    elif mode == "interesting":
        current = config["reply_modes"]["interesting"]["enabled"]
        config["reply_modes"]["interesting"]["enabled"] = not current
        if not current:
            picker.start()
            await ctx.message.edit(content="Interesting picker **enabled**.")
        else:
            picker.stop()
            await ctx.message.edit(content="Interesting picker **disabled**.")
    elif mode == "mention":
        current = config["reply_modes"]["mention"]["enabled"]
        config["reply_modes"]["mention"]["enabled"] = not current
        state = "enabled" if not current else "disabled"
        await ctx.message.edit(content=f"Mention reply mode **{state}**.")
    elif mode == "dms":
        current = config["behavior"].get("reply_to_dms", True)
        config["behavior"]["reply_to_dms"] = not current
        state = "enabled" if not current else "disabled"
        await ctx.message.edit(content=f"DM replies **{state}**.")
    elif mode == "groupdms":
        current = config["behavior"].get("reply_to_group_dms", False)
        config["behavior"]["reply_to_group_dms"] = not current
        state = "enabled" if not current else "disabled"
        await ctx.message.edit(content=f"Group DM replies **{state}**.")
    elif mode in {"websearch", "web_search"}:
        current = llm.web_search_enabled
        llm.web_search_enabled = not current
        config["llm"].setdefault("web_search", {})["enabled"] = not current
        state = "enabled" if not current else "disabled"
        await ctx.message.edit(content=f"Web search tool **{state}**.")
    else:
        await ctx.message.edit(
            content=(
                f"Unknown mode `{mode}`. Use: random, interesting, mention, "
                "dms, groupdms, websearch"
            )
        )


@bot.command(name="status")
async def show_status(ctx: commands.Context[commands.Bot]):
    """Show current bot status and mode states."""
    rm = config["reply_modes"]
    lines = [
        f"**Model**: `{llm.model}`",
        f"**Mention replies**: {'on' if rm['mention']['enabled'] else 'off'}",
        f"**Random replies**: {'on' if rm['random']['enabled'] else 'off'}"
        + (f" (1 in {rm['random']['one_in']})" if rm["random"]["enabled"] else ""),
        f"**Interesting picker**: {'on' if rm['interesting']['enabled'] else 'off'}"
        + (
            f" (every {rm['interesting']['interval_minutes']}m)"
            if rm["interesting"]["enabled"]
            else ""
        ),
        f"**DM replies**: {'on' if config['behavior'].get('reply_to_dms', True) else 'off'}",
        f"**Group DM replies**: {'on' if config['behavior'].get('reply_to_group_dms', False) else 'off'} (mention only)",
        f"**Web search tool**: {'on' if llm.web_search_enabled else 'off'}",
    ]
    await ctx.message.edit(content="\n".join(lines))


@bot.command(name="model")
async def set_model(ctx: commands.Context[commands.Bot], model_name: str):
    """Change the model at runtime. Usage: >model gpt-4o"""
    llm.model = model_name
    await ctx.message.edit(content=f"Model set to `{model_name}`.")


@bot.command(name="prompt")
async def set_prompt(ctx: commands.Context[commands.Bot], *, text: str):
    """Update the system prompt at runtime. Usage: >prompt You are a helpful assistant."""
    llm.system_prompt = text
    await ctx.message.edit(content=f"System prompt updated ({len(text)} chars).")


@bot.command(name="odds")
async def set_odds(ctx: commands.Context[commands.Bot], n: int):
    """Set random reply odds. Usage: >odds 25 (reply to 1 in 25 messages)"""
    if n < 1:
        await ctx.message.edit(content="Odds must be at least 1.")
        return
    config["reply_modes"]["random"]["one_in"] = n
    await ctx.message.edit(content=f"Random reply odds set to 1 in {n}.")


@bot.command(name="temp")
async def set_temperature(ctx: commands.Context[commands.Bot], value: float):
    """Set LLM temperature. Usage: >temp 0.7"""
    llm.temperature = value
    await ctx.message.edit(content=f"Temperature set to `{value}`.")


if __name__ == "__main__":
    bot.run(config["discord"]["token"])
