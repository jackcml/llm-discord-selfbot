# LLM Discord Selfbot

A Discord selfbot that interfaces with any OpenAI-compatible LLM API to automatically reply to messages.

> **Note:** This is a selfbot (runs on a user account, not a bot account). Use at your own risk; selfbots violate Discord's Terms of Service.

## Features

- **Mention replies**: responds when @mentioned, with optional channel, user, and role whitelists
- **Random replies**: probabilistically replies to 1 in every X messages
- **Choice replies**: periodically reviews recent messages and picks one to reply to
- **DM support**: always replies in 1-on-1 DMs; mention-only in group DMs (both independently toggleable)
- **Vision support**: understands images when using a vision-capable model
- **Custom emoji**: can use any custom emoji available to the account in the current server
- **Web search tool**: opt-in Chat Completions function tool for current information
- **Conversation memory**: per-user history plus recent channel context, explicitly separated from the one Discord message the model should answer
- **Provider-agnostic**: works with any OpenAI-compatible API (Anthropic, xAI/Grok, OpenAI, OpenRouter, etc.)
- **Human-like typing delay**: configurable simulated typing indicator
- **Runtime commands**: adjust settings on the fly via Discord messages

## Setup

```bash
pip install -r requirements.txt
```

Copy `config.example.yaml` to `config.yaml` and fill in your tokens.

## Usage

```bash
python bot.py
```

For guild replies, the bot reads the current server's gateway-managed emoji
cache, exposes every emoji the account can use to the LLM as a `:name:` alias,
and converts chosen aliases to Discord markup before sending. Emoji changes are
picked up through Discord's normal guild update events; DMs do not receive a
server emoji catalog.

## Commands

All commands use the configured prefix (default `>`). Only the selfbot's own user can trigger them.

| Command               | Description                       |
| --------------------- | --------------------------------- |
| `>toggle mention`     | Enable/disable mention replies    |
| `>toggle random`      | Enable/disable random replies     |
| `>toggle interesting` | Enable/disable interesting picker |
| `>toggle dms`         | Enable/disable DM replies         |
| `>toggle groupdms`    | Enable/disable group DM replies   |
| `>toggle websearch`   | Enable/disable web search tool    |
| `>status`             | Show current configuration        |
| `>model <name>`       | Switch LLM model                  |
| `>prompt <text>`      | Update system prompt              |
| `>temp <X>`           | Set LLM temperature               |
| `>odds <N>`           | Set random reply odds (1 in N)    |

## Configuration

See [`config.example.yaml`](config.example.yaml) for all options including:

- Channel, user, and role ID whitelists for mention replies; channel ID whitelists for random replies
- Separate context limits for DMs vs channels
- Vision toggle, typing delay, bot ignoring
- Web search/fetch tool toggle and result/character limits
- Interesting picker interval and lookback settings

Mention role restrictions can be scoped per server. Each entry in
`role_ids_by_guild` overrides the global `role_ids` fallback for that server,
including an empty list to remove the role restriction:

```yaml
reply_modes:
  mention:
    role_ids: []
    role_ids_by_guild:
      "111111111111111111": [] # respond to everyone in server A
      "222222222222222222":
        - "333333333333333333" # only members with this role in server B
```

Channel restrictions always apply. User and role allowlists are alternative
grants: a member may be allowed either by an explicit `user_ids` entry or by
having any configured role. If both lists are empty, everyone is allowed.
