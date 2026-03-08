# LLM Discord Selfbot

A Discord selfbot that interfaces with any OpenAI-compatible LLM API to automatically reply to messages.

> **Note:** This is a selfbot (runs on a user account, not a bot account). Use at your own risk; selfbots violate Discord's Terms of Service.

## Features

- **Mention replies**: responds when @mentioned, with optional channel and user whitelists
- **Random replies**: probabilistically replies to 1 in every X messages
- **Choice replies**: periodically reviews recent messages and picks one to reply to
- **DM support**: always replies in 1-on-1 DMs; mention-only in group DMs (both independently toggleable)
- **Vision support**: understands images when using a vision-capable model
- **Conversation memory**: per-user conversation tracking that persists through channel noise
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

## Commands

All commands use the configured prefix (default `>`). Only the selfbot's own user can trigger them.

| Command               | Description                       |
| --------------------- | --------------------------------- |
| `>toggle mention`     | Enable/disable mention replies    |
| `>toggle random`      | Enable/disable random replies     |
| `>toggle interesting` | Enable/disable interesting picker |
| `>toggle dms`         | Enable/disable DM replies         |
| `>toggle groupdms`    | Enable/disable group DM replies   |
| `>status`             | Show current configuration        |
| `>model <name>`       | Switch LLM model                  |
| `>prompt <text>`      | Update system prompt              |
| `>temp <X>`           | Set LLM temperature               |
| `>odds <N>`           | Set random reply odds (1 in N)    |

## Configuration

See [`config.example.yaml`](config.example.yaml) for all options including:

- Channel and user ID whitelists for mention/random modes
- Separate context limits for DMs vs channels
- Vision toggle, typing delay, bot ignoring
- Interesting picker interval and lookback settings
