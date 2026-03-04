import re

import discord


def clean_message_content(message: discord.Message) -> str:
    """Convert a Discord message into clean plaintext for Claude."""
    content = message.content

    # Resolve user mentions: <@123456> -> @DisplayName
    for user in message.mentions:
        content = content.replace(f"<@{user.id}>", f"@{user.display_name}")
        content = content.replace(f"<@!{user.id}>", f"@{user.display_name}")

    # Resolve channel mentions: <#123456> -> #channel-name
    for channel in message.channel_mentions:
        content = content.replace(f"<#{channel.id}>", f"#{channel.name}")

    # Resolve role mentions: <@&123456> -> @RoleName
    for role in message.role_mentions:
        content = content.replace(f"<@&{role.id}>", f"@{role.name}")

    # Strip custom emoji to just the name: <:emoji:123456> -> :emoji:
    content = re.sub(r"<a?:(\w+):\d+>", r":\1:", content)

    # Note attachments
    if message.attachments:
        attachment_notes = [f"[attachment: {a.filename}]" for a in message.attachments]
        content += " " + " ".join(attachment_notes)

    # Note embeds
    if message.embeds:
        content += f" [{len(message.embeds)} embed(s)]"

    return content.strip()


def extract_image_urls(message: discord.Message) -> list[dict]:
    """Extract image attachment URLs from a Discord message.

    Returns a list of {"url": str, "content_type": str} for image attachments only.
    """
    images = []
    for attachment in message.attachments:
        ct = attachment.content_type or ""
        if ct.startswith("image/"):
            images.append({"url": attachment.url, "content_type": ct})
    return images


def clean_response(text: str) -> str:
    """Collapse consecutive newlines into single newlines for Discord."""
    return re.sub(r"\n{2,}", "\n", text).strip()


def split_message(text: str, limit: int = 2000) -> list[str]:
    """Split text into chunks that fit within Discord's message limit.

    Tries to split on paragraph boundaries, then sentence boundaries,
    then word boundaries as a last resort.
    """
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        # Try to split on a double newline (paragraph boundary)
        split_pos = remaining.rfind("\n\n", 0, limit)

        # Try single newline
        if split_pos == -1 or split_pos < limit // 2:
            split_pos = remaining.rfind("\n", 0, limit)

        # Try sentence boundary (. ! ?)
        if split_pos == -1 or split_pos < limit // 2:
            for sep in (". ", "! ", "? "):
                pos = remaining.rfind(sep, 0, limit)
                if pos > limit // 2:
                    split_pos = pos + len(sep) - 1  # include the punctuation
                    break

        # Try word boundary (space)
        if split_pos == -1 or split_pos < limit // 2:
            split_pos = remaining.rfind(" ", 0, limit)

        # Hard cut as absolute last resort
        if split_pos == -1 or split_pos < limit // 4:
            split_pos = limit

        chunks.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip()

    return chunks
