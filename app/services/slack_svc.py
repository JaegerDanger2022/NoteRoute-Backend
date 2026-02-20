import asyncio
import logging

from slack_sdk import WebClient

logger = logging.getLogger(__name__)


def _list_channels_sync(access_token: str) -> list[dict]:
    client = WebClient(token=access_token)
    results = []
    cursor = None

    while True:
        kwargs: dict = {"limit": 200, "exclude_archived": True}
        if cursor:
            kwargs["cursor"] = cursor

        resp = client.conversations_list(**kwargs)
        for ch in resp.get("channels", []):
            results.append({
                "id": ch["id"],
                "name": ch["name"],
                "url": None,  # Slack channels don't have a direct URL via API
            })

        meta = resp.get("response_metadata", {})
        cursor = meta.get("next_cursor")
        if not cursor:
            break

    return results


def _post_message_sync(channel_id: str, content: str, access_token: str) -> None:
    client = WebClient(token=access_token)
    client.chat_postMessage(channel=channel_id, text=content)


def _fetch_channel_messages_sync(channel_id: str, access_token: str, max_chars: int = 4000) -> str:
    """Fetch recent messages from a Slack channel as plain text, capped at max_chars."""
    client = WebClient(token=access_token)
    resp = client.conversations_history(channel=channel_id, limit=50)
    lines = []
    for msg in resp.get("messages", []):
        text = msg.get("text", "").strip()
        if text:
            lines.append(text)
        if sum(len(l) for l in lines) >= max_chars:
            break
    return "\n".join(lines)[:max_chars]


async def fetch_channel_messages(channel_id: str, access_token: str, max_chars: int = 4000) -> str:
    return await asyncio.to_thread(_fetch_channel_messages_sync, channel_id, access_token, max_chars)


async def list_channels(access_token: str) -> list[dict]:
    return await asyncio.to_thread(_list_channels_sync, access_token)


async def post_message(channel_id: str, content: str, access_token: str) -> None:
    await asyncio.to_thread(_post_message_sync, channel_id, content, access_token)
