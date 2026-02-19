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


async def list_channels(access_token: str) -> list[dict]:
    return await asyncio.to_thread(_list_channels_sync, access_token)


async def post_message(channel_id: str, content: str, access_token: str) -> None:
    await asyncio.to_thread(_post_message_sync, channel_id, content, access_token)
