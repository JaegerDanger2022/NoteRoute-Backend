import logging
from datetime import datetime, timezone

from notion_client import AsyncClient

logger = logging.getLogger(__name__)


async def list_pages(access_token: str) -> list[dict]:
    """List all accessible Notion pages (flat). Used internally for parent-page lookup."""
    client = AsyncClient(auth=access_token)
    results = []
    cursor = None

    while True:
        kwargs: dict = {
            "filter": {"property": "object", "value": "page"},
            "page_size": 100,
        }
        if cursor:
            kwargs["start_cursor"] = cursor

        resp = await client.search(**kwargs)
        for page in resp.get("results", []):
            results.append({
                "id": page["id"],
                "name": _extract_title(page),
                "url": page.get("url"),
            })

        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    return results


async def list_top_level_pages(access_token: str) -> list[dict]:
    """List only workspace-level (top-level) Notion pages with has_children flag."""
    client = AsyncClient(auth=access_token)
    results = []
    cursor = None

    while True:
        kwargs: dict = {
            "filter": {"property": "object", "value": "page"},
            "page_size": 100,
        }
        if cursor:
            kwargs["start_cursor"] = cursor

        resp = await client.search(**kwargs)
        for page in resp.get("results", []):
            parent = page.get("parent", {})
            if parent.get("type") == "workspace":
                results.append({
                    "id": page["id"],
                    "name": _extract_title(page),
                    "url": page.get("url"),
                    "has_children": page.get("has_children", False),
                })

        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    return results


async def list_child_pages(page_id: str, access_token: str) -> list[dict]:
    """Return direct child pages of a page with has_children flag."""
    client = AsyncClient(auth=access_token)
    results = []
    cursor = None

    while True:
        kwargs: dict = {"block_id": page_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp = await client.blocks.children.list(**kwargs)
        for block in resp.get("results", []):
            if block.get("type") == "child_page":
                results.append({
                    "id": block["id"],
                    "name": block.get("child_page", {}).get("title") or "Untitled",
                    "url": None,
                    "has_children": block.get("has_children", False),
                })
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    return results


async def create_page(
    parent_page_id: str,
    title: str,
    content: str,
    access_token: str,
) -> dict:
    """Create a new child page under parent_page_id with title and text content."""
    client = AsyncClient(auth=access_token)
    page = await client.pages.create(
        parent={"page_id": parent_page_id},
        properties={
            "title": {"title": [{"text": {"content": title}}]}
        },
        children=[
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"text": {"content": content}}]
                },
            }
        ],
    )
    return {
        "id": page["id"],
        "name": title,
        "url": page.get("url"),
    }


async def append_block(page_id: str, content: str, access_token: str) -> None:
    """Append a callout block with timestamp to an existing Notion page."""
    timestamp = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    client = AsyncClient(auth=access_token)
    await client.blocks.children.append(
        block_id=page_id,
        children=[
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [
                        {"text": {"content": f"{timestamp}\n{content}"}}
                    ],
                    "icon": {"emoji": "📝"},
                    "color": "default",
                },
            }
        ],
    )


async def fetch_page_text(
    page_id: str,
    access_token: str,
    max_chars: int = 100000,
    include_subpages: bool = True,
    _depth: int = 0,
) -> str:
    """Fetch plain text from a Notion page's blocks (paginated), capped at max_chars.

    If include_subpages=True, recursively fetches child pages up to 5 levels deep.
    If include_subpages=False, only fetches the direct blocks of this page.
    """
    if _depth > 5:
        return ""
    client = AsyncClient(auth=access_token)
    parts = []
    cursor = None

    while True:
        kwargs: dict = {"block_id": page_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp = await client.blocks.children.list(**kwargs)
        for block in resp.get("results", []):
            if block.get("type") == "child_page":
                if include_subpages:
                    used = sum(len(p) for p in parts)
                    remaining = max_chars - used
                    if remaining > 0:
                        sub = await fetch_page_text(
                            block["id"], access_token, remaining, include_subpages, _depth + 1
                        )
                        if sub:
                            parts.append(sub)
                # else: skip child_page blocks entirely
            else:
                text = _extract_block_text(block)
                if text:
                    parts.append(text)
            if sum(len(p) for p in parts) >= max_chars:
                return "\n".join(parts)[:max_chars]
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    return "\n".join(parts)[:max_chars]


def _extract_block_text(block: dict) -> str:
    btype = block.get("type", "")
    data = block.get(btype, {})
    rich_text = data.get("rich_text", [])
    return "".join(t.get("plain_text", "") for t in rich_text)


def _extract_title(page: dict) -> str:
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            rich_text = prop.get("title", [])
            if rich_text:
                return "".join(t.get("plain_text", "") for t in rich_text)
    return "Untitled"
