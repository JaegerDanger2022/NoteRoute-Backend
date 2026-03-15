import logging
from datetime import datetime, timezone

from notion_client import AsyncClient

logger = logging.getLogger(__name__)


async def list_pages(access_token: str) -> list[dict]:
    """List all accessible Notion pages flat. Used internally for parent-page lookup."""
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


async def list_all_pages(access_token: str) -> list[dict]:
    """List all accessible Notion pages for the resource picker.

    Top-level (workspace-parent) pages are returned as-is.
    Sub-pages (page-parent) are prefixed with '· ' so the user can
    distinguish hierarchy in a flat list.
    """
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
            parent_type = parent.get("type", "")
            # Notion returns type="page_id" for page-parented pages (not "page")
            raw_parent_id = parent.get("page_id") if parent_type == "page_id" else None
            # Normalise to hyphenated UUID — parent.page_id is sometimes unhyphenated
            page_id = page["id"]
            parent_id = (
                f"{raw_parent_id[0:8]}-{raw_parent_id[8:12]}-{raw_parent_id[12:16]}-{raw_parent_id[16:20]}-{raw_parent_id[20:]}"
                if raw_parent_id and "-" not in raw_parent_id
                else raw_parent_id
            )
            title = _extract_title(page)
            name = title if parent_type == "workspace" else f"· {title}"
            results.append({
                "id": page_id,
                "name": name,
                "title": title,
                "parent_id": parent_id,
                "url": page.get("url"),
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


_RICH_TEXT_LIMIT = 2000  # Notion API hard limit per rich_text element


def _split_content_blocks(content: str) -> list[dict]:
    """Split content into paragraph blocks, each within Notion's 2000-char rich_text limit."""
    blocks = []
    for i in range(0, max(len(content), 1), _RICH_TEXT_LIMIT):
        chunk = content[i:i + _RICH_TEXT_LIMIT]
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": chunk}}]},
        })
    return blocks


async def create_page(
    parent_page_id: str | None,
    title: str,
    content: str,
    access_token: str,
) -> dict:
    """Create a Notion page under parent_page_id, or at workspace root if None."""
    client = AsyncClient(auth=access_token)
    if parent_page_id:
        parent = {"page_id": parent_page_id}
    else:
        parent = {"type": "workspace", "workspace": True}
    page = await client.pages.create(
        parent=parent,
        properties={
            "title": {"title": [{"text": {"content": title}}]}
        },
        children=_split_content_blocks(content),
    )
    return {
        "id": page["id"],
        "name": title,
        "url": page.get("url"),
    }


async def append_block(page_id: str, content: str, access_token: str) -> None:
    """Append a callout block with timestamp to an existing Notion page.

    Splits content into multiple rich_text elements to stay within Notion's
    2000-char per element limit.
    """
    timestamp = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    client = AsyncClient(auth=access_token)
    full_text = f"{timestamp}\n{content}"
    rich_text = [
        {"text": {"content": full_text[i:i + _RICH_TEXT_LIMIT]}}
        for i in range(0, max(len(full_text), 1), _RICH_TEXT_LIMIT)
    ]
    await client.blocks.children.append(
        block_id=page_id,
        children=[
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": rich_text,
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
