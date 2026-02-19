import logging

from notion_client import AsyncClient

logger = logging.getLogger(__name__)


async def list_pages(access_token: str) -> list[dict]:
    """List accessible Notion pages."""
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
    """Append a text paragraph block to an existing Notion page."""
    client = AsyncClient(auth=access_token)
    await client.blocks.children.append(
        block_id=page_id,
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


def _extract_title(page: dict) -> str:
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            rich_text = prop.get("title", [])
            if rich_text:
                return "".join(t.get("plain_text", "") for t in rich_text)
    return "Untitled"
