# Phase 3: Notion API integration


async def append_block(page_id: str, content: str, access_token: str) -> None:
    """Append a text block to a Notion page."""
    raise NotImplementedError("Notion service not yet implemented (Phase 3)")


async def list_pages(access_token: str) -> list[dict]:
    """List accessible Notion pages for slot selection."""
    raise NotImplementedError("Notion service not yet implemented (Phase 3)")
