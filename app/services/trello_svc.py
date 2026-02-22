import logging

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.trello.com/1"


def _params(api_key: str, token: str, **extra) -> dict:
    return {"key": api_key, "token": token, **extra}


async def get_user_info(api_key: str, token: str) -> dict:
    """Return the authenticated Trello member's identity: {id, name, email}."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_BASE}/members/me",
            params=_params(api_key, token, fields="id,fullName,email"),
        )
        resp.raise_for_status()
        user = resp.json()
    return {
        "id": user["id"],
        "name": user.get("fullName", "Trello User"),
        "email": user.get("email"),
    }


async def list_boards(api_key: str, token: str) -> list[dict]:
    """Return all open boards the user is a member of: [{id, name, url}]."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_BASE}/members/me/boards",
            params=_params(api_key, token, fields="id,name,url", filter="open"),
        )
        resp.raise_for_status()
        boards = resp.json()
    return [{"id": b["id"], "name": b["name"], "url": b.get("url")} for b in boards]


async def list_lists(board_id: str, api_key: str, token: str) -> list[dict]:
    """Return all open lists on a board: [{id, name, url}]."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_BASE}/boards/{board_id}/lists",
            params=_params(api_key, token, fields="id,name", filter="open"),
        )
        resp.raise_for_status()
        lists = resp.json()
    return [{"id": l["id"], "name": l["name"], "url": None} for l in lists]


def _split_into_items(content: str, max_len: int = 100) -> list[str]:
    """Split content into short checklist items (by newlines then sentences)."""
    raw: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Further split long lines on sentence boundaries
        for sentence in line.split(". "):
            sentence = sentence.strip().rstrip(".")
            if sentence:
                raw.append(sentence[:max_len])
    return raw or [content[:max_len]]


def _format_as_bullets(content: str) -> str:
    """Prefix each non-empty line with '- '."""
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    return "\n".join(f"- {l}" for l in lines) if lines else content


async def _add_checklist(card_id: str, items: list[str], api_key: str, token: str) -> None:
    """Create a checklist on a card and populate it with items."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/cards/{card_id}/checklists",
            params=_params(api_key, token),
            json={"name": "Notes"},
        )
        resp.raise_for_status()
        checklist_id = resp.json()["id"]
        for item in items:
            await client.post(
                f"{_BASE}/checklists/{checklist_id}/checkItems",
                params=_params(api_key, token),
                json={"name": item},
            )


async def create_card(
    list_id: str,
    name: str,
    description: str,
    api_key: str,
    token: str,
    fmt: str = "note",
) -> dict:
    """Create a card in a list and return {id, name, url}.

    fmt: "note" (plain text desc), "bullet" (bullet-prefixed desc), "checklist" (Trello checklist).
    """
    desc = _format_as_bullets(description) if fmt == "bullet" else description
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/cards",
            params=_params(api_key, token),
            json={"idList": list_id, "name": name, "desc": desc if fmt != "checklist" else ""},
        )
        resp.raise_for_status()
        card = resp.json()
    if fmt == "checklist":
        await _add_checklist(card["id"], _split_into_items(description), api_key, token)
    return {"id": card["id"], "name": card["name"], "url": card.get("shortUrl")}


async def list_cards_for_picker(list_id: str, api_key: str, token: str) -> list[dict]:
    """Return cards in a list for UI display: [{id, name}]."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_BASE}/lists/{list_id}/cards",
            params=_params(api_key, token, fields="id,name"),
        )
        resp.raise_for_status()
        cards = resp.json()
    return [{"id": c["id"], "name": c["name"]} for c in cards]


async def append_to_card(
    card_id: str,
    content: str,
    api_key: str,
    token: str,
    fmt: str = "note",
) -> None:
    """Append content to an existing card.

    fmt: "note" (plain text), "bullet" (bullet-prefixed lines), "checklist" (new checklist on card).
    """
    if fmt == "checklist":
        await _add_checklist(card_id, _split_into_items(content), api_key, token)
        return

    formatted = _format_as_bullets(content) if fmt == "bullet" else content
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_BASE}/cards/{card_id}",
            params=_params(api_key, token, fields="desc"),
        )
        resp.raise_for_status()
        existing_desc = resp.json().get("desc", "").strip()

        new_desc = f"{existing_desc}\n\n---\n\n{formatted}" if existing_desc else formatted

        resp = await client.put(
            f"{_BASE}/cards/{card_id}",
            params=_params(api_key, token),
            json={"desc": new_desc},
        )
        resp.raise_for_status()


async def fetch_list_cards(
    list_id: str, api_key: str, token: str, max_chars: int = 50000
) -> str:
    """Return card names + descriptions from a list joined by newlines, capped at max_chars."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_BASE}/lists/{list_id}/cards",
            params=_params(api_key, token, fields="name,desc"),
        )
        resp.raise_for_status()
        cards = resp.json()

    lines = []
    total = 0
    for card in cards:
        line = card.get("name", "").strip()
        desc = card.get("desc", "").strip()
        if desc:
            line = f"{line}: {desc}"
        if line:
            lines.append(line)
            total += len(line)
            if total >= max_chars:
                break

    return "\n".join(lines)[:max_chars]
