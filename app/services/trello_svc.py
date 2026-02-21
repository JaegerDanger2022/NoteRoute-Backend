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


async def create_card(
    list_id: str,
    name: str,
    description: str,
    api_key: str,
    token: str,
) -> dict:
    """Create a card in a list and return {id, name, url}."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/cards",
            params=_params(api_key, token),
            json={"idList": list_id, "name": name, "desc": description},
        )
        resp.raise_for_status()
        card = resp.json()
    return {"id": card["id"], "name": card["name"], "url": card.get("shortUrl")}


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
