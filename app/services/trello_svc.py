import asyncio
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


async def create_list(board_id: str, name: str, api_key: str, token: str) -> dict:
    """Create a new list on a board and return {id, name}."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/lists",
            params=_params(api_key, token),
            json={"name": name, "idBoard": board_id},
        )
        resp.raise_for_status()
        lst = resp.json()
    return {"id": lst["id"], "name": lst["name"]}


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


async def _add_checklist(card_id: str, items: list[str], api_key: str, token: str, title: str = "Notes") -> None:
    """Create a checklist on a card and populate it with items."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/cards/{card_id}/checklists",
            params=_params(api_key, token),
            json={"name": title or "Notes"},
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


async def append_items_to_checklist(checklist_id: str, items: list[str], api_key: str, token: str) -> None:
    """Add items to an existing Trello checklist."""
    async with httpx.AsyncClient() as client:
        for item in items:
            await client.post(
                f"{_BASE}/checklists/{checklist_id}/checkItems",
                params=_params(api_key, token),
                json={"name": item},
            )


async def append_to_card(
    card_id: str,
    content: str,
    api_key: str,
    token: str,
    fmt: str = "note",
    checklist_title: str | None = None,
    checklist_id: str | None = None,
) -> None:
    """Append content to an existing card.

    fmt: "note" — append to description.
    fmt: "checklist" + checklist_id — append items to that existing checklist.
    fmt: "checklist" + checklist_title — create a new checklist with that title.
    """
    if fmt == "checklist":
        if checklist_id:
            await append_items_to_checklist(checklist_id, _split_into_items(content), api_key, token)
        else:
            await _add_checklist(card_id, _split_into_items(content), api_key, token, title=checklist_title or "Notes")
        return

    formatted = content
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


async def fetch_card_detail(card_id: str, api_key: str, token: str) -> dict:
    """Return full card detail matching Trello's card view.

    Returns: {id, name, desc, url, labels, due, due_complete, members,
              cover_color, comment_count, attachment_count,
              checklists: [{name, items: [{name, complete}]}]}
    """
    async with httpx.AsyncClient() as client:
        card_resp, checklists_resp, members_resp = await asyncio.gather(
            client.get(
                f"{_BASE}/cards/{card_id}",
                params=_params(
                    api_key, token,
                    fields="id,name,desc,shortUrl,labels,due,dueComplete,cover,badges",
                ),
            ),
            client.get(
                f"{_BASE}/cards/{card_id}/checklists",
                params=_params(api_key, token),
            ),
            client.get(
                f"{_BASE}/cards/{card_id}/members",
                params=_params(api_key, token, fields="id,fullName,initials"),
            ),
        )
        card_resp.raise_for_status()
        checklists_resp.raise_for_status()
        # members endpoint may 404 on older tokens — tolerate gracefully
        raw_members = members_resp.json() if members_resp.status_code == 200 else []
        card = card_resp.json()
        raw_checklists = checklists_resp.json()

    checklists = [
        {
            "id": cl["id"],
            "name": cl["name"],
            "items": [
                {"name": item["name"], "complete": item["state"] == "complete"}
                for item in sorted(cl.get("checkItems", []), key=lambda x: x.get("pos", 0))
            ],
        }
        for cl in raw_checklists
    ]

    labels = [
        {"name": lbl.get("name", ""), "color": lbl.get("color", "")}
        for lbl in card.get("labels", [])
    ]
    members = [
        {"name": m.get("fullName", ""), "initials": m.get("initials", "")}
        for m in raw_members
    ]
    badges = card.get("badges", {})
    cover = card.get("cover", {})

    return {
        "id": card["id"],
        "name": card.get("name", ""),
        "desc": card.get("desc", ""),
        "url": card.get("shortUrl"),
        "labels": labels,
        "due": card.get("due"),
        "due_complete": card.get("dueComplete", False),
        "members": members,
        "cover_color": cover.get("color") if cover else None,
        "comment_count": badges.get("comments", 0),
        "attachment_count": badges.get("attachments", 0),
        "checklists": checklists,
    }


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
