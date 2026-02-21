import logging

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.todoist.com/rest/v2"


def _headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


async def get_user_info(access_token: str) -> dict:
    """Return the authenticated Todoist user's identity: {id, name, email}."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{_BASE}/user", headers=_headers(access_token))
        resp.raise_for_status()
        user = resp.json()
    return {
        "id": str(user.get("id", "unknown")),
        "name": user.get("full_name", "Todoist User"),
        "email": user.get("email"),
    }


async def list_projects(access_token: str) -> list[dict]:
    """Return all user projects as [{id, name, url, is_inbox_project}]."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{_BASE}/projects", headers=_headers(access_token))
        resp.raise_for_status()
        projects = resp.json()
    return [
        {
            "id": str(p["id"]),
            "name": p["name"],
            "url": None,
            "is_inbox_project": p.get("is_inbox_project", False),
        }
        for p in projects
    ]


async def list_sections(project_id: str, access_token: str) -> list[dict]:
    """Return all sections in a project as [{id, name, url}]."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_BASE}/sections",
            params={"project_id": project_id},
            headers=_headers(access_token),
        )
        resp.raise_for_status()
        sections = resp.json()
    return [
        {
            "id": str(s["id"]),
            "name": s["name"],
            "url": None,
        }
        for s in sections
    ]


async def create_task(
    content: str,
    description: str,
    access_token: str,
    project_id: str | None = None,
    section_id: str | None = None,
) -> dict:
    """Create a new task and return {id, name, url}.

    'content' is the task title; 'description' is the markdown body.
    If section_id is provided it takes priority for placement.
    """
    payload: dict = {
        "content": content,
        "description": description,
    }
    if section_id:
        payload["section_id"] = section_id
    if project_id:
        payload["project_id"] = project_id

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/tasks",
            json=payload,
            headers=_headers(access_token),
        )
        resp.raise_for_status()
        task = resp.json()

    return {
        "id": str(task["id"]),
        "name": task["content"],
        "url": task.get("url"),
    }


async def fetch_project_tasks(
    project_id: str, access_token: str, max_chars: int = 50000
) -> str:
    """Return task titles from a project joined by newlines, capped at max_chars."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_BASE}/tasks",
            params={"project_id": project_id},
            headers=_headers(access_token),
        )
        resp.raise_for_status()
        tasks = resp.json()

    lines = []
    total = 0
    for task in tasks:
        line = task.get("content", "").strip()
        if line:
            lines.append(line)
            total += len(line)
            if total >= max_chars:
                break

    return "\n".join(lines)[:max_chars]
