import logging

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.todoist.com/api/v1"


def _headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


async def get_user_info(access_token: str) -> dict:
    """Return the authenticated Todoist user's identity: {id, name, email}.

    Uses the API v1 /user endpoint (REST v2 has no /user endpoint;
    Sync API v9 /sync returns 410 Gone).
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.todoist.com/api/v1/user",
            headers=_headers(access_token),
        )
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
        data = resp.json()
    # API v1 returns a paginated wrapper: {"results": [...], "next_cursor": ...}
    projects = data.get("results", data) if isinstance(data, dict) else data
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
        data = resp.json()
    # API v1 returns a paginated wrapper: {"results": [...], "next_cursor": ...}
    sections = data.get("results", data) if isinstance(data, dict) else data
    return [
        {
            "id": str(s["id"]),
            "name": s["name"],
            "url": None,
        }
        for s in sections
    ]


TODOIST_FREE_PROJECT_LIMIT = 5
TODOIST_PRO_PROJECT_LIMIT = 300


async def create_project(name: str, access_token: str) -> dict:
    """Create a new Todoist project and return {id, name, url}."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/projects",
            json={"name": name},
            headers=_headers(access_token),
        )
        resp.raise_for_status()
        project = resp.json()
    return {
        "id": str(project["id"]),
        "name": project["name"],
        "url": project.get("url"),
    }


def count_user_projects(projects: list[dict]) -> int:
    """Return the number of non-inbox projects."""
    return sum(1 for p in projects if not p.get("is_inbox_project", False))


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


async def _fetch_tasks(params: dict, access_token: str, max_chars: int) -> str:
    """Shared helper: fetch tasks with given filter params, return title + description lines."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_BASE}/tasks",
            params=params,
            headers=_headers(access_token),
        )
        resp.raise_for_status()
        data = resp.json()
    tasks = data.get("results", data) if isinstance(data, dict) else data

    lines = []
    total = 0
    for task in tasks:
        title = task.get("content", "").strip()
        desc = task.get("description", "").strip()
        line = f"{title}: {desc}" if desc else title
        if line:
            lines.append(line)
            total += len(line)
            if total >= max_chars:
                break

    return "\n".join(lines)[:max_chars]


async def fetch_project_tasks(
    project_id: str, access_token: str, max_chars: int = 50000
) -> str:
    """Return task titles+descriptions from a project joined by newlines."""
    return await _fetch_tasks({"project_id": project_id}, access_token, max_chars)


async def fetch_section_tasks(
    section_id: str, access_token: str, max_chars: int = 50000
) -> str:
    """Return task titles+descriptions from a section joined by newlines."""
    return await _fetch_tasks({"section_id": section_id}, access_token, max_chars)
