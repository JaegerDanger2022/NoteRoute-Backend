import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.config import settings
from app.models.global_config import GlobalConfig
from app.models.integration import Integration
from app.models.slot import KnowledgeSlot
from app.models.source import Source
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(request: Request) -> str:
    """FastAPI dependency: raises 403 if the caller's Firebase UID is not in the allowlist."""
    uid: str = request.state.firebase_uid
    allowed = [u.strip() for u in settings.ADMIN_FIREBASE_UIDS.split(",") if u.strip()]
    if uid not in allowed:
        raise HTTPException(status_code=403, detail="Admin access required")
    return uid


# ── Auth check ────────────────────────────────────────────────────────────────

@router.get("/me")
async def admin_me(request: Request) -> dict:
    """Check whether the current user is an admin. Always returns 200 — safe for frontend probe."""
    uid: str = request.state.firebase_uid
    allowed = [u.strip() for u in settings.ADMIN_FIREBASE_UIDS.split(",") if u.strip()]
    return {"is_admin": uid in allowed, "uid": uid, "configured_uids": allowed}


# ── GlobalConfig endpoints ────────────────────────────────────────────────────

class ConfigPatchRequest(BaseModel):
    use_nova: bool | None = None


@router.get("/config")
async def get_config(_: str = Depends(_require_admin)) -> dict:
    config = await GlobalConfig.get_config()
    return {"use_nova": config.use_nova}


@router.patch("/config")
async def patch_config(
    body: ConfigPatchRequest,
    _: str = Depends(_require_admin),
) -> dict:
    config = await GlobalConfig.get_config()
    if body.use_nova is not None:
        config.use_nova = body.use_nova
    await config.save()
    logger.info("GlobalConfig updated: use_nova=%s", config.use_nova)
    return {"use_nova": config.use_nova}


# ── Internal endpoint (no auth — used by LangGraph) ──────────────────────────

@router.get("/internal/config")
async def internal_config() -> dict:
    """No-auth internal endpoint for LangGraph to read GlobalConfig."""
    config = await GlobalConfig.get_config()
    return {"use_nova": config.use_nova}


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def admin_stats(_: str = Depends(_require_admin)) -> dict:
    total_users = await User.count()
    total_slots = await KnowledgeSlot.count()
    total_sources = await Source.count()
    return {
        "total_users": total_users,
        "total_slots": total_slots,
        "total_sources": total_sources,
    }


# ── User list ─────────────────────────────────────────────────────────────────

@router.get("/users")
async def admin_users(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    _: str = Depends(_require_admin),
) -> dict:
    skip = (page - 1) * page_size
    users = await User.find_all().skip(skip).limit(page_size).to_list()
    total = await User.count()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "users": [
            {
                "id": str(u.id),
                "email": u.email,
                "tier": u.tier,
                "slots_count": u.usage.slots_count,
                "sources_count": u.usage.sources_count,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ],
    }
