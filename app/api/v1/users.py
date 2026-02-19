from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.exceptions import NotFoundError
from app.models.source import Source
from app.models.user import User
from beanie import PydanticObjectId

router = APIRouter(prefix="/users", tags=["users"])


class UserUpdateRequest(BaseModel):
    display_name: str | None = None


class ActiveSourceRequest(BaseModel):
    source_id: str | None = None


@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)) -> dict:
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "display_name": current_user.display_name,
        "tier": current_user.tier,
        "active_source_id": str(current_user.active_source_id) if current_user.active_source_id else None,
        "limits": {
            "max_sources": current_user.limits.max_sources,
            "max_slots": current_user.limits.max_slots,
            "max_routes_per_month": current_user.limits.max_routes_per_month,
        },
        "usage": {
            "routes_this_month": current_user.usage.routes_this_month,
            "slots_count": current_user.usage.slots_count,
            "sources_count": current_user.usage.sources_count,
        },
    }


@router.patch("/me")
async def update_me(
    body: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    if body.display_name is not None:
        current_user.display_name = body.display_name
        await current_user.save()
    return {"id": str(current_user.id), "display_name": current_user.display_name}


@router.patch("/me/active-source")
async def set_active_source(
    body: ActiveSourceRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    if body.source_id is None:
        current_user.active_source_id = None
        await current_user.save()
        return {"active_source_id": None}

    source_id = PydanticObjectId(body.source_id)
    source = await Source.find_one(
        Source.id == source_id,
        Source.user_id == current_user.id,
        Source.is_active == True,
    )
    if not source:
        raise NotFoundError("Source not found")

    current_user.active_source_id = source_id
    await current_user.save()
    return {"active_source_id": str(source_id)}
