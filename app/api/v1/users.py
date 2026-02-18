from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/users", tags=["users"])


class UserUpdateRequest(BaseModel):
    display_name: str | None = None


@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)) -> dict:
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "display_name": current_user.display_name,
        "tier": current_user.tier,
        "limits": {
            "max_sources": current_user.limits.max_sources,
            "max_slots": current_user.limits.max_slots,
            "max_routes_per_month": current_user.limits.max_routes_per_month,
        },
        "usage": {
            "routes_this_month": current_user.usage.routes_this_month,
            "slots_count": current_user.usage.slots_count,
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
