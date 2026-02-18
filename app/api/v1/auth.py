from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/verify")
async def verify_token(current_user: User = Depends(get_current_user)) -> dict:
    """Verify a Firebase ID token and return the resolved user profile."""
    return {
        "id": str(current_user.id),
        "firebase_uid": current_user.firebase_uid,
        "email": current_user.email,
        "display_name": current_user.display_name,
        "tier": current_user.tier,
    }
