from fastapi import Request
from pymongo.errors import DuplicateKeyError

from app.config import settings
from app.models.user import User


def is_admin(user: User) -> bool:
    """Return True if the user's Firebase UID is in the admin allowlist."""
    allowed = [u.strip() for u in settings.ADMIN_FIREBASE_UIDS.split(",") if u.strip()]
    return user.firebase_uid in allowed


async def get_current_user(request: Request) -> User:
    """Resolve the authenticated user from request state (set by AuthMiddleware)."""
    firebase_uid: str = request.state.firebase_uid
    firebase_email: str = request.state.firebase_email

    user = await User.find_one(User.firebase_uid == firebase_uid)
    if user is None:
        try:
            user = User(firebase_uid=firebase_uid, email=firebase_email)
            await user.insert()
        except DuplicateKeyError:
            # Race condition: another request inserted first — fetch it
            user = await User.find_one(User.firebase_uid == firebase_uid)
    return user
