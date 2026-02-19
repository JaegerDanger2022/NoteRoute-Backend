from fastapi import Request
from pymongo.errors import DuplicateKeyError

from app.models.user import User


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
