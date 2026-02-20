import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.core.exceptions import NotFoundError
from app.core.security import decrypt_token, encrypt_token
from app.models.integration import Integration
from app.models.user import User
from app.services import gdocs_svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gdocs", tags=["gdocs"])


async def _get_google_access_token(user: User) -> str:
    """Return a valid Google access token for the user, refreshing if needed."""
    integration = await Integration.find_one(
        Integration.user_id == user.id,
        Integration.provider == "google",
        Integration.is_active == True,
    )
    if not integration:
        raise NotFoundError("No active Google integration")

    if not integration.tokens.refresh_token:
        return decrypt_token(integration.tokens.access_token)

    needs_refresh = True
    if integration.tokens.expires_at:
        expires = integration.tokens.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        needs_refresh = (expires - datetime.now(timezone.utc)).total_seconds() < 60

    if needs_refresh:
        refresh_token = decrypt_token(integration.tokens.refresh_token)
        new_access, expires_at = await gdocs_svc.refresh_access_token(refresh_token)
        integration.tokens.access_token = encrypt_token(new_access)
        integration.tokens.expires_at = expires_at
        await integration.save()
        return new_access

    return decrypt_token(integration.tokens.access_token)


@router.get("/{document_id}/tabs")
async def get_document_tabs(
    document_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return the list of tabs for a Google Doc. Used by the frontend tab picker."""
    access_token = await _get_google_access_token(current_user)
    tabs = await gdocs_svc.list_tabs(document_id, access_token)
    return {"tabs": tabs}
