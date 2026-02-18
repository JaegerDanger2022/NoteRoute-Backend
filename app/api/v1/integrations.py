from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from app.api.deps import get_current_user
from app.config import settings
from app.models.user import User

router = APIRouter(prefix="/integrations", tags=["integrations"])

# OAuth2 scopes per provider
_NOTION_AUTH_URL = "https://api.notion.com/v1/oauth/authorize"
_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_SLACK_AUTH_URL = "https://slack.com/oauth/v2/authorize"

_GOOGLE_SCOPES = "https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/drive.readonly"
_SLACK_SCOPES = "channels:read,chat:write"


@router.get("/{provider}/connect")
async def connect_integration(
    provider: str,
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    """Redirect the client to the provider's OAuth consent screen."""
    state = str(current_user.id)  # used to look up user in callback

    if provider == "notion":
        url = (
            f"{_NOTION_AUTH_URL}"
            f"?client_id={settings.NOTION_CLIENT_ID}"
            f"&response_type=code"
            f"&owner=user"
            f"&redirect_uri={settings.NOTION_REDIRECT_URI}"
            f"&state={state}"
        )
    elif provider == "google":
        url = (
            f"{_GOOGLE_AUTH_URL}"
            f"?client_id={settings.GOOGLE_CLIENT_ID}"
            f"&response_type=code"
            f"&scope={_GOOGLE_SCOPES}"
            f"&access_type=offline"
            f"&prompt=consent"
            f"&redirect_uri={settings.GOOGLE_REDIRECT_URI}"
            f"&state={state}"
        )
    elif provider == "slack":
        url = (
            f"{_SLACK_AUTH_URL}"
            f"?client_id={settings.SLACK_CLIENT_ID}"
            f"&scope={_SLACK_SCOPES}"
            f"&redirect_uri={settings.SLACK_REDIRECT_URI}"
            f"&state={state}"
        )
    else:
        from app.core.exceptions import NotFoundError
        raise NotFoundError(f"Unknown provider: {provider}")

    return RedirectResponse(url=url)


@router.get("/{provider}/callback")
async def oauth_callback(provider: str, code: str, state: str) -> dict:
    """Handle OAuth callback — exchange code for tokens and store encrypted."""
    # TODO (Phase 3): implement token exchange + encrypted storage per provider
    return {"status": "callback_received", "provider": provider, "state": state}


@router.get("")
async def list_integrations(current_user: User = Depends(get_current_user)) -> list[dict]:
    from app.models.integration import Integration
    integrations = await Integration.find(
        Integration.user_id == current_user.id,
        Integration.is_active == True,
    ).to_list()
    return [
        {
            "id": str(i.id),
            "provider": i.provider,
            "workspace_name": i.workspace_name,
            "provider_email": i.provider_email,
            "connected_at": i.connected_at.isoformat(),
        }
        for i in integrations
    ]


@router.delete("/{provider}", status_code=204)
async def disconnect_integration(
    provider: str,
    current_user: User = Depends(get_current_user),
) -> None:
    from app.models.integration import Integration
    integration = await Integration.find_one(
        Integration.user_id == current_user.id,
        Integration.provider == provider,
    )
    if integration:
        integration.is_active = False
        await integration.save()
