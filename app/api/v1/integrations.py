from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from app.api.deps import get_current_user
from app.config import settings
from app.core.exceptions import NotFoundError
from app.core.security import encrypt_token
from app.models.integration import Integration, OAuthTokens
from app.models.user import User

router = APIRouter(prefix="/integrations", tags=["integrations"])

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_SLACK_AUTH_URL = "https://slack.com/oauth/v2/authorize"

_GOOGLE_SCOPES = "https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/drive.readonly"
_SLACK_SCOPES = "channels:read,chat:write"


@router.get("/{provider}/connect")
async def connect_integration(
    provider: str,
    current_user: User = Depends(get_current_user),
) -> dict | RedirectResponse:
    """Initiate connection to a provider."""
    state = str(current_user.id)

    if provider == "notion":
        # Internal integration — token is shared, just store it for this user now
        await _store_notion_internal(current_user)
        return {"status": "connected", "provider": "notion"}

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
        return RedirectResponse(url=url)

    elif provider == "slack":
        url = (
            f"{_SLACK_AUTH_URL}"
            f"?client_id={settings.SLACK_CLIENT_ID}"
            f"&scope={_SLACK_SCOPES}"
            f"&redirect_uri={settings.SLACK_REDIRECT_URI}"
            f"&state={state}"
        )
        return RedirectResponse(url=url)

    else:
        raise NotFoundError(f"Unknown provider: {provider}")


@router.get("/{provider}/callback")
async def oauth_callback(provider: str, code: str, state: str) -> dict:
    """Handle OAuth callback — exchange code for tokens and store encrypted."""
    # TODO (Phase 3): implement token exchange + encrypted storage for Google and Slack
    return {"status": "callback_received", "provider": provider, "state": state}


@router.get("")
async def list_integrations(current_user: User = Depends(get_current_user)) -> list[dict]:
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
    integration = await Integration.find_one(
        Integration.user_id == current_user.id,
        Integration.provider == provider,
    )
    if integration:
        integration.is_active = False
        await integration.save()


async def _store_notion_internal(user: User) -> None:
    """Store the shared internal Notion token for this user."""
    token = settings.NOTION_INTEGRATION_TOKEN

    # Fetch workspace info from Notion to populate metadata
    workspace_name = None
    bot_id = "internal"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.notion.com/v1/users/me",
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": "2022-06-28",
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            bot_id = data.get("bot", {}).get("owner", {}).get("workspace_name", "internal") or "internal"
            workspace_name = data.get("bot", {}).get("owner", {}).get("workspace_name")

    existing = await Integration.find_one(
        Integration.user_id == user.id,
        Integration.provider == "notion",
    )

    encrypted = encrypt_token(token)
    if existing:
        existing.tokens.access_token = encrypted
        existing.is_active = True
        existing.workspace_name = workspace_name
        await existing.save()
    else:
        integration = Integration(
            user_id=user.id,
            provider="notion",
            tokens=OAuthTokens(access_token=encrypted),
            provider_user_id=bot_id,
            workspace_name=workspace_name,
        )
        await integration.insert()
