from datetime import datetime, timezone

import httpx
from beanie import PydanticObjectId
from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from app.api.deps import get_current_user
from app.config import settings
from app.core.exceptions import NotFoundError
from app.core.security import encrypt_token
from app.models.integration import Integration, OAuthTokens
from app.models.source import Source
from app.models.user import User

router = APIRouter(prefix="/integrations", tags=["integrations"])

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SLACK_AUTH_URL = "https://slack.com/oauth/v2/authorize"
_SLACK_TOKEN_URL = "https://slack.com/api/oauth.v2.access"

_GOOGLE_SCOPES = (
    "https://www.googleapis.com/auth/documents "
    "https://www.googleapis.com/auth/drive.readonly"
)
_SLACK_SCOPES = "channels:read,chat:write"


@router.get("/{provider}/connect", response_model=None)
async def connect_integration(
    provider: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Initiate connection to a provider.

    For Notion: connects immediately and returns status.
    For Google/Slack: returns the OAuth URL for the client to open in a browser.
    """
    state = str(current_user.id)

    if provider == "notion":
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
        return {"status": "redirect", "provider": "google", "url": url}

    elif provider == "slack":
        url = (
            f"{_SLACK_AUTH_URL}"
            f"?client_id={settings.SLACK_CLIENT_ID}"
            f"&scope={_SLACK_SCOPES}"
            f"&redirect_uri={settings.SLACK_REDIRECT_URI}"
            f"&state={state}"
        )
        return {"status": "redirect", "provider": "slack", "url": url}

    else:
        raise NotFoundError(f"Unknown provider: {provider}")


@router.get("/{provider}/callback", response_model=None)
async def oauth_callback(provider: str, code: str, state: str) -> RedirectResponse:
    """Handle OAuth callback — exchange code for tokens, store encrypted, redirect to app."""
    if provider == "google":
        await _handle_google_callback(code, state)
    elif provider == "slack":
        await _handle_slack_callback(code, state)
    else:
        raise NotFoundError(f"Unknown provider: {provider}")

    return RedirectResponse(
        url=f"noteroute://oauth/success?provider={provider}",
        status_code=302,
    )


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _store_notion_internal(user: User) -> None:
    """Store the shared internal Notion token and upsert a Source for the user."""
    token = settings.NOTION_INTEGRATION_TOKEN
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
            bot_id = (
                data.get("bot", {}).get("owner", {}).get("workspace_name", "internal")
                or "internal"
            )
            workspace_name = data.get("bot", {}).get("owner", {}).get("workspace_name")

    encrypted = encrypt_token(token)
    existing = await Integration.find_one(
        Integration.user_id == user.id,
        Integration.provider == "notion",
    )
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

    await _upsert_source(
        user=user,
        provider="notion",
        name=workspace_name or "Notion Workspace",
        connected_account_id=bot_id,
        connected_account_email=None,
    )


async def _handle_google_callback(code: str, state: str) -> None:
    """Exchange Google auth code for tokens, store, upsert Source."""
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        token_resp.raise_for_status()
        tokens = token_resp.json()

        access_token = tokens["access_token"]
        refresh_token = tokens.get("refresh_token")

        profile_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        profile = profile_resp.json() if profile_resp.status_code == 200 else {}

    provider_user_id = profile.get("id", "unknown")
    provider_email = profile.get("email")
    display_name = profile.get("name", "Google Account")

    user_id = PydanticObjectId(state)
    user = await User.get(user_id)
    if not user:
        return

    existing = await Integration.find_one(
        Integration.user_id == user_id,
        Integration.provider == "google",
    )
    encrypted_access = encrypt_token(access_token)
    encrypted_refresh = encrypt_token(refresh_token) if refresh_token else None

    if existing:
        existing.tokens.access_token = encrypted_access
        if encrypted_refresh:
            existing.tokens.refresh_token = encrypted_refresh
        existing.provider_email = provider_email
        existing.is_active = True
        await existing.save()
    else:
        await Integration(
            user_id=user_id,
            provider="google",
            tokens=OAuthTokens(
                access_token=encrypted_access,
                refresh_token=encrypted_refresh,
            ),
            provider_user_id=provider_user_id,
            provider_email=provider_email,
        ).insert()

    await _upsert_source(
        user=user,
        provider="google",
        name=display_name,
        connected_account_id=provider_user_id,
        connected_account_email=provider_email,
    )


async def _handle_slack_callback(code: str, state: str) -> None:
    """Exchange Slack auth code for tokens, store, upsert Source."""
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            _SLACK_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.SLACK_CLIENT_ID,
                "client_secret": settings.SLACK_CLIENT_SECRET,
                "redirect_uri": settings.SLACK_REDIRECT_URI,
            },
        )
        token_resp.raise_for_status()
        data = token_resp.json()

    if not data.get("ok"):
        raise ValueError(f"Slack token exchange failed: {data.get('error')}")

    access_token = data["access_token"]
    team = data.get("team", {})
    workspace_name = team.get("name", "Slack Workspace")
    workspace_id = team.get("id", "unknown")

    user_id = PydanticObjectId(state)
    user = await User.get(user_id)
    if not user:
        return

    existing = await Integration.find_one(
        Integration.user_id == user_id,
        Integration.provider == "slack",
    )
    encrypted = encrypt_token(access_token)

    if existing:
        existing.tokens.access_token = encrypted
        existing.workspace_name = workspace_name
        existing.is_active = True
        await existing.save()
    else:
        await Integration(
            user_id=user_id,
            provider="slack",
            tokens=OAuthTokens(access_token=encrypted),
            provider_user_id=workspace_id,
            workspace_name=workspace_name,
        ).insert()

    await _upsert_source(
        user=user,
        provider="slack",
        name=workspace_name,
        connected_account_id=workspace_id,
        connected_account_email=None,
    )


async def _upsert_source(
    user: User,
    provider: str,
    name: str,
    connected_account_id: str,
    connected_account_email: str | None,
) -> None:
    """Create or reactivate a Source document for the user+provider pair."""
    existing = await Source.find_one(
        Source.user_id == user.id,
        Source.provider == provider,
    )
    now = datetime.now(timezone.utc)

    if existing:
        existing.name = name
        existing.connected_account_id = connected_account_id
        existing.connected_account_email = connected_account_email
        existing.is_active = True
        existing.updated_at = now
        await existing.save()
    else:
        await Source(
            user_id=user.id,
            provider=provider,
            name=name,
            connected_account_id=connected_account_id,
            connected_account_email=connected_account_email,
        ).insert()
        user.usage.sources_count += 1
        await user.save()
