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

# Phase 1: read-only — shown at connect time (no scary permissions)
_GOOGLE_CONNECT_SCOPES = (
    "https://www.googleapis.com/auth/drive.readonly "
    "https://www.googleapis.com/auth/userinfo.email"
)
# Phase 2: write — requested only when user first delivers a note
_GOOGLE_WRITE_SCOPE = "https://www.googleapis.com/auth/documents"

_SLACK_SCOPES = "channels:read,chat:write"


@router.get("/{provider}/connect", response_model=None)
async def connect_integration(
    provider: str,
    current_user: User = Depends(get_current_user),
    platform: str = "mobile",
) -> dict:
    """Initiate connection to a provider.

    For Notion: connects immediately and returns status.
    For Google/Slack: returns the OAuth URL for the client to open in a browser.
    Google connect requests read-only scopes only; write access is requested
    incrementally before the first delivery.

    Pass ?platform=web from the Next.js frontend so the OAuth callback
    redirects to the web app instead of the mobile deep link.
    """
    # Append |web to state so the callback knows which platform to redirect to
    state = str(current_user.id)
    if platform == "web":
        state += "|web"

    if provider == "notion":
        await _store_notion_internal(current_user)
        return {"status": "connected", "provider": "notion"}

    elif provider == "google":
        url = (
            f"{_GOOGLE_AUTH_URL}"
            f"?client_id={settings.GOOGLE_CLIENT_ID}"
            f"&response_type=code"
            f"&scope={_GOOGLE_CONNECT_SCOPES}"
            f"&access_type=offline"
            f"&include_granted_scopes=true"
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


@router.get("/google/upgrade-write")
async def upgrade_google_write(
    current_user: User = Depends(get_current_user),
    platform: str = "mobile",
) -> dict:
    """Return an OAuth URL that adds the documents (write) scope incrementally.

    Google merges this with already-granted scopes so the user only sees
    the new permission on the consent screen.

    Pass ?platform=web from the Next.js frontend so the callback redirects
    back to the web app instead of the mobile deep link.
    """
    integration = await Integration.find_one(
        Integration.user_id == current_user.id,
        Integration.provider == "google",
        Integration.is_active == True,
    )
    # Encode platform into state so callback knows where to redirect
    state = str(current_user.id)
    if platform == "web":
        state += "|web"
    state += "__upgrade"

    login_hint = integration.provider_email if integration else None

    url = (
        f"{_GOOGLE_AUTH_URL}"
        f"?client_id={settings.GOOGLE_CLIENT_ID}"
        f"&response_type=code"
        f"&scope={_GOOGLE_WRITE_SCOPE}"
        f"&access_type=offline"
        f"&include_granted_scopes=true"
        f"&redirect_uri={settings.GOOGLE_REDIRECT_URI}"
        f"&state={state}"
    )
    if login_hint:
        url += f"&login_hint={login_hint}"

    return {"status": "redirect", "url": url}


@router.get("/google/has-write")
async def google_has_write(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Check whether the user has granted the documents (write) scope."""
    integration = await Integration.find_one(
        Integration.user_id == current_user.id,
        Integration.provider == "google",
        Integration.is_active == True,
    )
    if not integration:
        return {"has_write": False}
    has_write = _GOOGLE_WRITE_SCOPE in integration.tokens.scopes
    return {"has_write": has_write}


@router.get("/{provider}/callback", response_model=None)
async def oauth_callback(provider: str, code: str, state: str) -> RedirectResponse:
    """Handle OAuth callback — exchange code for tokens, store encrypted, redirect to app.

    State format:
      "{user_id}"              → mobile deep link
      "{user_id}|web"          → web app callback
      "{user_id}__upgrade"     → mobile write-scope upgrade
      "{user_id}|web__upgrade" → web write-scope upgrade
    """
    is_upgrade = state.endswith("__upgrade") or "|web__upgrade" in state
    is_web = "|web" in state

    if provider == "google":
        await _handle_google_callback(code, state)
    elif provider == "slack":
        await _handle_slack_callback(code, state)
    else:
        raise NotFoundError(f"Unknown provider: {provider}")

    if is_web:
        # Redirect to the Next.js OAuth callback page
        web_url = f"{settings.WEB_APP_URL}/oauth/callback?provider={provider}&status=success"
        if is_upgrade:
            web_url += "&scope_upgrade=true"
        return RedirectResponse(url=web_url, status_code=302)

    # Mobile deep link
    deep_link = f"noteroute://oauth/success?provider={provider}"
    if is_upgrade:
        deep_link += "&scope_upgrade=true"

    return RedirectResponse(url=deep_link, status_code=302)


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
    # Strip platform and upgrade suffixes before using state as a user_id
    raw_user_id = state.removesuffix("__upgrade").split("|")[0]

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
        # Parse and store all granted scopes from the token response
        granted_scopes = tokens.get("scope", "").split()

        profile_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        profile = profile_resp.json() if profile_resp.status_code == 200 else {}

    provider_user_id = profile.get("id", "unknown")
    provider_email = profile.get("email")
    display_name = profile.get("name", "Google Account")

    user_id = PydanticObjectId(raw_user_id)
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
        # Merge new scopes with any previously granted ones
        existing.tokens.scopes = list(set(existing.tokens.scopes) | set(granted_scopes))
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
                scopes=granted_scopes,
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

    raw_user_id = state.split("|")[0]
    user_id = PydanticObjectId(raw_user_id)
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
