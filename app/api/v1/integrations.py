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

# Single scope: drive.file grants read+write to files the user explicitly selects
# via Google Picker and to files created by this app.
_GOOGLE_CONNECT_SCOPES = (
    "https://www.googleapis.com/auth/drive.file "
    "https://www.googleapis.com/auth/userinfo.email"
)
# Write scope is now included in the connect scopes (drive.file covers read+write).
_GOOGLE_WRITE_SCOPE = "https://www.googleapis.com/auth/drive.file"

_SLACK_SCOPES = "channels:read,chat:write"

_TODOIST_AUTH_URL = "https://todoist.com/oauth/authorize"
_TODOIST_TOKEN_URL = "https://todoist.com/oauth/access_token"
_TODOIST_SCOPES = "data:read_write"

_TRELLO_AUTH_URL = "https://trello.com/1/authorize"


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
        # Internal integration — no OAuth flow. Verify the shared token and store it.
        if not settings.NOTION_INTEGRATION_TOKEN:
            raise ValueError("NOTION_INTEGRATION_TOKEN is not configured")
        async with httpx.AsyncClient() as client:
            me_resp = await client.get(
                "https://api.notion.com/v1/users/me",
                headers={
                    "Authorization": f"Bearer {settings.NOTION_INTEGRATION_TOKEN}",
                    "Notion-Version": "2022-06-28",
                },
            )
            me_resp.raise_for_status()
            me = me_resp.json()
        bot_id = me.get("id", "unknown")
        workspace_name = "Notion"
        provider_email = me.get("person", {}).get("email")
        encrypted = encrypt_token(settings.NOTION_INTEGRATION_TOKEN)
        existing = await Integration.find_one(
            Integration.user_id == current_user.id,
            Integration.provider == "notion",
        )
        if existing:
            existing.tokens.access_token = encrypted
            existing.provider_email = provider_email
            existing.is_active = True
            await existing.save()
        else:
            await Integration(
                user_id=current_user.id,
                provider="notion",
                tokens=OAuthTokens(access_token=encrypted),
                provider_user_id=bot_id,
                provider_email=provider_email,
            ).insert()
        await _upsert_source(
            user=current_user,
            provider="notion",
            name=workspace_name,
            connected_account_id=bot_id,
            connected_account_email=provider_email,
        )
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

    elif provider == "todoist":
        url = (
            f"{_TODOIST_AUTH_URL}"
            f"?client_id={settings.TODOIST_CLIENT_ID}"
            f"&scope={_TODOIST_SCOPES}"
            f"&state={state}"
        )
        return {"status": "redirect", "provider": "todoist", "url": url}

    elif provider == "trello":
        from urllib.parse import quote
        # Trello returns the token as a URL *fragment* (#token=...) appended to return_url.
        # Point directly to the NextJS trello-callback page so the browser can read the hash.
        # Only encode the state param value, not the whole URL — Trello validates the return_url
        # origin against allowed origins and rejects if the scheme/host is percent-encoded.
        trello_callback_page = f"{settings.WEB_APP_URL}/oauth/trello-callback"
        return_url = f"{trello_callback_page}?state={quote(state, safe='')}"
        url = (
            f"{_TRELLO_AUTH_URL}"
            f"?key={settings.TRELLO_API_KEY}"
            f"&name=NoteRoute"
            f"&expiration=never"
            f"&scope=read,write"
            f"&response_type=token"
            f"&return_url={quote(return_url, safe=':/?=&')}"
        )
        return {"status": "redirect", "provider": "trello", "url": url}

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
    """Check whether the user has granted the drive.file (write) scope."""
    integration = await Integration.find_one(
        Integration.user_id == current_user.id,
        Integration.provider == "google",
        Integration.is_active == True,
    )
    if not integration:
        return {"has_write": False}
    has_write = _GOOGLE_WRITE_SCOPE in integration.tokens.scopes
    return {"has_write": has_write}


@router.get("/google/access-token")
async def google_access_token(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return a fresh Google access token for use with the Google Picker API."""
    from app.api.v1.sources import _get_fresh_google_token

    integration = await Integration.find_one(
        Integration.user_id == current_user.id,
        Integration.provider == "google",
        Integration.is_active == True,
    )
    if not integration:
        raise NotFoundError("Google integration not found")

    access_token = await _get_fresh_google_token(integration)
    return {"access_token": access_token}


@router.get("/trello/token", response_model=None)
async def trello_token(token: str, state: str) -> dict:
    """Called by the NextJS trello-callback page after extracting the token from
    window.location.hash. Trello uses response_type=token which delivers the token
    as a URL fragment (#token=...) — never sent to the server — so the frontend
    page must extract it and call this endpoint directly."""
    await _handle_trello_callback(token, state)
    return {"status": "connected", "provider": "trello"}


@router.get("/{provider}/callback", response_model=None)
async def oauth_callback(provider: str, code: str, state: str) -> RedirectResponse:
    """Handle OAuth callback — exchange code for tokens, store encrypted, redirect to app.

    State format:
      "{user_id}"              → mobile deep link
      "{user_id}|web"          → web app callback
      "{user_id}__upgrade"     → mobile write-scope upgrade
      "{user_id}|web__upgrade" → web write-scope upgrade
    """
    from fastapi import HTTPException as _HTTPException
    from urllib.parse import quote as _quote

    is_upgrade = state.endswith("__upgrade") or "|web__upgrade" in state
    is_web = "|web" in state

    try:
        if provider == "google":
            await _handle_google_callback(code, state)
        elif provider == "slack":
            await _handle_slack_callback(code, state)
        elif provider == "todoist":
            await _handle_todoist_callback(code, state)
        else:
            raise NotFoundError(f"Unknown provider: {provider}")
    except _HTTPException as exc:
        # Redirect with error instead of returning raw JSON to the browser
        error_msg = _quote(str(exc.detail), safe="")
        if is_web:
            return RedirectResponse(
                url=f"{settings.WEB_APP_URL}/oauth/callback?provider={provider}&status=error&message={error_msg}",
                status_code=302,
            )
        return RedirectResponse(
            url=f"noteroute://oauth/error?provider={provider}&message={error_msg}",
            status_code=302,
        )

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


async def _assert_provider_id_unclaimed(provider: str, provider_user_id: str, current_user_id) -> None:
    """Raise 409 if another NoteRoute account already owns this provider identity."""
    from fastapi import HTTPException as _HTTPException
    clash = await Integration.find_one(
        Integration.provider == provider,
        Integration.provider_user_id == provider_user_id,
        Integration.user_id != current_user_id,
    )
    if clash:
        raise _HTTPException(
            status_code=409,
            detail=f"This {provider} account is already connected to another NoteRoute account.",
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
        await _assert_provider_id_unclaimed("google", provider_user_id, user_id)
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
    authed_user_id = data.get("authed_user", {}).get("id")

    raw_user_id = state.split("|")[0]
    user_id = PydanticObjectId(raw_user_id)
    user = await User.get(user_id)
    if not user:
        return

    # Fetch the authed user's email via users.info
    from app.services import slack_svc
    provider_email: str | None = None
    if authed_user_id:
        try:
            user_info = await slack_svc.get_user_info(access_token, authed_user_id)
            provider_email = user_info.get("email")
        except Exception:
            pass  # email is best-effort

    existing = await Integration.find_one(
        Integration.user_id == user_id,
        Integration.provider == "slack",
    )
    encrypted = encrypt_token(access_token)

    if existing:
        existing.tokens.access_token = encrypted
        existing.workspace_name = workspace_name
        existing.provider_email = provider_email
        existing.is_active = True
        await existing.save()
    else:
        await _assert_provider_id_unclaimed("slack", workspace_id, user_id)
        await Integration(
            user_id=user_id,
            provider="slack",
            tokens=OAuthTokens(access_token=encrypted),
            provider_user_id=workspace_id,
            provider_email=provider_email,
            workspace_name=workspace_name,
        ).insert()

    await _upsert_source(
        user=user,
        provider="slack",
        name=workspace_name,
        connected_account_id=workspace_id,
        connected_account_email=provider_email,
    )


async def _upsert_source(
    user: User,
    provider: str,
    name: str,
    connected_account_id: str,
    connected_account_email: str | None,
) -> None:
    """Create or reactivate a Source document for the user+provider pair.

    Raises HTTPException 429 if the user has hit their source limit.
    Re-connecting an already-active integration never counts against the limit.
    """
    from fastapi import HTTPException as _HTTPException
    from app.api.deps import is_admin as _is_admin

    existing = await Source.find_one(
        Source.user_id == user.id,
        Source.provider == provider,
    )
    now = datetime.now(timezone.utc)

    if existing:
        was_inactive = not existing.is_active
        if was_inactive and not _is_admin(user) and user.usage.sources_count >= user.limits.max_sources:
            raise _HTTPException(
                status_code=429,
                detail=f"Source limit reached ({user.limits.max_sources}). Upgrade to Pro for more.",
            )
        existing.name = name
        existing.connected_account_id = connected_account_id
        existing.connected_account_email = connected_account_email
        existing.is_active = True
        existing.updated_at = now
        await existing.save()
        if was_inactive:
            user.usage.sources_count += 1
            await user.save()
    else:
        if not _is_admin(user) and user.usage.sources_count >= user.limits.max_sources:
            raise _HTTPException(
                status_code=429,
                detail=f"Source limit reached ({user.limits.max_sources}). Upgrade to Pro for more.",
            )
        await Source(
            user_id=user.id,
            provider=provider,
            name=name,
            connected_account_id=connected_account_id,
            connected_account_email=connected_account_email,
        ).insert()
        user.usage.sources_count += 1
        await user.save()


async def _handle_trello_callback(token: str, state: str) -> None:
    """Store Trello OAuth token, fetch user info, upsert Integration + Source.

    Trello issues a single long-lived token (no refresh, no expiry).
    The token is used alongside the static TRELLO_API_KEY for all API calls.
    """
    from app.services import trello_svc

    raw_user_id = state.split("|")[0]

    user_info = await trello_svc.get_user_info(settings.TRELLO_API_KEY, token)

    provider_user_id = user_info["id"]
    provider_email = user_info.get("email")
    display_name = user_info["name"]

    user_id = PydanticObjectId(raw_user_id)
    user = await User.get(user_id)
    if not user:
        return

    encrypted = encrypt_token(token)
    existing = await Integration.find_one(
        Integration.user_id == user_id,
        Integration.provider == "trello",
    )

    if existing:
        existing.tokens.access_token = encrypted
        existing.provider_email = provider_email
        existing.workspace_name = display_name
        existing.is_active = True
        await existing.save()
    else:
        await _assert_provider_id_unclaimed("trello", provider_user_id, user_id)
        await Integration(
            user_id=user_id,
            provider="trello",
            tokens=OAuthTokens(access_token=encrypted),
            provider_user_id=provider_user_id,
            provider_email=provider_email,
            workspace_name=display_name,
        ).insert()

    await _upsert_source(
        user=user,
        provider="trello",
        name=f"{display_name}'s Trello",
        connected_account_id=provider_user_id,
        connected_account_email=provider_email,
    )


async def _handle_todoist_callback(code: str, state: str) -> None:
    """Exchange Todoist auth code for access token, store, upsert Source.

    Todoist issues a single long-lived bearer token (no refresh token, no expiry).
    """
    from app.services import todoist_svc

    raw_user_id = state.split("|")[0]

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            _TODOIST_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.TODOIST_CLIENT_ID,
                "client_secret": settings.TODOIST_CLIENT_SECRET,
            },
        )
        token_resp.raise_for_status()
        data = token_resp.json()

    access_token = data["access_token"]
    user_info = await todoist_svc.get_user_info(access_token)

    provider_user_id = user_info["id"]
    provider_email = user_info["email"]
    display_name = user_info["name"]

    user_id = PydanticObjectId(raw_user_id)
    user = await User.get(user_id)
    if not user:
        return

    encrypted = encrypt_token(access_token)
    existing = await Integration.find_one(
        Integration.user_id == user_id,
        Integration.provider == "todoist",
    )

    if existing:
        existing.tokens.access_token = encrypted
        existing.provider_email = provider_email
        existing.workspace_name = display_name
        existing.is_active = True
        await existing.save()
    else:
        await _assert_provider_id_unclaimed("todoist", provider_user_id, user_id)
        await Integration(
            user_id=user_id,
            provider="todoist",
            tokens=OAuthTokens(access_token=encrypted),
            provider_user_id=provider_user_id,
            provider_email=provider_email,
            workspace_name=display_name,
        ).insert()

    await _upsert_source(
        user=user,
        provider="todoist",
        name=f"{display_name}'s Todoist",
        connected_account_id=provider_user_id,
        connected_account_email=provider_email,
    )
