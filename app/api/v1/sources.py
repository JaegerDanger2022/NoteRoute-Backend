import logging
from datetime import datetime, timezone

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.exceptions import NotFoundError, TierLimitError
from app.core.security import decrypt_token, encrypt_token
from app.models.integration import Integration
from app.models.slot import KnowledgeSlot
from app.models.source import Source
from app.models.user import User
from app.services import gdocs_svc, notion_svc, slack_svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sources", tags=["sources"])


class SourceCreateRequest(BaseModel):
    provider: str
    name: str
    tags: list[str] = []


class SourceUpdateRequest(BaseModel):
    name: str | None = None
    tags: list[str] | None = None


def _source_to_dict(source: Source) -> dict:
    return {
        "id": str(source.id),
        "provider": source.provider,
        "name": source.name,
        "tags": source.tags,
        "connected_account_email": source.connected_account_email,
        "connected_account_id": source.connected_account_id,
        "is_active": source.is_active,
        "created_at": source.created_at.isoformat(),
    }


@router.get("")
async def list_sources(current_user: User = Depends(get_current_user)) -> list[dict]:
    sources = await Source.find(
        Source.user_id == current_user.id,
        Source.is_active == True,
    ).to_list()
    return [_source_to_dict(s) for s in sources]


@router.post("", status_code=201)
async def create_source(
    body: SourceCreateRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    if current_user.usage.sources_count >= current_user.limits.max_sources:
        raise TierLimitError(
            f"Source limit reached ({current_user.limits.max_sources}). Upgrade your plan."
        )

    # Integration must already be connected
    integration = await Integration.find_one(
        Integration.user_id == current_user.id,
        Integration.provider == body.provider,
        Integration.is_active == True,
    )
    if not integration:
        raise NotFoundError(
            f"No active {body.provider} integration found. Connect it first."
        )

    # Check if source for this provider already exists (reactivate if so)
    existing = await Source.find_one(
        Source.user_id == current_user.id,
        Source.provider == body.provider,
    )
    now = datetime.now(timezone.utc)

    if existing:
        existing.name = body.name
        existing.tags = body.tags
        existing.is_active = True
        existing.updated_at = now
        await existing.save()
        return _source_to_dict(existing)

    source = Source(
        user_id=current_user.id,
        provider=body.provider,
        name=body.name,
        tags=body.tags,
        connected_account_email=integration.provider_email,
        connected_account_id=integration.provider_user_id,
    )
    await source.insert()

    current_user.usage.sources_count += 1
    await current_user.save()

    return _source_to_dict(source)


@router.patch("/{source_id}")
async def update_source(
    source_id: PydanticObjectId,
    body: SourceUpdateRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    source = await Source.find_one(
        Source.id == source_id,
        Source.user_id == current_user.id,
    )
    if not source:
        raise NotFoundError("Source not found")

    if body.name is not None:
        source.name = body.name
    if body.tags is not None:
        source.tags = body.tags
    source.updated_at = datetime.now(timezone.utc)
    await source.save()
    return _source_to_dict(source)


@router.delete("/{source_id}", status_code=204)
async def delete_source(
    source_id: PydanticObjectId,
    current_user: User = Depends(get_current_user),
) -> None:
    source = await Source.find_one(
        Source.id == source_id,
        Source.user_id == current_user.id,
    )
    if not source:
        raise NotFoundError("Source not found")

    source.is_active = False
    source.updated_at = datetime.now(timezone.utc)
    await source.save()

    # Soft-delete all slots belonging to this source
    await KnowledgeSlot.find(
        KnowledgeSlot.source_id == source_id,
        KnowledgeSlot.user_id == current_user.id,
    ).update({"$set": {"is_active": False}})

    current_user.usage.sources_count = max(0, current_user.usage.sources_count - 1)
    if str(current_user.active_source_id) == str(source_id):
        current_user.active_source_id = None
    await current_user.save()


async def _get_fresh_google_token(integration: Integration) -> str:
    """Return a valid Google access token, refreshing if expired or no expiry stored."""
    from datetime import timezone
    needs_refresh = True
    if integration.tokens.expires_at:
        expires = integration.tokens.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        needs_refresh = (expires - datetime.now(timezone.utc)).total_seconds() < 60

    if needs_refresh and integration.tokens.refresh_token:
        refresh_token = decrypt_token(integration.tokens.refresh_token)
        new_access, expires_at = await gdocs_svc.refresh_access_token(refresh_token)
        integration.tokens.access_token = encrypt_token(new_access)
        integration.tokens.expires_at = expires_at
        await integration.save()
        return new_access

    return decrypt_token(integration.tokens.access_token)


@router.get("/{source_id}/resources")
async def list_resources(
    source_id: PydanticObjectId,
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """List provider resources (pages/docs/channels) for slot selection."""
    source = await Source.find_one(
        Source.id == source_id,
        Source.user_id == current_user.id,
        Source.is_active == True,
    )
    if not source:
        raise NotFoundError("Source not found")

    integration = await Integration.find_one(
        Integration.user_id == current_user.id,
        Integration.provider == source.provider,
        Integration.is_active == True,
    )
    if not integration:
        raise NotFoundError(f"No active {source.provider} integration found")

    if source.provider == "google":
        access_token = await _get_fresh_google_token(integration)
    else:
        access_token = decrypt_token(integration.tokens.access_token)

    if source.provider == "notion":
        return await notion_svc.list_pages(access_token)
    elif source.provider == "google":
        return await gdocs_svc.list_documents(access_token)
    elif source.provider == "slack":
        return await slack_svc.list_channels(access_token)
    else:
        raise NotFoundError(f"Unknown provider: {source.provider}")
