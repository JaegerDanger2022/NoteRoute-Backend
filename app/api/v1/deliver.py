import logging
from datetime import datetime, timezone

from beanie import PydanticObjectId
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.exceptions import NotFoundError
from app.core.security import decrypt_token, encrypt_token
from app.models.integration import Integration
from app.models.route import Route, RouteEvent
from app.models.slot import KnowledgeSlot, SlotDestination
from app.models.source import Source
from app.models.user import User
from app.services import gdocs_svc, notion_svc, slack_svc, vector_svc
from app.utils.embeddings import embed_text
import asyncio

logger = logging.getLogger(__name__)

# Internal endpoint — called by LangGraph, not authenticated via Firebase
router = APIRouter(prefix="/deliver", tags=["deliver"])


async def _get_access_token(integration: Integration) -> str:
    """Return a valid access token, auto-refreshing Google tokens when expired."""
    if integration.provider != "google" or not integration.tokens.refresh_token:
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


class DeliverRequest(BaseModel):
    run_id: str
    slot_id: str | None = None
    content: str
    summary: str = ""
    user_id: str
    save_as_slot: bool = False
    target_tab_id: str | None = None


@router.post("")
async def deliver(body: DeliverRequest) -> dict:
    """
    Called by LangGraph after slot confirmation.
    Writes content to the provider, updates the Route record.
    If save_as_slot=True, creates a new resource + KnowledgeSlot instead.
    """
    user = await User.get(PydanticObjectId(body.user_id))
    if not user:
        raise NotFoundError("User not found")

    route = await Route.find_one(Route.run_id == body.run_id)
    if not route:
        raise NotFoundError("Route not found")

    now = datetime.now(timezone.utc)

    try:
        if body.save_as_slot:
            result = await _save_as_new_slot(user, body.content, body.summary, route)
        else:
            result = await _deliver_to_slot(body.slot_id, body.content, user, body.target_tab_id)

        route.status = "delivered"
        route.summary = body.summary or None
        route.completed_at = now
        route.events.append(RouteEvent(
            event_type="delivered",
            metadata=result,
        ))
        await route.save()

        return {"status": "delivered", **result}

    except Exception as exc:
        logger.exception("Delivery failed for run_id=%s", body.run_id)
        route.status = "failed"
        route.completed_at = now
        route.events.append(RouteEvent(
            event_type="failed",
            metadata={"error": str(exc)},
        ))
        await route.save()
        raise


async def _deliver_to_slot(slot_id: str | None, content: str, user: User, target_tab_id: str | None = None) -> dict:
    if not slot_id:
        raise ValueError("No slot_id provided for delivery")

    slot = await KnowledgeSlot.get(PydanticObjectId(slot_id))
    if not slot:
        raise NotFoundError(f"Slot {slot_id} not found")

    source = await Source.get(slot.source_id)
    if not source:
        raise NotFoundError("Source not found for slot")

    integration = await Integration.find_one(
        Integration.user_id == user.id,
        Integration.provider == source.provider,
        Integration.is_active == True,
    )
    if not integration:
        raise NotFoundError(f"No active {source.provider} integration")

    access_token = await _get_access_token(integration)

    if source.provider == "notion":
        await notion_svc.append_block(slot.destination.resource_id, content, access_token)
    elif source.provider == "google":
        await gdocs_svc.append_content(slot.destination.resource_id, content, access_token, tab_id=target_tab_id)
    elif source.provider == "slack":
        await slack_svc.post_message(slot.destination.resource_id, content, access_token)

    # Update last_used_at on integration
    integration.last_used_at = datetime.now(timezone.utc)
    await integration.save()

    return {
        "slot_id": slot_id,
        "slot_name": slot.name,
        "provider": source.provider,
    }


async def _save_as_new_slot(
    user: User,
    content: str,
    summary: str,
    route: Route,
) -> dict:
    """Create a new resource in the provider + save it as a KnowledgeSlot."""
    if not user.active_source_id:
        raise ValueError("No active source set on user")

    source = await Source.get(user.active_source_id)
    if not source:
        raise NotFoundError("Active source not found")

    integration = await Integration.find_one(
        Integration.user_id == user.id,
        Integration.provider == source.provider,
        Integration.is_active == True,
    )
    if not integration:
        raise NotFoundError(f"No active {source.provider} integration")

    access_token = await _get_access_token(integration)

    # Title from summary (first 60 chars) or first line of content
    title = (summary or content)[:60].strip()
    if not title:
        title = "Note"

    # Create the resource in the provider
    if source.provider == "notion":
        # For Notion, we need a parent page. Use the first accessible page as parent.
        pages = await notion_svc.list_pages(access_token)
        if not pages:
            raise ValueError("No Notion pages available to create child page under")
        parent_id = pages[0]["id"]
        resource = await notion_svc.create_page(parent_id, title, content, access_token)
    elif source.provider == "google":
        resource = await gdocs_svc.create_document(title, content, access_token)
    elif source.provider == "slack":
        # For Slack, post to the first available channel and use that as the slot
        channels = await slack_svc.list_channels(access_token)
        if not channels:
            raise ValueError("No Slack channels available")
        channel = channels[0]
        await slack_svc.post_message(channel["id"], content, access_token)
        resource = channel
    else:
        raise ValueError(f"Unknown provider: {source.provider}")

    # Create the KnowledgeSlot
    slot = KnowledgeSlot(
        user_id=user.id,
        source_id=source.id,
        name=title,
        description=summary or content[:200],
        content_sample=content[:500],
        destination=SlotDestination(
            resource_id=resource["id"],
            resource_name=resource["name"],
            resource_url=resource.get("url"),
        ),
        tags=source.tags,
    )
    await slot.insert()

    user.usage.slots_count += 1
    await user.save()

    # Embed and upsert to S3 Vectors
    try:
        tags_str = " ".join(source.tags) if source.tags else ""
        source_context = f"{source.name} {source.provider} {tags_str} | "
        summary_vec, content_vec = await asyncio.gather(
            embed_text(source_context + slot.description),
            embed_text(source_context + slot.content_sample),
        )
        vector_svc.upsert_slot(slot, summary_vec, content_vec)
    except Exception:
        logger.exception("Vector upsert failed for new slot %s", slot.id)

    # Link route to the new slot
    route.confirmed_slot_id = slot.id
    await route.save()

    integration.last_used_at = datetime.now(timezone.utc)
    await integration.save()

    return {
        "slot_id": str(slot.id),
        "slot_name": slot.name,
        "provider": source.provider,
        "saved_as_new_slot": True,
    }
