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
from app.config import settings
from app.services import gdocs_svc, notion_svc, slack_svc, todoist_svc, trello_svc, vector_svc
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


def _is_token_revoked(err_str: str) -> bool:
    """Return True if the error looks like a provider-side token revocation or 401."""
    lowered = err_str.lower()
    return any(kw in lowered for kw in ("401", "unauthorized", "invalid_token", "token revoked", "token expired", "invalid credentials", "403", "forbidden"))


async def _mark_integration_inactive(user: User, exc: Exception) -> None:
    """Best-effort: mark the integration for the user's active source as inactive."""
    try:
        source = await Source.get(user.active_source_id) if user.active_source_id else None
        if not source:
            return
        integration = await Integration.find_one(
            Integration.user_id == user.id,
            Integration.provider == source.provider,
        )
        if integration and integration.is_active:
            integration.is_active = False
            await integration.save()
            logger.warning(
                "Marked %s integration inactive for user %s due to token error: %s",
                source.provider, user.id, exc,
            )
    except Exception:
        logger.exception("Could not mark integration inactive after token error")


class DeliverRequest(BaseModel):
    run_id: str
    slot_id: str | None = None
    content: str
    summary: str = ""
    user_id: str
    save_as_slot: bool = False
    target_tab_id: str | None = None
    doc_title: str | None = None
    trello_format: str = "note"  # "note" | "checklist"
    trello_checklist_title: str | None = None
    trello_checklist_id: str | None = None
    notion_parent_page_id: str | None = None  # parent page when save_as_slot=True for notion


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
            result = await _save_as_new_slot(user, body.content, body.summary, route, body.doc_title, body.notion_parent_page_id)
        else:
            result = await _deliver_to_slot(body.slot_id, body.content, user, body.target_tab_id, body.summary, body.trello_format, body.trello_checklist_title, body.trello_checklist_id)

        route.status = "delivered"
        route.summary = body.summary or None
        route.completed_at = now
        route.delivery_url = result.get("resource_url")
        route.slot_name = result.get("slot_name")
        route.events.append(RouteEvent(
            event_type="delivered",
            metadata=result,
        ))
        await route.save()

        return {"status": "delivered", **result}

    except Exception as exc:
        logger.exception("Delivery failed for run_id=%s", body.run_id)
        err_str = str(exc)
        # If the provider rejected the token, mark the integration inactive so the user knows
        if _is_token_revoked(err_str):
            await _mark_integration_inactive(user, exc)
        route.status = "failed"
        route.completed_at = now
        route.events.append(RouteEvent(
            event_type="failed",
            metadata={"error": err_str},
        ))
        await route.save()
        raise


async def _deliver_to_slot(slot_id: str | None, content: str, user: User, target_tab_id: str | None = None, summary: str = "", trello_format: str = "note", trello_checklist_title: str | None = None, trello_checklist_id: str | None = None) -> dict:
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
    g_refresh = decrypt_token(integration.tokens.refresh_token) if integration.provider == "google" and integration.tokens.refresh_token else ""

    if source.provider == "notion":
        await notion_svc.append_block(slot.destination.resource_id, content, access_token)
    elif source.provider == "google":
        await gdocs_svc.append_content(slot.destination.resource_id, content, access_token, tab_id=target_tab_id, refresh_token=g_refresh)
    elif source.provider == "slack":
        await slack_svc.post_message(slot.destination.resource_id, content, access_token)
    elif source.provider == "todoist":
        resource_id = slot.destination.resource_id
        task_title = (summary.split("\n")[0].strip()[:80] if summary else None) or (content.split(". ")[0].strip()[:80]) or "Note"
        if " > " in (slot.destination.resource_name or ""):
            await todoist_svc.create_task(task_title, content, access_token, section_id=resource_id)
        else:
            await todoist_svc.create_task(task_title, content, access_token, project_id=resource_id)
    elif source.provider == "trello":
        target_card_id = target_tab_id  # reuse target_tab_id field for Trello card ID
        if target_card_id:
            await trello_svc.append_to_card(target_card_id, content, settings.TRELLO_API_KEY, access_token, fmt=trello_format, checklist_title=trello_checklist_title, checklist_id=trello_checklist_id)
        else:
            # Use first sentence of the summary as a coherent title, capped at 50 chars
            first_sentence = (summary or content).split(".")[0].strip()
            if len(first_sentence) <= 50:
                card_name = first_sentence.rstrip(".,;:") or "Note"
            else:
                truncated = first_sentence[:50]
                last_space = truncated.rfind(" ")
                card_name = (truncated[:last_space] if last_space > 10 else truncated).rstrip(".,;:") or "Note"
            await trello_svc.create_card(slot.destination.resource_id, card_name, content, settings.TRELLO_API_KEY, access_token, fmt=trello_format)

    # Update last_used_at on integration
    integration.last_used_at = datetime.now(timezone.utc)
    await integration.save()

    # Re-index slot in background — doesn't block delivery response
    asyncio.create_task(_reindex_slot(slot, source, content))

    return {
        "slot_id": slot_id,
        "slot_name": slot.name,
        "provider": source.provider,
        "resource_url": slot.destination.resource_url if slot.destination else None,
    }


async def _reindex_slot(slot: KnowledgeSlot, source: Source, new_content: str) -> None:
    """Append delivered content to slot's content_sample and upsert Pinecone vectors."""
    try:
        # Rolling content_sample: keep last 1000 chars + new content (max 500 chars)
        combined = (slot.content_sample + "\n" + new_content[:500]).strip()
        slot.content_sample = combined[-1000:]
        slot.updated_at = datetime.now(timezone.utc)
        await slot.save()

        # Re-embed with updated content; summary vector uses name + description for strong signal
        tags_str = " ".join(source.tags) if source.tags else ""
        source_context = f"{source.name} {source.provider} {tags_str} | "
        summary_vec, content_vec = await asyncio.gather(
            embed_text(source_context + slot.name + ": " + slot.description),
            embed_text(source_context + slot.content_sample),
        )
        vector_svc.upsert_slot(slot, summary_vec, content_vec)
        logger.info("Re-indexed slot %s after delivery", slot.id)
    except Exception:
        logger.exception("Background re-index failed for slot %s", slot.id)


async def _save_as_new_slot(
    user: User,
    content: str,
    summary: str,
    route: Route,
    doc_title: str | None = None,
    notion_parent_page_id: str | None = None,
    override_source_id=None,
) -> dict:
    """Create a new resource in the provider + save it as a KnowledgeSlot."""
    source_id = override_source_id or user.active_source_id
    if not source_id:
        raise ValueError("No active source set on user")

    source = await Source.get(source_id)
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
    g_refresh = decrypt_token(integration.tokens.refresh_token) if integration.provider == "google" and integration.tokens.refresh_token else ""

    # User-provided title takes priority; fall back to summary/content
    if doc_title and doc_title.strip():
        title = doc_title.strip()[:120]
    else:
        title = (summary or content)[:60].strip() or "Note"

    # Create the resource in the provider
    if source.provider == "notion":
        resource = await notion_svc.create_page(notion_parent_page_id or None, title, content, access_token)
    elif source.provider == "google":
        resource = await gdocs_svc.create_document(title, content, access_token, refresh_token=g_refresh)
    elif source.provider == "slack":
        # For Slack, post to the first available channel and use that as the slot
        channels = await slack_svc.list_channels(access_token)
        if not channels:
            raise ValueError("No Slack channels available")
        channel = channels[0]
        await slack_svc.post_message(channel["id"], content, access_token)
        resource = channel
    elif source.provider == "todoist":
        # Create a new Todoist project for this slot.
        # Check project count first — Todoist free plan allows 5, pro allows 300.
        projects = await todoist_svc.list_projects(access_token)
        user_project_count = todoist_svc.count_user_projects(projects)
        is_pro = getattr(user, "tier", "free") == "pro"
        project_limit = todoist_svc.TODOIST_PRO_PROJECT_LIMIT if is_pro else todoist_svc.TODOIST_FREE_PROJECT_LIMIT
        if user_project_count >= project_limit:
            raise ValueError(
                f"Todoist project limit reached ({project_limit} projects). "
                f"{'Upgrade your Todoist plan' if not is_pro else 'Delete unused projects'} to create more."
            )
        project = await todoist_svc.create_project(title, access_token)
        task_title = (summary.split("\n")[0].strip()[:80] if summary else None) or (content.split(". ")[0].strip()[:80]) or "Note"
        task = await todoist_svc.create_task(task_title, content, access_token, project_id=project["id"])
        resource = {"id": project["id"], "name": project["name"], "url": project.get("url")}
    elif source.provider == "trello":
        # Save to the first list of the first board
        boards = await trello_svc.list_boards(settings.TRELLO_API_KEY, access_token)
        if not boards:
            raise ValueError("No Trello boards found")
        lists = await trello_svc.list_lists(boards[0]["id"], settings.TRELLO_API_KEY, access_token)
        if not lists:
            raise ValueError("No Trello lists found on first board")
        card = await trello_svc.create_card(lists[0]["id"], title, content, settings.TRELLO_API_KEY, access_token)
        resource = {"id": lists[0]["id"], "name": f"{boards[0]['name']} > {lists[0]['name']}", "url": card.get("url")}
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

    # Embed and upsert to Pinecone
    try:
        tags_str = " ".join(source.tags) if source.tags else ""
        source_context = f"{source.name} {source.provider} {tags_str} | "
        summary_vec, content_vec = await asyncio.gather(
            embed_text(source_context + slot.name + ": " + slot.description),
            embed_text(source_context + slot.content_sample),
        )
        vector_svc.upsert_slot(slot, summary_vec, content_vec)
        slot.index_status = "indexed"
        slot.index_name = settings.PINECONE_INDEX_NAME
        await slot.save()
    except Exception:
        logger.exception("Vector upsert failed for new slot %s", slot.id)
        slot.index_status = "failed"
        await slot.save()

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
        "resource_url": slot.destination.resource_url if slot.destination else None,
    }
