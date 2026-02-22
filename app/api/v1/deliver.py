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


class DeliverRequest(BaseModel):
    run_id: str
    slot_id: str | None = None
    content: str
    summary: str = ""
    user_id: str
    save_as_slot: bool = False
    target_tab_id: str | None = None
    doc_title: str | None = None
    trello_format: str = "note"  # "note" | "bullet" | "checklist"


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
            result = await _save_as_new_slot(user, body.content, body.summary, route, body.doc_title)
        else:
            result = await _deliver_to_slot(body.slot_id, body.content, user, body.target_tab_id, body.summary, body.trello_format)

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
        route.status = "failed"
        route.completed_at = now
        route.events.append(RouteEvent(
            event_type="failed",
            metadata={"error": str(exc)},
        ))
        await route.save()
        raise


async def _deliver_to_slot(slot_id: str | None, content: str, user: User, target_tab_id: str | None = None, summary: str = "", trello_format: str = "note") -> dict:
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
    elif source.provider == "todoist":
        resource_id = slot.destination.resource_id
        # Use first sentence / first 80 chars of transcript as title; full transcript as body
        first_sentence = content.split(".")[0].strip()
        task_title = (first_sentence[:80] if first_sentence else content[:80]).strip() or "Note"
        if " > " in (slot.destination.resource_name or ""):
            await todoist_svc.create_task(task_title, content, access_token, section_id=resource_id)
        else:
            await todoist_svc.create_task(task_title, content, access_token, project_id=resource_id)
    elif source.provider == "trello":
        target_card_id = target_tab_id  # reuse target_tab_id field for Trello card ID
        if target_card_id:
            await trello_svc.append_to_card(target_card_id, content, settings.TRELLO_API_KEY, access_token, fmt=trello_format)
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

        # Re-embed with updated content; summary vector uses description (stable)
        tags_str = " ".join(source.tags) if source.tags else ""
        source_context = f"{source.name} {source.provider} {tags_str} | "
        summary_vec, content_vec = await asyncio.gather(
            embed_text(source_context + slot.description),
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

    # User-provided title takes priority; fall back to summary/content
    if doc_title and doc_title.strip():
        title = doc_title.strip()[:120]
    else:
        title = (summary or content)[:60].strip() or "Note"

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
    elif source.provider == "todoist":
        # Save to the user's inbox project (falls back to first project)
        projects = await todoist_svc.list_projects(access_token)
        inbox = next((p for p in projects if p.get("is_inbox_project")), None)
        target = inbox or (projects[0] if projects else None)
        if not target:
            raise ValueError("No Todoist projects found")
        # Use first sentence / first 80 chars of transcript as title; full transcript as body
        first_sentence = content.split(".")[0].strip()
        task_title = (first_sentence[:80] if first_sentence else content[:80]).strip() or "Note"
        task = await todoist_svc.create_task(task_title, content, access_token, project_id=target["id"])
        resource = {"id": task["id"], "name": task["name"], "url": task.get("url")}
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

    # Embed and upsert to S3 Vectors
    try:
        tags_str = " ".join(source.tags) if source.tags else ""
        source_context = f"{source.name} {source.provider} {tags_str} | "
        summary_vec, content_vec = await asyncio.gather(
            embed_text(source_context + slot.description),
            embed_text(source_context + slot.content_sample),
        )
        vector_svc.upsert_slot(slot, summary_vec, content_vec)
        slot.index_status = "indexed"
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
