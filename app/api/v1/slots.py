import asyncio
import logging
from datetime import datetime, timezone

from beanie import PydanticObjectId
from beanie.operators import Set
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from pydantic import BaseModel

from app.api.deps import get_current_user, is_admin
from app.config import settings
from app.core.exceptions import BadRequestError, NotFoundError, TierLimitError
from app.core.security import decrypt_token, encrypt_token
from app.models.integration import Integration
from app.models.slot import KnowledgeSlot, SlotDestination
from app.models.source import Source
from app.models.user import User
from app.services import claude_svc, gdocs_svc, notion_svc, slack_svc, todoist_svc, trello_svc, vector_svc
from app.services.claude_svc import CustomLLMCreds
from app.services.vector_svc import CustomIndexCreds
from app.utils.embeddings import CustomBedrockCreds, embed_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slots", tags=["slots"])


class SlotCreateRequest(BaseModel):
    source_id: str
    name: str
    description: str
    content_sample: str = ""
    destination: SlotDestination
    tags: list[str] = []
    include_subpages: bool = True  # Notion only: recursively index child pages
    read_content: bool = False


class SlotBulkCreateRequest(BaseModel):
    slots: list[SlotCreateRequest]


class SlotUpdateRequest(BaseModel):
    source_id: str | None = None
    name: str | None = None
    description: str | None = None
    content_sample: str | None = None
    tags: list[str] | None = None


def _slot_to_dict(slot: KnowledgeSlot) -> dict:
    return {
        "id": str(slot.id),
        "source_id": str(slot.source_id),
        "name": slot.name,
        "description": slot.description,
        "content_sample": slot.content_sample,
        "destination": slot.destination.model_dump(),
        "tags": slot.tags,
        "read_content": slot.read_content,
        "index_status": slot.index_status,
        "index_name": slot.index_name,
        "is_active": slot.is_active,
        "created_at": slot.created_at.isoformat(),
        "updated_at": slot.updated_at.isoformat(),
    }


@router.get("")
async def list_slots(
    source_id: str | None = None,
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    # Only return primary slots — section children (parent_slot_id set) are internal
    query = [
        KnowledgeSlot.user_id == current_user.id,
        KnowledgeSlot.is_active == True,
        KnowledgeSlot.parent_slot_id == None,
    ]
    if source_id:
        query.append(KnowledgeSlot.source_id == PydanticObjectId(source_id))
    slots = await KnowledgeSlot.find(*query).to_list()
    return [_slot_to_dict(s) for s in slots]


@router.get("/internal/batch")
async def batch_slot_metadata(ids: str = Query(..., description="Comma-separated slot ObjectIds")) -> list[dict]:
    """Internal endpoint (no auth) — returns slot metadata for LangGraph rank_node enrichment."""
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    if not id_list:
        return []

    object_ids = []
    for raw_id in id_list:
        try:
            object_ids.append(PydanticObjectId(raw_id))
        except Exception:
            pass

    slots = await KnowledgeSlot.find(
        {"_id": {"$in": object_ids}, "is_active": True}
    ).to_list()

    # Batch-fetch sources to resolve integration_type (provider)
    source_ids = list({s.source_id for s in slots})
    sources = await Source.find({"_id": {"$in": source_ids}}).to_list()
    source_map = {s.id: s for s in sources}

    result = []
    for slot in slots:
        src = source_map.get(slot.source_id)
        result.append({
            "slot_id": str(slot.id),
            "slot_name": slot.name,
            "integration_type": src.provider if src else "unknown",
            "resource_id": slot.destination.resource_id,
        })
    return result


async def _fetch_resource_content(slot: KnowledgeSlot, provider: str, user_id: str) -> str:
    """Fetch raw text content from the provider resource this slot points to."""
    integration = await Integration.find_one(
        Integration.user_id == slot.user_id,
        Integration.provider == provider,
        Integration.is_active == True,
    )
    if not integration:
        return ""
    access_token = decrypt_token(integration.tokens.access_token)
    g_refresh = decrypt_token(integration.tokens.refresh_token) if provider == "google" and integration.tokens.refresh_token else ""
    resource_id = slot.destination.resource_id
    if provider == "notion":
        return await notion_svc.fetch_page_text(resource_id, access_token, include_subpages=slot.include_subpages)
    elif provider == "google":
        return await gdocs_svc.fetch_document_text(resource_id, access_token, refresh_token=g_refresh)
    elif provider == "slack":
        return await slack_svc.fetch_channel_messages(resource_id, access_token)
    elif provider == "todoist":
        # Section slots (parent_slot_id set) use section_id; project slots use project_id
        if slot.parent_slot_id is not None:
            return await todoist_svc.fetch_section_tasks(resource_id, access_token)
        return await todoist_svc.fetch_project_tasks(resource_id, access_token)
    elif provider == "trello":
        return await trello_svc.fetch_list_cards(resource_id, settings.TRELLO_API_KEY, access_token)
    return ""


def _resolve_custom_creds(user: User) -> tuple[CustomIndexCreds | None, CustomBedrockCreds | None, CustomLLMCreds | None]:
    """Extract decrypted custom index, bedrock, and LLM creds from the user if configured."""
    cfg = user.custom_index
    custom_index = None
    custom_bedrock = None
    if cfg and cfg.index_status == "ready":
        custom_index = CustomIndexCreds(
            pinecone_api_key=decrypt_token(cfg.pinecone_api_key),
            index_name=cfg.index_name,
        )
        if cfg.bedrock_aws_access_key_id and cfg.bedrock_aws_secret_access_key:
            custom_bedrock = CustomBedrockCreds(
                aws_access_key_id=decrypt_token(cfg.bedrock_aws_access_key_id),
                aws_secret_access_key=decrypt_token(cfg.bedrock_aws_secret_access_key),
                aws_region=cfg.bedrock_aws_region or "us-east-1",
            )

    custom_llm = None
    if user.custom_llm:
        custom_llm = CustomLLMCreds(
            provider=user.custom_llm.provider,
            api_key=decrypt_token(user.custom_llm.api_key),
        )

    return custom_index, custom_bedrock, custom_llm


async def _embed_and_enrich_slot(
    slot_id: str,
    user_id: str,
    source_name: str,
    provider: str,
    user_provided_description: bool,
    user_provided_tags: bool,
) -> None:
    """Background task: optionally read content, enrich metadata, embed, and upsert."""
    slot = await KnowledgeSlot.get(PydanticObjectId(slot_id))
    if not slot:
        return

    # Resolve custom index/bedrock/llm creds from user config
    user = await User.get(PydanticObjectId(user_id))
    custom_index, custom_bedrock, custom_llm = _resolve_custom_creds(user) if user else (None, None, None)

    # Check if custom index was deleted before we try to use it
    if custom_index and not await asyncio.to_thread(vector_svc.check_custom_index_exists, custom_index):
        if user and user.custom_index:
            user.custom_index.index_status = "deleted"
            await user.save()
        custom_index = None  # Fall back to shared index

    # Step 1: read & summarize resource content if user opted in
    if slot.read_content:
        try:
            raw_content = await _fetch_resource_content(slot, provider, user_id)
            if raw_content.strip():
                summary = await claude_svc.summarize_slot_content(slot.name, raw_content, custom_llm)
                slot.content_sample = summary
                slot.raw_content = raw_content  # persisted for content_vector embedding
                await slot.save()
                logger.info("Content indexed for slot %s", slot_id)
        except Exception:
            logger.exception("Content reading failed for slot %s", slot_id)

    # Step 2: auto-enrich description/tags if not user-provided
    if not user_provided_description or not user_provided_tags:
        try:
            meta = await claude_svc.infer_slot_metadata(slot.name, source_name, provider, custom_llm)
            changed = False
            if not user_provided_description and meta.get("description"):
                slot.description = meta["description"]
                changed = True
            if not user_provided_tags and meta.get("tags"):
                slot.tags = meta["tags"]
                changed = True
            if changed:
                await slot.save()
        except Exception:
            logger.exception("Auto-enrich failed for slot %s", slot_id)

    # Step 3: embed and upsert to Pinecone, then mark indexed
    try:
        source = await Source.get(slot.source_id)
        summary_vec, content_vecs = await _embed_slot(slot, source, custom_bedrock)
        if len(content_vecs) > 1:
            vector_svc.upsert_slot_content_chunks(slot, summary_vec, content_vecs, custom_index)
        else:
            vector_svc.upsert_slot(slot, summary_vec, content_vecs[0], custom_index)
        slot.index_status = "indexed"
        slot.index_name = custom_index.index_name if custom_index else settings.PINECONE_INDEX_NAME
        # Persist encrypted API key so we can delete from the correct index later,
        # even if the user has since disconnected or rotated their custom index.
        if custom_index:
            slot.index_api_key = encrypt_token(custom_index.pinecone_api_key)
        else:
            slot.index_api_key = ""  # shared index — no key needed
        await slot.save()
        logger.info("Vector upsert complete for slot %s", slot_id)
    except Exception:
        logger.exception("Vector upsert failed for slot %s", slot_id)
        try:
            slot.index_status = "failed"
            await slot.save()
        except Exception:
            pass


async def _build_todoist_section_slots(
    project_resource: SlotDestination,
    project_name: str,
    source_id: PydanticObjectId,
    user_id,
    parent_slot_id: PydanticObjectId,
    tags: list[str],
    read_content: bool,
    access_token: str,
) -> list[KnowledgeSlot]:
    """Return unsaved KnowledgeSlot objects for every section under a Todoist project.

    Section slots are owned by the project slot (parent_slot_id) and do NOT
    count toward the user's slot quota — the project slot counts as 1.
    """
    try:
        sections = await todoist_svc.list_sections(project_resource.resource_id, access_token)
    except Exception:
        logger.warning("Could not fetch sections for Todoist project %s", project_resource.resource_id)
        return []

    slots = []
    for section in sections:
        section_name = f"{project_name} > {section['name']}"
        slots.append(KnowledgeSlot(
            user_id=user_id,
            source_id=source_id,
            parent_slot_id=parent_slot_id,
            name=section_name,
            description=section_name,
            content_sample="",
            destination=SlotDestination(
                resource_id=section["id"],
                resource_name=section_name,
                resource_url=None,
            ),
            tags=tags,
            read_content=read_content,
        ))
    return slots


@router.post("", status_code=201)
async def create_slot(
    body: SlotCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
) -> dict:
    if not body.read_content and not (body.description or "").strip():
        raise BadRequestError("Description is required when Read & index content is off.")
    if not is_admin(current_user) and current_user.usage.slots_count >= current_user.limits.max_slots:
        raise TierLimitError(
            f"Slot limit reached ({current_user.limits.max_slots}). Upgrade your plan."
        )

    source_id = PydanticObjectId(body.source_id)
    source = await Source.find_one(
        Source.id == source_id,
        Source.user_id == current_user.id,
        Source.is_active == True,
    )
    if not source:
        raise NotFoundError("Source not found")

    slot = KnowledgeSlot(
        user_id=current_user.id,
        source_id=source_id,
        name=body.name,
        description=body.description,
        content_sample=body.content_sample,
        destination=body.destination,
        tags=body.tags,
        include_subpages=body.include_subpages,
        read_content=body.read_content,
    )
    await slot.insert()
    current_user.usage.slots_count += 1

    # For Todoist project slots (not sections), also create child slots for all sections.
    # Section slots share the project slot's quota entry — they don't add to slots_count.
    extra_slots: list[KnowledgeSlot] = []
    if source.provider == "todoist" and " > " not in (body.destination.resource_name or ""):
        integration = await Integration.find_one(
            Integration.user_id == current_user.id,
            Integration.provider == "todoist",
            Integration.is_active == True,
        )
        if integration:
            from app.core.security import decrypt_token as _decrypt
            access_token = _decrypt(integration.tokens.access_token)
            extra_slots = await _build_todoist_section_slots(
                body.destination,
                body.name,
                source_id,
                current_user.id,
                slot.id,
                body.tags,
                body.read_content,
                access_token,
            )
            for s in extra_slots:
                await s.insert()

    await current_user.save()

    # Embedding + enrich always run in background so the response is fast
    user_provided_description = bool(body.description and body.description.strip() and body.description.strip() != body.name.strip())
    user_provided_tags = bool(body.tags)
    background_tasks.add_task(
        _embed_and_enrich_slot,
        str(slot.id),
        str(current_user.id),
        source.name,
        source.provider,
        user_provided_description,
        user_provided_tags,
    )
    for s in extra_slots:
        background_tasks.add_task(
            _embed_and_enrich_slot,
            str(s.id),
            str(current_user.id),
            source.name,
            source.provider,
            False,
            bool(body.tags),
        )

    return _slot_to_dict(slot)


@router.post("/bulk", status_code=201)
async def bulk_create_slots(
    body: SlotBulkCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Create multiple slots in one request (used by Trello to create one slot per list)."""
    for req in body.slots:
        if not req.read_content and not (req.description or "").strip():
            raise BadRequestError("Description is required when Read & index content is off.")
    available = current_user.limits.max_slots - current_user.usage.slots_count
    if not is_admin(current_user) and len(body.slots) > available:
        raise TierLimitError(
            f"Adding {len(body.slots)} slots would exceed your limit of {current_user.limits.max_slots}."
        )

    created = []
    for req in body.slots:
        source_id = PydanticObjectId(req.source_id)
        source = await Source.find_one(
            Source.id == source_id,
            Source.user_id == current_user.id,
            Source.is_active == True,
        )
        if not source:
            continue
        slot = KnowledgeSlot(
            user_id=current_user.id,
            source_id=source_id,
            name=req.name,
            description=req.description,
            content_sample=req.content_sample,
            destination=req.destination,
            tags=req.tags,
            read_content=req.read_content,
        )
        await slot.insert()
        current_user.usage.slots_count += 1
        user_provided_description = bool(
            req.description and req.description.strip() and req.description.strip() != req.name.strip()
        )
        background_tasks.add_task(
            _embed_and_enrich_slot,
            str(slot.id),
            str(current_user.id),
            source.name,
            source.provider,
            user_provided_description,
            bool(req.tags),
        )
        created.append(_slot_to_dict(slot))

    await current_user.save()
    return created


@router.get("/{slot_id}")
async def get_slot(
    slot_id: PydanticObjectId,
    current_user: User = Depends(get_current_user),
) -> dict:
    slot = await KnowledgeSlot.find_one(
        KnowledgeSlot.id == slot_id,
        KnowledgeSlot.user_id == current_user.id,
    )
    if not slot:
        raise NotFoundError("Slot not found")
    return _slot_to_dict(slot)


@router.patch("/{slot_id}")
async def update_slot(
    slot_id: PydanticObjectId,
    body: SlotUpdateRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    slot = await KnowledgeSlot.find_one(
        KnowledgeSlot.id == slot_id,
        KnowledgeSlot.user_id == current_user.id,
    )
    if not slot:
        raise NotFoundError("Slot not found")

    updates: dict = {KnowledgeSlot.updated_at: datetime.now(timezone.utc)}

    if body.source_id is not None:
        new_source_id = PydanticObjectId(body.source_id)
        source = await Source.find_one(
            Source.id == new_source_id,
            Source.user_id == current_user.id,
            Source.is_active == True,
        )
        if not source:
            raise NotFoundError("Source not found")
        updates[KnowledgeSlot.source_id] = new_source_id
        slot.source_id = new_source_id
    if body.name is not None:
        updates[KnowledgeSlot.name] = body.name
        slot.name = body.name
    if body.description is not None:
        updates[KnowledgeSlot.description] = body.description
        slot.description = body.description
    if body.content_sample is not None:
        updates[KnowledgeSlot.content_sample] = body.content_sample
        slot.content_sample = body.content_sample
    if body.tags is not None:
        updates[KnowledgeSlot.tags] = body.tags
        slot.tags = body.tags

    await slot.update(Set(updates))

    custom_index, custom_bedrock, _ = _resolve_custom_creds(current_user)
    try:
        source = await Source.get(slot.source_id)
        summary_vec, content_vecs = await _embed_slot(slot, source, custom_bedrock)
        if len(content_vecs) > 1:
            vector_svc.upsert_slot_content_chunks(slot, summary_vec, content_vecs, custom_index)
        else:
            vector_svc.upsert_slot(slot, summary_vec, content_vecs[0], custom_index)
        slot.index_status = "indexed"
        await slot.save()
    except Exception:
        logger.exception("vector upsert failed for slot %s", slot.id)
        slot.index_status = "failed"
        await slot.save()

    return _slot_to_dict(slot)


@router.post("/{slot_id}/reindex", status_code=202)
async def reindex_slot(
    slot_id: PydanticObjectId,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Re-read card/page content and re-embed a slot without deleting it.

    Useful when the user originally added a slot without 'Read & index content'
    and wants to upgrade it, or when the source content has changed significantly.
    """
    slot = await KnowledgeSlot.find_one(
        KnowledgeSlot.id == slot_id,
        KnowledgeSlot.user_id == current_user.id,
    )
    if not slot:
        raise NotFoundError("Slot not found")

    source = await Source.find_one(
        Source.id == slot.source_id,
        Source.user_id == current_user.id,
    )
    if not source:
        raise NotFoundError("Source not found")

    slot.index_status = "indexing"
    slot.read_content = True
    await slot.save()

    background_tasks.add_task(
        _embed_and_enrich_slot,
        str(slot.id),
        str(current_user.id),
        source.name,
        source.provider,
        bool(slot.description),
        bool(slot.tags),
    )
    return {"slot_id": str(slot.id), "status": "indexing"}


@router.delete("/{slot_id}", status_code=204)
async def delete_slot(
    slot_id: PydanticObjectId,
    current_user: User = Depends(get_current_user),
) -> None:
    slot = await KnowledgeSlot.find_one(
        KnowledgeSlot.id == slot_id,
        KnowledgeSlot.user_id == current_user.id,
    )
    if not slot:
        raise NotFoundError("Slot not found")

    slot_id_str = str(slot.id)

    # Cascade-delete child section slots (Todoist project slots own their sections)
    child_slots = await KnowledgeSlot.find(
        KnowledgeSlot.parent_slot_id == slot.id,
        KnowledgeSlot.user_id == current_user.id,
    ).to_list()
    for child in child_slots:
        child_id_str = str(child.id)
        await child.delete()
        try:
            child_delete_creds = vector_svc.resolve_delete_creds(child.index_name, child.index_api_key)
            vector_svc.delete_slot(child_id_str, child_delete_creds)
        except Exception:
            logger.exception("vector delete failed for child slot %s", child_id_str)

    await slot.delete()

    # Only decrement once — section slots don't count toward the quota
    if slot.parent_slot_id is None:
        current_user.usage.slots_count = max(0, current_user.usage.slots_count - 1)
    await current_user.save()

    try:
        delete_creds = vector_svc.resolve_delete_creds(slot.index_name, slot.index_api_key)
        vector_svc.delete_slot(slot_id_str, delete_creds)
    except Exception:
        logger.exception("vector delete failed for slot %s", slot_id_str)


async def _embed_slot(
    slot: KnowledgeSlot,
    source: Source | None = None,
    custom_bedrock: CustomBedrockCreds | None = None,
) -> tuple[list[float], list[list[float]]]:
    """Return (summary_vector, content_vectors).

    content_vectors is a list — one entry per chunk when raw_content is present,
    otherwise a single-element list using the summary/description text.
    """
    source_context = ""
    if source:
        tags_str = " ".join(source.tags) if source.tags else ""
        source_context = f"{source.name} {source.provider} {tags_str} | "

    summary_text = source_context + slot.name + ": " + slot.description

    if slot.raw_content:
        # Split raw content into overlapping chunks and embed each independently.
        # This prevents long lists from diluting minority topics into the overall topic mass.
        chunks = vector_svc._split_chunks(slot.raw_content)
        chunk_texts = [source_context + c for c in chunks]
        # Embed summary + all chunks concurrently
        results = await asyncio.gather(
            embed_text(summary_text, custom_bedrock),
            *[embed_text(ct, custom_bedrock) for ct in chunk_texts],
        )
        summary_vec = results[0]
        content_vecs = list(results[1:])
    else:
        content_text = source_context + (slot.content_sample or slot.description)
        summary_vec, single_content_vec = await asyncio.gather(
            embed_text(summary_text, custom_bedrock),
            embed_text(content_text, custom_bedrock),
        )
        content_vecs = [single_content_vec]

    return summary_vec, content_vecs
