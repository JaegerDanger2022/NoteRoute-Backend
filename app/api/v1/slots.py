import asyncio
import logging
from datetime import datetime, timezone

from beanie import PydanticObjectId
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.exceptions import NotFoundError, TierLimitError
from app.core.security import decrypt_token
from app.models.integration import Integration
from app.models.slot import KnowledgeSlot, SlotDestination
from app.models.source import Source
from app.models.user import User
from app.services import claude_svc, gdocs_svc, notion_svc, slack_svc, vector_svc
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
    read_content: bool = False


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
        "is_active": slot.is_active,
        "created_at": slot.created_at.isoformat(),
        "updated_at": slot.updated_at.isoformat(),
    }


@router.get("")
async def list_slots(
    source_id: str | None = None,
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    query = [KnowledgeSlot.user_id == current_user.id, KnowledgeSlot.is_active == True]
    if source_id:
        query.append(KnowledgeSlot.source_id == PydanticObjectId(source_id))
    slots = await KnowledgeSlot.find(*query).to_list()
    return [_slot_to_dict(s) for s in slots]


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
    resource_id = slot.destination.resource_id
    if provider == "notion":
        return await notion_svc.fetch_page_text(resource_id, access_token)
    elif provider == "google":
        return await gdocs_svc.fetch_document_text(resource_id, access_token)
    elif provider == "slack":
        return await slack_svc.fetch_channel_messages(resource_id, access_token)
    return ""


def _resolve_custom_creds(user: User) -> tuple[CustomIndexCreds | None, CustomBedrockCreds | None]:
    """Extract decrypted custom index + bedrock creds from the user if configured and ready."""
    cfg = user.custom_index
    if not cfg or cfg.index_status not in ("ready",):
        return None, None
    custom_index = CustomIndexCreds(
        pinecone_api_key=decrypt_token(cfg.pinecone_api_key),
        index_name=cfg.index_name,
    )
    custom_bedrock = None
    if cfg.bedrock_aws_access_key_id and cfg.bedrock_aws_secret_access_key:
        custom_bedrock = CustomBedrockCreds(
            aws_access_key_id=decrypt_token(cfg.bedrock_aws_access_key_id),
            aws_secret_access_key=decrypt_token(cfg.bedrock_aws_secret_access_key),
            aws_region=cfg.bedrock_aws_region or "us-east-1",
        )
    return custom_index, custom_bedrock


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

    # Resolve custom index/bedrock creds from user config
    user = await User.get(PydanticObjectId(user_id))
    custom_index, custom_bedrock = _resolve_custom_creds(user) if user else (None, None)

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
                summary = await claude_svc.summarize_slot_content(slot.name, raw_content)
                slot.content_sample = summary
                await slot.save()
                logger.info("Content indexed for slot %s", slot_id)
        except Exception:
            logger.exception("Content reading failed for slot %s", slot_id)

    # Step 2: auto-enrich description/tags with Haiku if not user-provided
    if not user_provided_description or not user_provided_tags:
        try:
            meta = await claude_svc.infer_slot_metadata(slot.name, source_name, provider)
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
        summary_vec, content_vec = await _embed_slot(slot, source, custom_bedrock)
        vector_svc.upsert_slot(slot, summary_vec, content_vec, custom_index)
        slot.index_status = "indexed"
        await slot.save()
        logger.info("Vector upsert complete for slot %s", slot_id)
    except Exception:
        logger.exception("Vector upsert failed for slot %s", slot_id)
        try:
            slot.index_status = "failed"
            await slot.save()
        except Exception:
            pass


@router.post("", status_code=201)
async def create_slot(
    body: SlotCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
) -> dict:
    if current_user.usage.slots_count >= current_user.limits.max_slots:
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
        read_content=body.read_content,
    )
    await slot.insert()

    current_user.usage.slots_count += 1
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

    return _slot_to_dict(slot)


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

    if body.source_id is not None:
        new_source_id = PydanticObjectId(body.source_id)
        source = await Source.find_one(
            Source.id == new_source_id,
            Source.user_id == current_user.id,
            Source.is_active == True,
        )
        if not source:
            raise NotFoundError("Source not found")
        slot.source_id = new_source_id
    if body.name is not None:
        slot.name = body.name
    if body.description is not None:
        slot.description = body.description
    if body.content_sample is not None:
        slot.content_sample = body.content_sample
    if body.tags is not None:
        slot.tags = body.tags

    slot.updated_at = datetime.now(timezone.utc)
    await slot.save()

    custom_index, custom_bedrock = _resolve_custom_creds(current_user)
    try:
        source = await Source.get(slot.source_id)
        summary_vec, content_vec = await _embed_slot(slot, source, custom_bedrock)
        vector_svc.upsert_slot(slot, summary_vec, content_vec, custom_index)
    except Exception:
        logger.exception("vector upsert failed for slot %s", slot.id)

    return _slot_to_dict(slot)


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

    await slot.delete()

    current_user.usage.slots_count = max(0, current_user.usage.slots_count - 1)
    await current_user.save()

    custom_index, _ = _resolve_custom_creds(current_user)
    try:
        vector_svc.delete_slot(slot_id_str, custom_index)
    except Exception:
        logger.exception("vector delete failed for slot %s", slot_id_str)


async def _embed_slot(
    slot: KnowledgeSlot,
    source: Source | None = None,
    custom_bedrock: CustomBedrockCreds | None = None,
) -> tuple[list[float], list[float]]:
    # Include source context in the embedding text for better routing
    source_context = ""
    if source:
        tags_str = " ".join(source.tags) if source.tags else ""
        source_context = f"{source.name} {source.provider} {tags_str} | "

    summary_text = source_context + slot.description
    content_text = source_context + (slot.content_sample or slot.description)

    summary_vec, content_vec = await asyncio.gather(
        embed_text(summary_text, custom_bedrock),
        embed_text(content_text, custom_bedrock),
    )
    return summary_vec, content_vec
