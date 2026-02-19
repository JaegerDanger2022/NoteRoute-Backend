import asyncio
import logging
from datetime import datetime, timezone

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.exceptions import NotFoundError, TierLimitError
from app.models.slot import KnowledgeSlot, SlotDestination
from app.models.source import Source
from app.models.user import User
from app.services import vector_svc
from app.utils.embeddings import embed_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slots", tags=["slots"])


class SlotCreateRequest(BaseModel):
    source_id: str
    name: str
    description: str
    content_sample: str = ""
    destination: SlotDestination
    tags: list[str] = []


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


@router.post("", status_code=201)
async def create_slot(
    body: SlotCreateRequest,
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
    )
    await slot.insert()

    current_user.usage.slots_count += 1
    await current_user.save()

    try:
        summary_vec, content_vec = await _embed_slot(slot, source)
        vector_svc.upsert_slot(slot, summary_vec, content_vec)
    except Exception:
        logger.exception("vector upsert failed for slot %s", slot.id)

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

    try:
        source = await Source.get(slot.source_id)
        summary_vec, content_vec = await _embed_slot(slot, source)
        vector_svc.upsert_slot(slot, summary_vec, content_vec)
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

    slot.is_active = False
    slot.updated_at = datetime.now(timezone.utc)
    await slot.save()

    current_user.usage.slots_count = max(0, current_user.usage.slots_count - 1)
    await current_user.save()

    try:
        vector_svc.delete_slot(str(slot.id))
    except Exception:
        logger.exception("vector delete failed for slot %s", slot.id)


async def _embed_slot(slot: KnowledgeSlot, source: Source | None = None) -> tuple[list[float], list[float]]:
    # Include source context in the embedding text for better routing
    source_context = ""
    if source:
        tags_str = " ".join(source.tags) if source.tags else ""
        source_context = f"{source.name} {source.provider} {tags_str} | "

    summary_text = source_context + slot.description
    content_text = source_context + (slot.content_sample or slot.description)

    summary_vec, content_vec = await asyncio.gather(
        embed_text(summary_text),
        embed_text(content_text),
    )
    return summary_vec, content_vec
