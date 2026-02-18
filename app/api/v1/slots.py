import logging
from datetime import datetime, timezone

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.exceptions import NotFoundError, TierLimitError
from app.models.slot import KnowledgeSlot, SlotDestination
from app.models.user import User
from app.services import pgvector_svc
from app.utils.embeddings import embed_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slots", tags=["slots"])


class SlotCreateRequest(BaseModel):
    name: str
    description: str
    content_sample: str = ""
    destination: SlotDestination
    tags: list[str] = []


class SlotUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    content_sample: str | None = None
    tags: list[str] | None = None


def _slot_to_dict(slot: KnowledgeSlot) -> dict:
    return {
        "id": str(slot.id),
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
async def list_slots(current_user: User = Depends(get_current_user)) -> list[dict]:
    slots = await KnowledgeSlot.find(
        KnowledgeSlot.user_id == current_user.id,
        KnowledgeSlot.is_active == True,
    ).to_list()
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

    slot = KnowledgeSlot(
        user_id=current_user.id,
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
        summary_vec, content_vec = await _embed_slot(slot)
        await pgvector_svc.upsert_slot(slot, summary_vec, content_vec)
    except Exception:
        logger.exception("pgvector upsert failed for slot %s", slot.id)

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
        summary_vec, content_vec = await _embed_slot(slot)
        await pgvector_svc.upsert_slot(slot, summary_vec, content_vec)
    except Exception:
        logger.exception("pgvector upsert failed for slot %s", slot.id)

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
        await pgvector_svc.delete_slot(str(slot.id), str(current_user.id))
    except Exception:
        logger.exception("pgvector delete failed for slot %s", slot.id)


async def _embed_slot(slot: KnowledgeSlot) -> tuple[list[float], list[float]]:
    summary_text = slot.description
    content_text = slot.content_sample or slot.description
    summary_vec, content_vec = await _embed_pair(summary_text, content_text)
    return summary_vec, content_vec


async def _embed_pair(summary_text: str, content_text: str) -> tuple[list[float], list[float]]:
    import asyncio
    summary_vec, content_vec = await asyncio.gather(
        embed_text(summary_text),
        embed_text(content_text),
    )
    return summary_vec, content_vec
