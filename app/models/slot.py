from datetime import datetime, timezone
from typing import Literal

from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING
from pydantic import BaseModel, Field


class SlotDestination(BaseModel):
    resource_id: str
    resource_name: str
    resource_url: str | None = None


class KnowledgeSlot(Document):
    user_id: PydanticObjectId
    source_id: PydanticObjectId
    parent_slot_id: PydanticObjectId | None = None  # set on section slots owned by a project slot
    name: str
    description: str
    content_sample: str = ""   # Claude-generated summary of resource content (displayed in UI)
    raw_content: str = ""      # Raw fetched text used for content_vector embedding
    destination: SlotDestination
    tags: list[str] = []
    read_content: bool = False
    index_status: Literal["pending", "indexing", "indexed", "failed"] = "pending"
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "knowledge_slots"
        indexes = [
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("source_id", ASCENDING)]),
            IndexModel([("user_id", ASCENDING), ("is_active", ASCENDING)]),
            IndexModel([("source_id", ASCENDING), ("is_active", ASCENDING)]),
        ]
