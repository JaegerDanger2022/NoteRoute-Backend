from datetime import datetime, timezone

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
    name: str
    description: str
    content_sample: str = ""
    destination: SlotDestination
    tags: list[str] = []
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
