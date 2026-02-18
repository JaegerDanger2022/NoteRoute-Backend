from datetime import datetime, timezone
from typing import Literal

from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING
from pydantic import BaseModel, Field


class RouteEvent(BaseModel):
    event_type: Literal[
        "transcribed", "searched", "ranked", "confirmed",
        "delivered", "failed", "rejected"
    ]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


class Route(Document):
    user_id: PydanticObjectId
    run_id: str
    audio_s3_key: str
    transcript: str | None = None
    summary: str | None = None
    confirmed_slot_id: PydanticObjectId | None = None
    status: Literal[
        "processing", "awaiting_confirmation", "delivered", "failed", "rejected"
    ] = "processing"
    events: list[RouteEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    class Settings:
        name = "routes"
        indexes = [
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("run_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING), ("status", ASCENDING)]),
            IndexModel([("created_at", ASCENDING)]),
        ]
