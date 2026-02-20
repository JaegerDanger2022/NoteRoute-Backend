from datetime import datetime, timezone
from typing import Literal, Union

from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING
from pydantic import BaseModel, Field


class RouteEvent(BaseModel):
    event_type: Literal[
        "transcribed", "searched", "ranked", "confirmed",
        "delivered", "failed", "rejected", "retried"
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
    status: Union[
        Literal["processing", "awaiting_confirmation", "delivered", "failed", "rejected"],
        str  # fallback for legacy documents with unexpected status values
    ] = "processing"
    events: list[RouteEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    delivery_url: str | None = None  # URL to the resource where content was delivered
    slot_name: str | None = None     # Cached at delivery time so history always shows it
    retry_count: int = 0

    class Settings:
        name = "routes"
        indexes = [
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("run_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING), ("status", ASCENDING)]),
            IndexModel([("created_at", ASCENDING)]),
        ]
