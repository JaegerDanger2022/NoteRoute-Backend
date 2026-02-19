from datetime import datetime, timezone
from typing import Literal

from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING
from pydantic import BaseModel, Field


class TierLimits(BaseModel):
    max_sources: int = 3
    max_slots: int = 50
    max_routes_per_month: int = 10


class UsageCounters(BaseModel):
    routes_this_month: int = 0
    slots_count: int = 0
    sources_count: int = 0
    period_reset_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class User(Document):
    firebase_uid: str
    email: str
    display_name: str | None = None
    tier: Literal["free", "pro", "power"] = "free"
    limits: TierLimits = Field(default_factory=TierLimits)
    usage: UsageCounters = Field(default_factory=UsageCounters)
    active_source_id: PydanticObjectId | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "user"
        indexes = [
            IndexModel([("firebase_uid", ASCENDING)], unique=True),
            IndexModel([("email", ASCENDING)]),
        ]
