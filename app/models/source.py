from datetime import datetime, timezone
from typing import Literal

from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING
from pydantic import Field


class Source(Document):
    user_id: PydanticObjectId
    provider: Literal["notion", "google", "slack"]
    name: str
    tags: list[str] = []
    connected_account_email: str | None = None
    connected_account_id: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "sources"
        indexes = [
            IndexModel([("user_id", ASCENDING)]),
            IndexModel(
                [("user_id", ASCENDING), ("provider", ASCENDING)],
                unique=True,
            ),
        ]
