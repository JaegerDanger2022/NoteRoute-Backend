from datetime import datetime, timezone
from typing import Literal

from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING
from pydantic import BaseModel, Field


class OAuthTokens(BaseModel):
    access_token: str        # stored encrypted via Fernet
    refresh_token: str | None = None
    expires_at: datetime | None = None
    scopes: list[str] = Field(default_factory=list)


class Integration(Document):
    user_id: PydanticObjectId
    provider: Literal["notion", "google", "slack", "todoist"]
    tokens: OAuthTokens
    provider_user_id: str
    provider_email: str | None = None
    workspace_name: str | None = None
    is_active: bool = True
    connected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: datetime | None = None

    class Settings:
        name = "integrations"
        indexes = [
            IndexModel(
                [("user_id", ASCENDING), ("provider", ASCENDING)],
                unique=True,
            ),
        ]
