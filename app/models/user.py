from datetime import datetime, timezone
from typing import Literal, Optional

from beanie import Document, PydanticObjectId
from pymongo import IndexModel, ASCENDING
from pydantic import BaseModel, Field


class TierLimits(BaseModel):
    max_sources: int = 1
    max_slots: int = 20
    max_image_inputs_per_month: int = 3


class UsageCounters(BaseModel):
    slots_count: int = 0
    sources_count: int = 0
    image_inputs_this_month: int = 0
    image_inputs_month: str = ""  # "YYYY-MM" — resets when month changes


class CustomIndexConfig(BaseModel):
    """BYOI — user-supplied Pinecone index and optional Bedrock credentials."""
    pinecone_api_key: str           # stored encrypted via Fernet
    index_name: str                 # auto-set to "noteroute-{user_id_short}"
    index_status: Literal["provisioning", "ready", "deleted", "error"] = "provisioning"
    provisioned_at: datetime | None = None
    # Optional BYOLLM: user's own AWS Bedrock credentials
    bedrock_aws_access_key_id: str | None = None     # encrypted
    bedrock_aws_secret_access_key: str | None = None  # encrypted
    bedrock_aws_region: str | None = None


class CustomLLMConfig(BaseModel):
    """BYOLLM — user-supplied OpenAI or Anthropic direct API key."""
    provider: Literal["openai", "anthropic"]
    api_key: str  # stored encrypted via Fernet


class User(Document):
    firebase_uid: str
    email: str
    display_name: str | None = None
    tier: Literal["free", "pro"] = "free"
    limits: TierLimits = Field(default_factory=TierLimits)
    usage: UsageCounters = Field(default_factory=UsageCounters)
    polar_subscription_id: str | None = None
    polar_interval: str | None = None  # monthly | quarterly | biannually | annually
    polar_cancel_at: datetime | None = None  # set when cancelled; downgrade happens at this date
    active_source_id: PydanticObjectId | None = None
    custom_index: CustomIndexConfig | None = None
    custom_llm: Optional[CustomLLMConfig] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "user"
        indexes = [
            IndexModel([("firebase_uid", ASCENDING)], unique=True),
            IndexModel([("email", ASCENDING)]),
        ]
