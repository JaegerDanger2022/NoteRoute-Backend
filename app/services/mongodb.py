from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings

_client: AsyncIOMotorClient | None = None


async def init_db() -> None:
    global _client
    _client = AsyncIOMotorClient(settings.MONGODB_URL)
    # Import here to avoid circular imports at module load time
    from app.models.user import User
    from app.models.source import Source
    from app.models.slot import KnowledgeSlot
    from app.models.route import Route
    from app.models.integration import Integration
    from app.models.global_config import GlobalConfig

    await init_beanie(
        database=_client[settings.MONGODB_DB_NAME],
        document_models=[User, Source, KnowledgeSlot, Route, Integration, GlobalConfig],
    )


async def close_db() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
