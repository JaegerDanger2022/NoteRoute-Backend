from beanie import Document


class GlobalConfig(Document):
    use_nova: bool = False

    class Settings:
        name = "global_config"

    @classmethod
    async def get_config(cls) -> "GlobalConfig":
        """Return the singleton config, creating it with defaults if it doesn't exist."""
        config = await cls.find_one()
        if config is None:
            config = cls()
            await config.insert()
        return config
