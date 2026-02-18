from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    APP_ENV: str = "development"
    DEBUG: bool = False

    # MongoDB
    MONGODB_URL: str
    MONGODB_DB_NAME: str = "noteroute"

    # Firebase
    FIREBASE_PROJECT_ID: str
    FIREBASE_SERVICE_ACCOUNT_JSON: str  # full JSON string as single env var

    # Postgres (pgvector)
    POSTGRES_URL: str

    # AWS
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    AWS_TRANSCRIBE_BUCKET: str

    # Bedrock models
    BEDROCK_EMBED_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
    CLAUDE_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # Encryption (Fernet key for OAuth token storage)
    ENCRYPTION_KEY: str

    # Internal service URLs
    LANGGRAPH_INTERNAL_URL: str = "http://noteroute-langgraph.railway.internal:8001"

    # OAuth — Notion
    NOTION_CLIENT_ID: str
    NOTION_CLIENT_SECRET: str
    NOTION_REDIRECT_URI: str

    # OAuth — Google
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str

    # OAuth — Slack
    SLACK_CLIENT_ID: str
    SLACK_CLIENT_SECRET: str
    SLACK_REDIRECT_URI: str

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:8081"


settings = Settings()
