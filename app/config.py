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

    # AWS
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    AWS_TRANSCRIBE_BUCKET: str

    # Bedrock models
    BEDROCK_EMBED_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
    CLAUDE_MODEL_ID: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "noteroute-shared"

    # Encryption (Fernet key for OAuth token storage)
    ENCRYPTION_KEY: str = ""

    # Internal service URLs — must NOT have a trailing slash
    LANGGRAPH_INTERNAL_URL: str = "http://noteroute-langgraph.railway.internal:8001"

    @property
    def langgraph_url(self) -> str:
        """Return LANGGRAPH_INTERNAL_URL with any trailing slash stripped."""
        return self.LANGGRAPH_INTERNAL_URL.rstrip("/")

    # Notion — internal integration token
    NOTION_INTEGRATION_TOKEN: str = ""

    # OAuth — Google
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = ""

    # OAuth — Slack
    SLACK_CLIENT_ID: str = ""
    SLACK_CLIENT_SECRET: str = ""
    SLACK_REDIRECT_URI: str = ""

    # OAuth — Todoist
    TODOIST_CLIENT_ID: str = ""
    TODOIST_CLIENT_SECRET: str = ""

    # OAuth — Trello (return_url points to WEB_APP_URL/oauth/trello-callback, not a backend route)
    TRELLO_API_KEY: str = ""

    # CORS (comma-separated list)
    ALLOWED_ORIGINS: str = "http://localhost:8081,http://localhost:3000"

    # Web app URL (used for OAuth callbacks from the Next.js frontend)
    WEB_APP_URL: str = "http://localhost:3000"

    # Admin — comma-separated Firebase UIDs allowed to access /admin endpoints
    ADMIN_FIREBASE_UIDS: str = ""

    # RevenueCat — webhook secret for server-side event verification
    REVENUECAT_WEBHOOK_SECRET: str = ""


settings = Settings()
