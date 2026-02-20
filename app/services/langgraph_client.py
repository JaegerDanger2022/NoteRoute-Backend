import httpx

from app.config import settings


async def run_pipeline(
    run_id: str,
    user_id: str,
    audio_s3_key: str,
    audio_duration_sec: float,
    source_id: str | None = None,
) -> dict:
    """Send a pipeline request to the LangGraph service and return its response."""
    payload = {
        "run_id": run_id,
        "user_id": user_id,
        "audio_s3_key": audio_s3_key,
        "audio_duration_sec": audio_duration_sec,
        "source_id": source_id,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.langgraph_url}/run",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
