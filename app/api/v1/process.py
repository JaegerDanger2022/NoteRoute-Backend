import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.exceptions import RateLimitError
from app.models.route import Route
from app.models.user import User
from app.services.langgraph_client import run_pipeline

router = APIRouter(prefix="/process", tags=["process"])


class ProcessRequest(BaseModel):
    audio_s3_key: str
    audio_duration_sec: float = 0.0


@router.post("")
async def process_voice_note(
    body: ProcessRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Kick off the LangGraph pipeline for a voice note that was uploaded to S3."""
    # Enforce monthly route limit
    if current_user.usage.routes_this_month >= current_user.limits.max_routes_per_month:
        raise RateLimitError()

    run_id = str(uuid.uuid4())

    # Create route record before dispatching so it's visible immediately
    route = Route(
        user_id=current_user.id,
        run_id=run_id,
        audio_s3_key=body.audio_s3_key,
        status="processing",
    )
    await route.insert()

    # Increment usage counter
    current_user.usage.routes_this_month += 1
    await current_user.save()

    # Dispatch to LangGraph (non-blocking — returns quickly with run_id)
    result = await run_pipeline(
        run_id=run_id,
        user_id=str(current_user.id),
        audio_s3_key=body.audio_s3_key,
        audio_duration_sec=body.audio_duration_sec,
    )

    return {
        "run_id": run_id,
        "route_id": str(route.id),
        "status": result.get("status", "processing"),
    }
