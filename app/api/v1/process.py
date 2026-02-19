import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.config import settings
from app.core.exceptions import RateLimitError, NotFoundError
from app.models.route import Route, RouteEvent
from app.models.user import User
from app.services.langgraph_client import run_pipeline

router = APIRouter(prefix="/process", tags=["process"])


class ProcessRequest(BaseModel):
    audio_s3_key: str
    audio_duration_sec: float = 0.0


class ProcessStreamRequest(BaseModel):
    audio_s3_key: str
    audio_duration_sec: float = 0.0


class ConfirmRequest(BaseModel):
    run_id: str
    confirmed_slot_id: str | None = None
    save_as_slot: bool = False


@router.post("")
async def process_voice_note(
    body: ProcessRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Kick off the LangGraph pipeline for a voice note that was uploaded to S3."""
    if current_user.usage.routes_this_month >= current_user.limits.max_routes_per_month:
        raise RateLimitError()

    run_id = str(uuid.uuid4())

    route = Route(
        user_id=current_user.id,
        run_id=run_id,
        audio_s3_key=body.audio_s3_key,
        status="processing",
    )
    await route.insert()

    current_user.usage.routes_this_month += 1
    await current_user.save()

    result = await run_pipeline(
        run_id=run_id,
        user_id=str(current_user.id),
        audio_s3_key=body.audio_s3_key,
        audio_duration_sec=body.audio_duration_sec,
        source_id=str(current_user.active_source_id) if current_user.active_source_id else None,
    )

    return {
        "run_id": run_id,
        "route_id": str(route.id),
        "status": result.get("status", "processing"),
    }


@router.post("/stream")
async def process_stream(
    body: ProcessStreamRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Start a streaming pipeline run and proxy the SSE stream from LangGraph."""
    if current_user.usage.routes_this_month >= current_user.limits.max_routes_per_month:
        raise RateLimitError()

    if not current_user.active_source_id:
        raise NotFoundError("No active source selected. Select a source before recording.")

    run_id = str(uuid.uuid4())

    route = Route(
        user_id=current_user.id,
        run_id=run_id,
        audio_s3_key=body.audio_s3_key,
        status="processing",
    )
    await route.insert()

    current_user.usage.routes_this_month += 1
    await current_user.save()

    payload = {
        "run_id": run_id,
        "user_id": str(current_user.id),
        "audio_s3_key": body.audio_s3_key,
        "audio_duration_sec": body.audio_duration_sec,
        "source_id": str(current_user.active_source_id),
    }

    async def _stream():
        # Send the run_id to the client first so it knows which run this is
        yield f"data: {{'run_id': '{run_id}', 'node': 'init'}}\n\n"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{settings.LANGGRAPH_INTERNAL_URL}/stream",
                json=payload,
            ) as response:
                async for chunk in response.aiter_text():
                    if chunk:
                        yield chunk

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/confirm")
async def confirm_slot(
    body: ConfirmRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Confirm a slot selection (or save-as-slot) and resume the LangGraph pipeline."""
    route = await Route.find_one(
        Route.run_id == body.run_id,
        Route.user_id == current_user.id,
    )
    if not route:
        raise NotFoundError("Route not found")

    # Forward to LangGraph to resume the interrupted graph
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.LANGGRAPH_INTERNAL_URL}/confirm",
            json={
                "run_id": body.run_id,
                "confirmed_slot_id": body.confirmed_slot_id,
                "save_as_slot": body.save_as_slot,
            },
        )
        resp.raise_for_status()

    # Update route record
    if body.confirmed_slot_id:
        from beanie import PydanticObjectId
        route.confirmed_slot_id = PydanticObjectId(body.confirmed_slot_id)
    route.status = "awaiting_confirmation"
    route.events.append(RouteEvent(
        event_type="confirmed",
        metadata={
            "confirmed_slot_id": body.confirmed_slot_id,
            "save_as_slot": body.save_as_slot,
        },
    ))
    await route.save()

    return {"status": "confirmed", "run_id": body.run_id}
