import json
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.config import settings
from app.core.exceptions import NotFoundError
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


class ProcessTextStreamRequest(BaseModel):
    text: str


class ConfirmRequest(BaseModel):
    run_id: str
    confirmed_slot_id: str | None = None
    save_as_slot: bool = False
    target_tab_id: str | None = None
    transcript: str | None = None


@router.post("")
async def process_voice_note(
    body: ProcessRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Kick off the LangGraph pipeline for a voice note that was uploaded to S3."""
    run_id = str(uuid.uuid4())

    route = Route(
        user_id=current_user.id,
        run_id=run_id,
        audio_s3_key=body.audio_s3_key,
        status="processing",
    )
    await route.insert()

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

    payload = {
        "run_id": run_id,
        "user_id": str(current_user.id),
        "audio_s3_key": body.audio_s3_key,
        "audio_duration_sec": body.audio_duration_sec,
        "source_id": str(current_user.active_source_id),
    }

    async def _stream():
        # Send the run_id to the client first so it knows which run this is
        yield f"data: {json.dumps({'run_id': run_id, 'node': 'init'})}\n\n"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{settings.langgraph_url}/stream",
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


@router.post("/text-stream")
async def process_text_stream(
    body: ProcessTextStreamRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Start a streaming pipeline run from raw text (skips transcription)."""
    if not current_user.active_source_id:
        raise NotFoundError("No active source selected. Select a source before submitting.")

    run_id = str(uuid.uuid4())

    route = Route(
        user_id=current_user.id,
        run_id=run_id,
        audio_s3_key="",
        status="processing",
    )
    await route.insert()

    payload = {
        "run_id": run_id,
        "user_id": str(current_user.id),
        "text": body.text,
        "source_id": str(current_user.active_source_id),
    }

    async def _stream():
        yield f"data: {json.dumps({'run_id': run_id, 'node': 'init'})}\n\n"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{settings.langgraph_url}/stream/text",
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
            f"{settings.langgraph_url}/confirm",
            json={
                "run_id": body.run_id,
                "confirmed_slot_id": body.confirmed_slot_id,
                "save_as_slot": body.save_as_slot,
                "target_tab_id": body.target_tab_id,
            },
        )
        resp.raise_for_status()

    # Update route record
    if body.confirmed_slot_id:
        from beanie import PydanticObjectId
        route.confirmed_slot_id = PydanticObjectId(body.confirmed_slot_id)
    if body.transcript:
        route.transcript = body.transcript
    route.status = "processing"
    route.events.append(RouteEvent(
        event_type="confirmed",
        metadata={
            "confirmed_slot_id": body.confirmed_slot_id,
            "save_as_slot": body.save_as_slot,
        },
    ))
    await route.save()

    return {"status": "confirmed", "run_id": body.run_id}
