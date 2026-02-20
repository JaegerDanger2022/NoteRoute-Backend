import json
import logging
import uuid
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)
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
    doc_title: str | None = None


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
        # Emit init event with run_id immediately
        yield f"data: {json.dumps({'run_id': run_id, 'node': 'init'})}\n\n"
        try:
            # Call LangGraph non-streaming /run endpoint — avoids chunked proxy issues
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{settings.langgraph_url}/run",
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
            # Emit events that the frontend expects
            if result.get("transcript"):
                yield f"event: transcribe\ndata: {json.dumps({'node': 'transcribe', 'transcript': result['transcript']})}\n\n"
            # Always emit rank event (even empty list) so frontend can transition out of "searching"
            ranked_slots = result.get("ranked_slots") or []
            if result.get("error"):
                yield f"data: {json.dumps({'node': 'error', 'error': result['error']})}\n\n"
            else:
                yield f"data: {json.dumps({'node': 'rank', 'ranked_slots': ranked_slots, 'summary': result.get('summary')})}\n\n"
        except Exception as e:
            logger.error("Pipeline stream failed: %s", e)
            yield f"data: {json.dumps({'node': 'error', 'error': str(e)})}\n\n"

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
        try:
            # Call LangGraph non-streaming /run endpoint — avoids chunked proxy issues
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{settings.langgraph_url}/run/text",
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
            ranked_slots = result.get("ranked_slots") or []
            if result.get("error"):
                yield f"data: {json.dumps({'node': 'error', 'error': result['error']})}\n\n"
            else:
                yield f"data: {json.dumps({'node': 'rank', 'ranked_slots': ranked_slots, 'summary': result.get('summary')})}\n\n"
        except Exception as e:
            logger.error("Text pipeline stream failed: %s", e)
            yield f"data: {json.dumps({'node': 'error', 'error': str(e)})}\n\n"

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
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{settings.langgraph_url}/confirm",
            json={
                "run_id": body.run_id,
                "confirmed_slot_id": body.confirmed_slot_id,
                "save_as_slot": body.save_as_slot,
                "target_tab_id": body.target_tab_id,
                "doc_title": body.doc_title,
            },
        )
        resp.raise_for_status()
        lg_result = resp.json()

    logger.info("LangGraph /confirm response: %s", lg_result)

    _valid_statuses = {"processing", "awaiting_confirmation", "delivered", "failed", "rejected"}
    delivery_status = lg_result.get("status", "failed")
    if delivery_status not in _valid_statuses:
        logger.warning("Unexpected delivery_status from LangGraph: %r — mapping to failed", delivery_status)
        delivery_status = "failed"

    # Update route record
    if body.confirmed_slot_id:
        from beanie import PydanticObjectId
        route.confirmed_slot_id = PydanticObjectId(body.confirmed_slot_id)
    if body.transcript:
        route.transcript = body.transcript
    route.status = delivery_status
    route.events.append(RouteEvent(
        event_type="confirmed",
        metadata={
            "confirmed_slot_id": body.confirmed_slot_id,
            "save_as_slot": body.save_as_slot,
            "delivery_status": delivery_status,
        },
    ))
    await route.save()

    return {
        "run_id": body.run_id,
        "delivery_status": delivery_status,
        "delivery_error": lg_result.get("delivery_error"),
        "delivered_at": lg_result.get("delivered_at"),
    }
