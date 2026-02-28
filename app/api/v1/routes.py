from datetime import datetime, timezone

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user
from app.core.exceptions import BadRequestError, NotFoundError
from app.models.route import Route, RouteEvent
from app.models.slot import KnowledgeSlot
from app.models.user import User

router = APIRouter(prefix="/routes", tags=["routes"])


def _route_to_dict(route: Route, slot_name: str | None = None) -> dict:
    # Prefer the live-lookup name (from KnowledgeSlot), fall back to cached value on route
    resolved_slot_name = slot_name if slot_name is not None else (route.slot_name or "")
    return {
        "id": str(route.id),
        "run_id": route.run_id,
        "delivery_status": route.status,
        "transcript": route.transcript,
        "summary": route.summary,
        "slot_name": resolved_slot_name,
        "confirmed_slot_id": str(route.confirmed_slot_id) if route.confirmed_slot_id else None,
        "created_at": route.created_at.isoformat(),
        "completed_at": route.completed_at.isoformat() if route.completed_at else None,
        "delivery_url": route.delivery_url,
        "retry_count": route.retry_count,
    }


_TERMINAL_STATUSES = {"delivered", "failed", "rejected"}


@router.get("")
async def list_routes(
    current_user: User = Depends(get_current_user),
    status: str | None = Query(default=None, description="Filter by status: delivered, failed, rejected"),
    q: str | None = Query(default=None, description="Search transcript text"),
) -> list[dict]:
    filters: list = [
        Route.user_id == current_user.id,
        {"status": {"$in": list(_TERMINAL_STATUSES)}},
    ]
    if status and status in _TERMINAL_STATUSES:
        filters[1] = {"status": status}
    if q:
        filters.append({"transcript": {"$regex": q, "$options": "i"}})

    routes = await Route.find(*filters).sort("-created_at").limit(100).to_list()

    # Batch-fetch slot names for routes that have a confirmed slot
    slot_ids = [r.confirmed_slot_id for r in routes if r.confirmed_slot_id]
    slot_map: dict = {}
    if slot_ids:
        slots = await KnowledgeSlot.find({"_id": {"$in": slot_ids}}).to_list()
        slot_map = {str(s.id): s.name for s in slots}

    results = [_route_to_dict(r, slot_map.get(str(r.confirmed_slot_id))) for r in routes]

    # Client-side slot name filter applied after name resolution
    if q:
        q_lower = q.lower()
        results = [
            r for r in results
            if q_lower in (r["transcript"] or "").lower() or q_lower in r["slot_name"].lower()
        ]

    return results


@router.get("/{route_id}")
async def get_route(
    route_id: PydanticObjectId,
    current_user: User = Depends(get_current_user),
) -> dict:
    route = await Route.find_one(
        Route.id == route_id,
        Route.user_id == current_user.id,
    )
    if not route:
        raise NotFoundError("Route not found")
    return _route_to_dict(route)


@router.post("/{route_id}/retry")
async def retry_route(
    route_id: PydanticObjectId,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Re-attempt delivery for a failed route using the same confirmed slot and transcript."""
    route = await Route.find_one(
        Route.id == route_id,
        Route.user_id == current_user.id,
    )
    if not route:
        raise NotFoundError("Route not found")
    if route.status != "failed":
        raise BadRequestError("Only failed routes can be retried")
    if not route.confirmed_slot_id or not route.transcript:
        raise BadRequestError("Route is missing slot or transcript — cannot retry")

    # Import here to avoid circular imports
    from app.api.v1.deliver import _deliver_to_slot

    now = datetime.now(timezone.utc)
    try:
        result = await _deliver_to_slot(
            str(route.confirmed_slot_id),
            route.transcript,
            current_user,
        )
        route.status = "delivered"
        route.completed_at = now
        route.delivery_url = result.get("resource_url")
        route.retry_count += 1
        route.events.append(RouteEvent(
            event_type="retried",
            metadata={"attempt": route.retry_count, **result},
        ))
        route.events.append(RouteEvent(
            event_type="delivered",
            metadata=result,
        ))
        await route.save()
        return {"status": "delivered", "retry_count": route.retry_count, **result}

    except Exception as exc:
        route.retry_count += 1
        route.events.append(RouteEvent(
            event_type="retried",
            metadata={"attempt": route.retry_count, "error": str(exc)},
        ))
        await route.save()
        raise


@router.delete("/{route_id}", status_code=204)
async def delete_route(
    route_id: PydanticObjectId,
    current_user: User = Depends(get_current_user),
) -> None:
    """Permanently delete a history entry for the current user."""
    route = await Route.find_one(
        Route.id == route_id,
        Route.user_id == current_user.id,
    )
    if not route:
        raise NotFoundError("Route not found")
    await route.delete()


@router.get("/{route_id}/history")
async def get_route_history(
    route_id: PydanticObjectId,
    current_user: User = Depends(get_current_user),
) -> dict:
    route = await Route.find_one(
        Route.id == route_id,
        Route.user_id == current_user.id,
    )
    if not route:
        raise NotFoundError("Route not found")
    return {
        "run_id": route.run_id,
        "events": [
            {
                "event_type": e.event_type,
                "timestamp": e.timestamp.isoformat(),
                "metadata": e.metadata,
            }
            for e in route.events
        ],
    }
