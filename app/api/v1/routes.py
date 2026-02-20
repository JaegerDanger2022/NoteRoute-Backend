from beanie import PydanticObjectId
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.core.exceptions import NotFoundError
from app.models.route import Route
from app.models.slot import KnowledgeSlot
from app.models.user import User

router = APIRouter(prefix="/routes", tags=["routes"])


def _route_to_dict(route: Route, slot_name: str | None = None) -> dict:
    return {
        "id": str(route.id),
        "run_id": route.run_id,
        "delivery_status": route.status,
        "transcript": route.transcript,
        "summary": route.summary,
        "slot_name": slot_name or "",
        "confirmed_slot_id": str(route.confirmed_slot_id) if route.confirmed_slot_id else None,
        "created_at": route.created_at.isoformat(),
        "completed_at": route.completed_at.isoformat() if route.completed_at else None,
    }


@router.get("")
async def list_routes(current_user: User = Depends(get_current_user)) -> list[dict]:
    routes = await Route.find(Route.user_id == current_user.id).sort("-created_at").limit(50).to_list()

    # Batch-fetch slot names for routes that have a confirmed slot
    slot_ids = [r.confirmed_slot_id for r in routes if r.confirmed_slot_id]
    slot_map: dict = {}
    if slot_ids:
        slots = await KnowledgeSlot.find({"_id": {"$in": slot_ids}}).to_list()
        slot_map = {str(s.id): s.name for s in slots}

    return [_route_to_dict(r, slot_map.get(str(r.confirmed_slot_id))) for r in routes]


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
