from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.core.exceptions import NotFoundError
from app.core.security import decrypt_token
from app.config import settings
from app.models.integration import Integration
from app.models.user import User
from app.services import trello_svc

router = APIRouter(prefix="/trello", tags=["trello"])


async def _get_trello_token(user: User) -> str:
    integration = await Integration.find_one(
        Integration.user_id == user.id,
        Integration.provider == "trello",
        Integration.is_active == True,
    )
    if not integration:
        raise NotFoundError("No active Trello integration")
    return decrypt_token(integration.tokens.access_token)


@router.get("/{list_id}/cards")
async def get_list_cards(
    list_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return cards in a Trello list for the card picker UI."""
    token = await _get_trello_token(current_user)
    cards = await trello_svc.list_cards_for_picker(list_id, settings.TRELLO_API_KEY, token)
    return {"cards": cards}
