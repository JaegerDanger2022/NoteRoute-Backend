import hashlib
import hmac
import logging

from beanie.operators import Set
from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.core.auth import get_current_user
from app.models.user import TierLimits, User
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])

# Tier limit definitions
_TIER_LIMITS: dict[str, TierLimits] = {
    "free": TierLimits(max_sources=1, max_slots=20, max_image_inputs_per_month=3),
    "pro": TierLimits(max_sources=20, max_slots=500, max_image_inputs_per_month=500),
}

# Map RevenueCat product identifiers → internal tier
# Set these to match your RevenueCat product IDs exactly.
_PRODUCT_TIER: dict[str, str] = {
    "noteroute_pro_monthly": "pro",
    "noteroute_pro_annual": "pro",
}


def _verify_signature(body: bytes, auth_header: str) -> bool:
    """Verify RevenueCat webhook HMAC-SHA256 signature."""
    if not settings.REVENUECAT_WEBHOOK_SECRET:
        # Secret not configured — skip verification (dev only)
        logger.warning("REVENUECAT_WEBHOOK_SECRET not set — skipping signature verification")
        return True
    expected = hmac.new(
        settings.REVENUECAT_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, auth_header)


async def _set_tier(user: User, tier: str) -> None:
    limits = _TIER_LIMITS.get(tier, _TIER_LIMITS["free"])
    await user.update(Set({
        User.tier: tier,
        User.limits: limits,
    }))
    logger.info("Updated user %s to tier=%s limits=%s", user.id, tier, limits)


@router.post("/webhook/revenuecat")
async def revenuecat_webhook(
    request: Request,
    authorization: str = Header(default=""),
) -> dict:
    """
    Receives RevenueCat server-to-server events and updates user tier.

    Configure in RevenueCat dashboard:
      URL: https://<your-backend>/api/v1/billing/webhook/revenuecat
      Authorization: <REVENUECAT_WEBHOOK_SECRET>
    """
    body = await request.body()

    if not _verify_signature(body, authorization):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    event = payload.get("event", {})
    event_type = event.get("type", "")
    app_user_id = event.get("app_user_id") or event.get("original_app_user_id")
    product_id = event.get("product_id", "")

    logger.info("RevenueCat webhook: type=%s app_user_id=%s product=%s", event_type, app_user_id, product_id)

    if not app_user_id:
        # Can't identify the user — acknowledge but ignore
        return {"received": True}

    # app_user_id is set to the Firebase UID in the mobile SDK ($RCAnonymousID is the fallback)
    user = await User.find_one(User.firebase_uid == app_user_id)
    if not user:
        logger.warning("RevenueCat webhook: no user found for app_user_id=%s", app_user_id)
        return {"received": True}

    # Store the RevenueCat ID for future reference
    if not user.revenuecat_id:
        await user.update(Set({User.revenuecat_id: app_user_id}))

    if event_type in ("INITIAL_PURCHASE", "RENEWAL", "PRODUCT_CHANGE", "UNCANCELLATION"):
        new_tier = _PRODUCT_TIER.get(product_id, "pro")
        await _set_tier(user, new_tier)

    elif event_type in ("CANCELLATION", "EXPIRATION", "BILLING_ISSUE"):
        await _set_tier(user, "free")

    return {"received": True}


@router.get("/me")
async def get_billing_status(current_user: User = Depends(get_current_user)) -> dict:
    """Return the current user's tier, limits, and usage — used by upgrade modal."""
    return {
        "tier": current_user.tier,
        "limits": {
            "max_sources": current_user.limits.max_sources,
            "max_slots": current_user.limits.max_slots,
            "max_image_inputs_per_month": current_user.limits.max_image_inputs_per_month,
        },
        "usage": {
            "sources_count": current_user.usage.sources_count,
            "slots_count": current_user.usage.slots_count,
            "image_inputs_this_month": current_user.usage.image_inputs_this_month,
        },
    }
