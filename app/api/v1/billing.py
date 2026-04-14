import hashlib
import hmac
import json
import logging

import httpx
from beanie.operators import Set
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.models.user import TierLimits, User
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])

POLAR_API = "https://api.polar.sh"

# Polar product price IDs — create these in Polar dashboard under Products
# One price per billing interval. Replace with your actual IDs after setup.
POLAR_PRICE_IDS: dict[str, str] = {
    "monthly":    "08fa1ac6-5846-473d-99ad-351f8de9e66d",
    "quarterly":  "price_quarterly_placeholder",
    "biannually": "price_biannually_placeholder",
    "annually":   "price_annually_placeholder",
}

# Tier limit definitions
_TIER_LIMITS: dict[str, TierLimits] = {
    "free": TierLimits(max_sources=1, max_slots=20, max_image_inputs_per_month=3),
    "pro": TierLimits(max_sources=20, max_slots=500, max_image_inputs_per_month=500),
}


def _verify_polar_signature(body: bytes, signature: str) -> bool:
    """Verify Polar webhook HMAC-SHA256 signature.

    Polar sends the signature as: sha256=<hex>
    Docs: https://docs.polar.sh/integrate/webhooks/overview
    """
    if not settings.POLAR_WEBHOOK_SECRET:
        logger.warning("POLAR_WEBHOOK_SECRET not set — skipping signature verification")
        return True
    expected = "sha256=" + hmac.new(
        settings.POLAR_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def _set_tier(user: User, tier: str) -> None:
    limits = _TIER_LIMITS.get(tier, _TIER_LIMITS["free"])
    await user.update(Set({
        User.tier: tier,
        User.limits: limits,
    }))
    logger.info("Updated user %s to tier=%s", user.id, tier)


# ── Initialize checkout session ───────────────────────────────────────────────

class InitializeRequest(BaseModel):
    interval: str = "monthly"  # monthly | quarterly | biannually | annually


@router.post("/polar/initialize")
async def initialize_checkout(
    body: InitializeRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Create a Polar checkout session for the current user.
    Returns { checkout_url } — the frontend redirects the user there.

    Polar docs: https://docs.polar.sh/integrate/checkout/embed
    """
    if not settings.POLAR_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="Polar not configured")

    price_id = POLAR_PRICE_IDS.get(body.interval)
    if not price_id:
        raise HTTPException(status_code=400, detail=f"Unknown interval: {body.interval}")

    payload = {
        "products": [{"price_id": price_id}],
        "customer_email": current_user.email,
        "metadata": {
            "firebase_uid": current_user.firebase_uid,
        },
        "success_url": f"{settings.WEB_APP_URL}/billing/callback?checkout_id={{CHECKOUT_ID}}",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{POLAR_API}/v1/checkouts",
            json=payload,
            headers={
                "Authorization": f"Bearer {settings.POLAR_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )

    if resp.status_code not in (200, 201):
        logger.error("Polar checkout failed: status=%s body=%s", resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail=f"Polar error: {resp.json().get('detail', resp.text)}")

    data = resp.json()
    return {
        "checkout_url": data.get("url"),
        "checkout_id": data.get("id"),
    }


# ── Webhook ───────────────────────────────────────────────────────────────────

@router.post("/webhook/polar")
async def polar_webhook(
    request: Request,
    webhook_signature: str = Header(default="", alias="webhook-signature"),
) -> dict:
    """
    Receives Polar server-to-server events and updates user tier.

    Configure in Polar dashboard → Settings → Webhooks:
      URL: https://<your-backend>/api/v1/billing/webhook/polar

    Relevant events to subscribe to:
      - subscription.active
      - subscription.updated
      - subscription.canceled
      - subscription.revoked
    """
    body = await request.body()

    if not _verify_polar_signature(body, webhook_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    event_type = payload.get("type", "")
    data = payload.get("data", {})

    # Customer email lives at data.customer.email or data.customer_email
    customer = data.get("customer", {})
    email = customer.get("email") or data.get("customer_email", "")

    logger.info("Polar webhook: event=%s email=%s", event_type, email)

    if not email:
        return {"received": True}

    user = await User.find_one(User.email == email)
    if not user:
        logger.warning("Polar webhook: no user found for email=%s", email)
        return {"received": True}

    # Subscription activated or renewed
    if event_type in ("subscription.active", "subscription.updated"):
        subscription_id = data.get("id")
        status = data.get("status")
        if status == "active" and subscription_id:
            await user.update(Set({User.polar_subscription_id: subscription_id}))
            await _set_tier(user, "pro")
        elif status in ("canceled", "revoked", "past_due", "unpaid"):
            await user.update(Set({User.polar_subscription_id: None}))
            await _set_tier(user, "free")

    # Subscription cancelled or revoked
    elif event_type in ("subscription.canceled", "subscription.revoked"):
        await user.update(Set({User.polar_subscription_id: None}))
        await _set_tier(user, "free")

    return {"received": True}


# ── Cancel subscription ───────────────────────────────────────────────────────

@router.post("/polar/cancel")
async def cancel_subscription(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Cancel the user's active Polar subscription."""
    if current_user.tier != "pro":
        raise HTTPException(status_code=400, detail="No active subscription to cancel")

    if not current_user.polar_subscription_id:
        raise HTTPException(status_code=400, detail="Subscription ID not found — please contact support")

    if not settings.POLAR_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="Polar not configured")

    async with httpx.AsyncClient() as client:
        resp = await client.delete(
            f"{POLAR_API}/v1/subscriptions/{current_user.polar_subscription_id}",
            headers={"Authorization": f"Bearer {settings.POLAR_ACCESS_TOKEN}"},
            timeout=15,
        )

    if resp.status_code not in (200, 204):
        logger.error("Polar cancel failed: status=%s body=%s", resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail=f"Polar error: {resp.text}")

    # Downgrade immediately — webhook will also fire as confirmation
    await current_user.update(Set({User.polar_subscription_id: None}))
    await _set_tier(current_user, "free")

    logger.info("User %s cancelled Polar subscription", current_user.id)
    return {"cancelled": True}


# ── Billing status ────────────────────────────────────────────────────────────

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
