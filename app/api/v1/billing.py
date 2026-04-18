import base64
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

POLAR_API = "https://sandbox-api.polar.sh"

# Polar product IDs — one product per billing interval.
# Find these in Polar dashboard → Products → click product → copy the ID from the URL.
POLAR_PRODUCT_IDS: dict[str, str] = {
    "monthly":    "44d1afb5-daa7-497c-aa79-73d01c38f8b5",
    "quarterly":  "b5cab4bc-3970-480e-b975-f8deee2b130c",
    "biannually": "2332d5da-0162-4c09-a556-572cdf812839",
    "annually":   "c09a378d-21cb-4344-8d01-130bfefdcda5",
}

# Tier limit definitions
_TIER_LIMITS: dict[str, TierLimits] = {
    "free": TierLimits(max_sources=1, max_slots=20, max_image_inputs_per_month=3),
    "pro": TierLimits(max_sources=20, max_slots=500, max_image_inputs_per_month=500),
}


def _verify_polar_signature(body: bytes, msg_id: str, timestamp: str, signature: str) -> bool:
    """Verify Polar webhook signature per Standard Webhooks spec.

    Signed string: "{msg_id}.{timestamp}.{body}"
    Secret is base64-encoded. Header format: "v1,<base64-signature>"
    """
    if not settings.POLAR_WEBHOOK_SECRET:
        logger.warning("POLAR_WEBHOOK_SECRET not set — skipping signature verification")
        return True
    try:
        secret_bytes = base64.b64decode(settings.POLAR_WEBHOOK_SECRET)
    except Exception:
        secret_bytes = settings.POLAR_WEBHOOK_SECRET.encode()

    signed = f"{msg_id}.{timestamp}.".encode() + body
    expected_bytes = hmac.new(secret_bytes, signed, hashlib.sha256).digest()
    expected_b64 = base64.b64encode(expected_bytes).decode()

    for part in signature.split(" "):
        if part.startswith("v1,") and hmac.compare_digest(part[3:], expected_b64):
            return True
    return False


async def _set_tier(user: User, tier: str) -> None:
    limits = _TIER_LIMITS.get(tier, _TIER_LIMITS["free"])
    await user.update(Set({
        User.tier: tier,
        User.limits: limits,
    }))
    logger.info("Updated user %s to tier=%s", user.id, tier)


# ── Plans (live prices from Polar) ───────────────────────────────────────────

@router.get("/plans")
async def get_plans() -> dict:
    """Fetch live prices for each interval from Polar and return formatted amounts."""
    if not settings.POLAR_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="Polar not configured")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        results: dict[str, str] = {}
        for interval, product_id in POLAR_PRODUCT_IDS.items():
            resp = await client.get(
                f"{POLAR_API}/v1/products/{product_id}",
                headers={"Authorization": f"Bearer {settings.POLAR_ACCESS_TOKEN}"},
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("Failed to fetch product %s: %s", product_id, resp.status_code)
                continue
            data = resp.json()
            for price in data.get("prices", []):
                if price.get("amount_type") == "fixed" and not price.get("is_archived"):
                    cents = price.get("price_amount", 0)
                    currency = price.get("price_currency", "usd").upper()
                    symbol = "$" if currency == "USD" else currency + " "
                    results[interval] = f"{symbol}{cents / 100:.2f}"
                    break

    return results


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

    product_id = POLAR_PRODUCT_IDS.get(body.interval)
    if not product_id:
        raise HTTPException(status_code=400, detail=f"Unknown interval: {body.interval}")

    payload = {
        "products": [product_id],
        "customer_email": current_user.email,
        "metadata": {
            "firebase_uid": current_user.firebase_uid,
        },
        "success_url": f"{settings.WEB_APP_URL}/billing/callback?checkout_id={{CHECKOUT_ID}}",
    }

    async with httpx.AsyncClient(follow_redirects=True) as client:
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
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=502, detail=f"Polar error: {detail}")

    data = resp.json()
    return {
        "checkout_url": data.get("url"),
        "checkout_id": data.get("id"),
    }


# ── Webhook ───────────────────────────────────────────────────────────────────

@router.post("/webhook/polar")
async def polar_webhook(
    request: Request,
    webhook_id: str = Header(default="", alias="webhook-id"),
    webhook_timestamp: str = Header(default="", alias="webhook-timestamp"),
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

    if not _verify_polar_signature(body, webhook_id, webhook_timestamp, webhook_signature):
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
