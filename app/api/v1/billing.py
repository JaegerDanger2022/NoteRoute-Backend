import hashlib
import hmac
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

PAYSTACK_API = "https://api.paystack.co"

# Plan codes from Paystack dashboard (Products → Plans)
PAYSTACK_PLAN_CODES: dict[str, str] = {
    "monthly":    "PLN_auj4ymqec30fwtk",
    "quarterly":  "PLN_bt3tq1kcgw8uf2y",
    "biannually": "PLN_ya121jv48qvlwmt",
    "annually":   "PLN_nvc217h1xd7hrpv",
}

# Tier limit definitions
_TIER_LIMITS: dict[str, TierLimits] = {
    "free": TierLimits(max_sources=1, max_slots=20, max_image_inputs_per_month=3),
    "pro": TierLimits(max_sources=20, max_slots=500, max_image_inputs_per_month=500),
}


def _verify_paystack_signature(body: bytes, signature: str) -> bool:
    """Verify Paystack webhook HMAC-SHA512 signature."""
    if not settings.PAYSTACK_SECRET_KEY:
        logger.warning("PAYSTACK_SECRET_KEY not set — skipping signature verification")
        return True
    expected = hmac.new(
        settings.PAYSTACK_SECRET_KEY.encode(),
        body,
        hashlib.sha512,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def _set_tier(user: User, tier: str) -> None:
    limits = _TIER_LIMITS.get(tier, _TIER_LIMITS["free"])
    await user.update(Set({
        User.tier: tier,
        User.limits: limits,
    }))
    logger.info("Updated user %s to tier=%s", user.id, tier)


# ── Initialize transaction ────────────────────────────────────────────────────

class InitializeRequest(BaseModel):
    interval: str = "monthly"  # monthly | quarterly | biannually | annually


@router.post("/paystack/initialize")
async def initialize_transaction(
    body: InitializeRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Ask Paystack to create a subscription transaction for the current user.
    Returns { authorization_url, access_code, reference } — the frontend
    uses authorization_url to redirect or open the Paystack popup.
    """
    if not settings.PAYSTACK_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Paystack not configured")

    plan_code = PAYSTACK_PLAN_CODES.get(body.interval)
    if not plan_code:
        raise HTTPException(status_code=400, detail=f"Unknown interval: {body.interval}")

    # Amounts in GHS pesewas (integer, no decimals). Must match Paystack plan amounts exactly.
    amounts_pesewas = {
        "monthly":    13200,    # GHS 132.00
        "quarterly":  34650,    # GHS 346.50
        "biannually": 79118,    # GHS 791.18
        "annually":   118958,   # GHS 1,189.58
    }

    payload = {
        "email": current_user.email,
        "amount": amounts_pesewas[body.interval],
        "plan": plan_code,
        "metadata": {
            "firebase_uid": current_user.firebase_uid,
            "cancel_action": settings.WEB_APP_URL,
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{PAYSTACK_API}/transaction/initialize",
            json=payload,
            headers={"Authorization": f"Bearer {settings.PAYSTACK_SECRET_KEY}"},
            timeout=15,
        )

    if resp.status_code != 200:
        logger.error("Paystack initialize failed: status=%s body=%s", resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail=f"Paystack error: {resp.json().get('message', resp.text)}")

    data = resp.json().get("data", {})
    return {
        "authorization_url": data.get("authorization_url"),
        "access_code": data.get("access_code"),
        "reference": data.get("reference"),
    }


# ── Webhook ───────────────────────────────────────────────────────────────────

@router.post("/webhook/paystack")
async def paystack_webhook(
    request: Request,
    x_paystack_signature: str = Header(default=""),
) -> dict:
    """
    Receives Paystack server-to-server events and updates user tier.

    Configure in Paystack dashboard → Settings → API Keys & Webhooks:
      URL: https://<your-backend>/api/v1/billing/webhook/paystack
    """
    body = await request.body()

    if not _verify_paystack_signature(body, x_paystack_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    event = payload.get("event", "")
    data = payload.get("data", {})

    customer = data.get("customer", {})
    email = customer.get("email") or data.get("email", "")

    logger.info("Paystack webhook: event=%s email=%s", event, email)

    if not email:
        return {"received": True}

    user = await User.find_one(User.email == email)
    if not user:
        logger.warning("Paystack webhook: no user found for email=%s", email)
        return {"received": True}

    # Subscription activated or renewed
    if event in (
        "subscription.create",
        "charge.success",
        "invoice.payment_succeeded",
        "subscription.not_renew",  # un-cancel
    ):
        await _set_tier(user, "pro")

    # Subscription cancelled, payment failed, or disabled
    elif event in (
        "subscription.disable",
        "invoice.payment_failed",
        "subscription.expiry_date_update",
    ):
        await _set_tier(user, "free")

    return {"received": True}


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
