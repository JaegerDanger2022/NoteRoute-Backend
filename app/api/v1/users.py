import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.deps import get_current_user, is_admin
from app.config import settings
from app.core.exceptions import NotFoundError, TierLimitError
from app.core.security import decrypt_token, encrypt_token
from app.models.source import Source
from app.models.user import CustomIndexConfig, CustomLLMConfig, User
from app.services import vector_svc
from app.services.vector_svc import CustomIndexCreds
from beanie import PydanticObjectId
from beanie.operators import Set

router = APIRouter(prefix="/users", tags=["users"])


class UserUpdateRequest(BaseModel):
    display_name: str | None = None


class ActiveSourceRequest(BaseModel):
    source_id: str | None = None


@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)) -> dict:
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "display_name": current_user.display_name,
        "tier": current_user.tier,
        "active_source_id": str(current_user.active_source_id) if current_user.active_source_id else None,
        "is_admin": is_admin(current_user),
        "limits": {
            "max_sources": current_user.limits.max_sources,
            "max_slots": current_user.limits.max_slots,
        },
        "usage": {
            "slots_count": current_user.usage.slots_count,
            "sources_count": current_user.usage.sources_count,
        },
    }


@router.patch("/me")
async def update_me(
    body: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    if body.display_name is not None:
        await current_user.update(Set({User.display_name: body.display_name}))
    return {"id": str(current_user.id), "display_name": body.display_name or current_user.display_name}


@router.delete("/me", status_code=204)
async def delete_me(
    current_user: User = Depends(get_current_user),
    delete_custom_vectors: bool = Query(default=False, description="Also delete vectors from the user's private Pinecone index"),
) -> None:
    """Permanently delete the current user's account and all associated data."""
    from app.models.integration import Integration
    from app.models.slot import KnowledgeSlot
    from app.models.route import Route
    from app.models.source import Source
    from app.services import vector_svc as _vector_svc

    user_id = current_user.id

    # Delete Pinecone vectors — shared index always, custom index only if opted in
    slots = await KnowledgeSlot.find(KnowledgeSlot.user_id == user_id).to_list()
    for slot in slots:
        try:
            is_custom = slot.index_name and slot.index_name != settings.PINECONE_INDEX_NAME
            if is_custom and not delete_custom_vectors:
                continue  # User chose to keep their private index data
            delete_creds = _vector_svc.resolve_delete_creds(slot.index_name, slot.index_api_key)
            await asyncio.to_thread(_vector_svc.delete_slot, str(slot.id), delete_creds, slot.chunk_count)
        except Exception:
            pass  # Best-effort — don't block account deletion

    # Delete all MongoDB documents
    await KnowledgeSlot.find(KnowledgeSlot.user_id == user_id).delete()
    await Source.find(Source.user_id == user_id).delete()
    await Integration.find(Integration.user_id == user_id).delete()
    await Route.find(Route.user_id == user_id).delete()
    await current_user.delete()

    # Delete the Firebase Auth user record so the email can be re-used
    # and the account is fully gone (not just signed out)
    try:
        from firebase_admin import auth as firebase_auth
        await asyncio.to_thread(firebase_auth.delete_user, current_user.firebase_uid)
    except Exception:
        pass  # Best-effort — MongoDB data is already gone


class CustomIndexRequest(BaseModel):
    pinecone_api_key: str
    bedrock_aws_access_key_id: str | None = None
    bedrock_aws_secret_access_key: str | None = None
    bedrock_aws_region: str | None = None


def _custom_index_response(cfg: CustomIndexConfig) -> dict:
    return {
        "index_name": cfg.index_name,
        "index_status": cfg.index_status,
        "provisioned_at": cfg.provisioned_at.isoformat() if cfg.provisioned_at else None,
        "has_bedrock_creds": bool(cfg.bedrock_aws_access_key_id),
        "bedrock_aws_region": cfg.bedrock_aws_region,
    }



@router.post("/me/custom-index", status_code=201)
async def connect_custom_index(
    body: CustomIndexRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    # TODO: enforce tier guard (pro/team only) once billing is live

    # Validate the API key before saving anything — fail fast with 400.
    user_id_short = str(current_user.id)[-8:]
    index_name = f"noteroute-{user_id_short}"
    try:
        await asyncio.to_thread(
            vector_svc.provision_custom_index,
            CustomIndexCreds(pinecone_api_key=body.pinecone_api_key, index_name=index_name),
        )
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "Unauthorized" in msg or "Invalid API Key" in msg or "unauthorized" in msg.lower():
            raise HTTPException(status_code=400, detail="Invalid Pinecone API key.")
        raise HTTPException(status_code=400, detail=f"Could not connect Pinecone index: {msg[:200]}")

    encrypted_key = encrypt_token(body.pinecone_api_key)
    cfg = CustomIndexConfig(
        pinecone_api_key=encrypted_key,
        index_name=index_name,
        index_status="ready",
        provisioned_at=datetime.now(timezone.utc),
        bedrock_aws_access_key_id=encrypt_token(body.bedrock_aws_access_key_id) if body.bedrock_aws_access_key_id else None,
        bedrock_aws_secret_access_key=encrypt_token(body.bedrock_aws_secret_access_key) if body.bedrock_aws_secret_access_key else None,
        bedrock_aws_region=body.bedrock_aws_region,
    )
    await current_user.update(Set({User.custom_index: cfg}))
    return _custom_index_response(cfg)


@router.get("/me/custom-index")
async def get_custom_index(current_user: User = Depends(get_current_user)) -> dict:
    if not current_user.custom_index:
        return {"index_status": "none"}

    cfg = current_user.custom_index
    # Check if user deleted the index or revoked the API key from Pinecone dashboard
    if cfg.index_status == "ready":
        creds = CustomIndexCreds(
            pinecone_api_key=decrypt_token(cfg.pinecone_api_key),
            index_name=cfg.index_name,
        )
        check = await asyncio.to_thread(vector_svc.check_custom_index_exists, creds)
        if check == "not_found":
            cfg.index_status = "deleted"
            await current_user.update(Set({User.custom_index: cfg}))
        elif check == "key_invalid":
            cfg.index_status = "key_invalid"
            await current_user.update(Set({User.custom_index: cfg}))

    return _custom_index_response(cfg)


@router.delete("/me/custom-index", status_code=204)
async def disconnect_custom_index(current_user: User = Depends(get_current_user)) -> None:
    await current_user.update(Set({User.custom_index: None}))


@router.get("/internal/{user_id}/custom-index-creds")
async def get_custom_index_creds_internal(user_id: str) -> dict:
    """Internal endpoint for LangGraph service — returns decrypted custom creds if available."""
    try:
        user = await User.get(PydanticObjectId(user_id))
    except Exception:
        return {"has_custom": False}
    if not user:
        return {"has_custom": False}

    result: dict = {"has_custom": False}

    # Custom Pinecone index creds
    cfg = user.custom_index
    if cfg and cfg.index_status == "ready":
        result["has_custom"] = True
        result["pinecone_api_key"] = decrypt_token(cfg.pinecone_api_key)
        result["index_name"] = cfg.index_name
        result["has_bedrock"] = bool(cfg.bedrock_aws_access_key_id)
        if cfg.bedrock_aws_access_key_id and cfg.bedrock_aws_secret_access_key:
            result["bedrock_aws_access_key_id"] = decrypt_token(cfg.bedrock_aws_access_key_id)
            result["bedrock_aws_secret_access_key"] = decrypt_token(cfg.bedrock_aws_secret_access_key)
            result["bedrock_aws_region"] = cfg.bedrock_aws_region or "us-east-1"

    # Custom LLM creds (BYOLLM)
    llm = user.custom_llm
    if llm:
        result["llm_provider"] = llm.provider
        result["llm_api_key"] = decrypt_token(llm.api_key)
    else:
        result["llm_provider"] = None
        result["llm_api_key"] = None

    return result


class CustomLLMRequest(BaseModel):
    provider: str  # "openai" or "anthropic"
    api_key: str


@router.post("/me/custom-llm", status_code=201)
async def connect_custom_llm(
    body: CustomLLMRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    if body.provider not in ("openai", "anthropic"):
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="provider must be 'openai' or 'anthropic'")
    await current_user.update(Set({User.custom_llm: CustomLLMConfig(
        provider=body.provider,  # type: ignore[arg-type]
        api_key=encrypt_token(body.api_key),
    )}))
    return {"provider": body.provider}


@router.get("/me/custom-llm")
async def get_custom_llm(current_user: User = Depends(get_current_user)) -> dict:
    if not current_user.custom_llm:
        return {"provider": None}
    return {"provider": current_user.custom_llm.provider}


@router.delete("/me/custom-llm", status_code=204)
async def disconnect_custom_llm(current_user: User = Depends(get_current_user)) -> None:
    await current_user.update(Set({User.custom_llm: None}))


@router.patch("/me/active-source")
async def set_active_source(
    body: ActiveSourceRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    if body.source_id is None:
        await current_user.update(Set({User.active_source_id: None}))
        return {"active_source_id": None}

    source_id = PydanticObjectId(body.source_id)
    source = await Source.find_one(
        Source.id == source_id,
        Source.user_id == current_user.id,
        Source.is_active == True,
    )
    if not source:
        raise NotFoundError("Source not found")

    await current_user.update(Set({User.active_source_id: source_id}))
    return {"active_source_id": str(source_id)}
