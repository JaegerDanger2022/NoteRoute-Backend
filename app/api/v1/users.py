import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.exceptions import NotFoundError, TierLimitError
from app.core.security import decrypt_token, encrypt_token
from app.models.source import Source
from app.models.user import CustomIndexConfig, User
from app.services import vector_svc
from app.services.vector_svc import CustomIndexCreds
from beanie import PydanticObjectId

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
        "limits": {
            "max_sources": current_user.limits.max_sources,
            "max_slots": current_user.limits.max_slots,
            "max_routes_per_month": current_user.limits.max_routes_per_month,
        },
        "usage": {
            "routes_this_month": current_user.usage.routes_this_month,
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
        current_user.display_name = body.display_name
        await current_user.save()
    return {"id": str(current_user.id), "display_name": current_user.display_name}


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


async def _provision_index_background(user_id: str, index_name: str, encrypted_key: str) -> None:
    """Background: create Pinecone index, then update index_status to ready."""
    from app.models.user import User as UserModel
    from beanie import PydanticObjectId as OId
    user = await UserModel.get(OId(user_id))
    if not user or not user.custom_index:
        return
    try:
        creds = CustomIndexCreds(
            pinecone_api_key=decrypt_token(encrypted_key),
            index_name=index_name,
        )
        await asyncio.to_thread(vector_svc.provision_custom_index, creds)
        user.custom_index.index_status = "ready"
        user.custom_index.provisioned_at = datetime.now(timezone.utc)
    except Exception:
        user.custom_index.index_status = "error"
    await user.save()


@router.post("/me/custom-index", status_code=201)
async def connect_custom_index(
    body: CustomIndexRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
) -> dict:
    # TODO: enforce tier guard (pro/team only) once billing is live
    user_id_short = str(current_user.id)[-8:]
    index_name = f"noteroute-{user_id_short}"

    encrypted_key = encrypt_token(body.pinecone_api_key)
    cfg = CustomIndexConfig(
        pinecone_api_key=encrypted_key,
        index_name=index_name,
        index_status="provisioning",
        bedrock_aws_access_key_id=encrypt_token(body.bedrock_aws_access_key_id) if body.bedrock_aws_access_key_id else None,
        bedrock_aws_secret_access_key=encrypt_token(body.bedrock_aws_secret_access_key) if body.bedrock_aws_secret_access_key else None,
        bedrock_aws_region=body.bedrock_aws_region,
    )
    current_user.custom_index = cfg
    await current_user.save()

    background_tasks.add_task(
        _provision_index_background,
        str(current_user.id),
        index_name,
        encrypted_key,
    )
    return _custom_index_response(cfg)


@router.get("/me/custom-index")
async def get_custom_index(current_user: User = Depends(get_current_user)) -> dict:
    if not current_user.custom_index:
        return {"index_status": "none"}

    cfg = current_user.custom_index
    # Check if user deleted it from Pinecone dashboard
    if cfg.index_status == "ready":
        creds = CustomIndexCreds(
            pinecone_api_key=decrypt_token(cfg.pinecone_api_key),
            index_name=cfg.index_name,
        )
        exists = await asyncio.to_thread(vector_svc.check_custom_index_exists, creds)
        if not exists:
            cfg.index_status = "deleted"
            await current_user.save()

    return _custom_index_response(cfg)


@router.delete("/me/custom-index", status_code=204)
async def disconnect_custom_index(current_user: User = Depends(get_current_user)) -> None:
    current_user.custom_index = None
    await current_user.save()


@router.get("/internal/{user_id}/custom-index-creds")
async def get_custom_index_creds_internal(user_id: str) -> dict:
    """Internal endpoint for LangGraph service — returns decrypted custom creds if available."""
    try:
        user = await User.get(PydanticObjectId(user_id))
    except Exception:
        return {"has_custom": False}
    if not user or not user.custom_index or user.custom_index.index_status != "ready":
        return {"has_custom": False}
    cfg = user.custom_index
    result: dict = {
        "has_custom": True,
        "pinecone_api_key": decrypt_token(cfg.pinecone_api_key),
        "index_name": cfg.index_name,
        "has_bedrock": bool(cfg.bedrock_aws_access_key_id),
    }
    if cfg.bedrock_aws_access_key_id and cfg.bedrock_aws_secret_access_key:
        result["bedrock_aws_access_key_id"] = decrypt_token(cfg.bedrock_aws_access_key_id)
        result["bedrock_aws_secret_access_key"] = decrypt_token(cfg.bedrock_aws_secret_access_key)
        result["bedrock_aws_region"] = cfg.bedrock_aws_region or "us-east-1"
    return result


@router.patch("/me/active-source")
async def set_active_source(
    body: ActiveSourceRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    if body.source_id is None:
        current_user.active_source_id = None
        await current_user.save()
        return {"active_source_id": None}

    source_id = PydanticObjectId(body.source_id)
    source = await Source.find_one(
        Source.id == source_id,
        Source.user_id == current_user.id,
        Source.is_active == True,
    )
    if not source:
        raise NotFoundError("Source not found")

    current_user.active_source_id = source_id
    await current_user.save()
    return {"active_source_id": str(source_id)}
