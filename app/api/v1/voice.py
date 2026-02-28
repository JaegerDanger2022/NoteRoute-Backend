import uuid
from datetime import datetime, timezone

import boto3
from beanie.operators import Set
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.deps import get_current_user, is_admin
from app.config import settings
from app.models.user import User

router = APIRouter(prefix="/voice", tags=["voice"])

_s3_client = None

_IMAGE_CONTENT_TYPES = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/heic": "heic",
}


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
    return _s3_client


class PresignResponse(BaseModel):
    upload_url: str
    s3_key: str
    expires_in: int


@router.post("/presign", response_model=PresignResponse)
async def get_presigned_upload_url(
    current_user: User = Depends(get_current_user),
) -> PresignResponse:
    """Return a pre-signed S3 URL for direct audio upload from the client."""
    s3_key = f"uploads/{current_user.id}/{uuid.uuid4()}.webm"
    upload_url = _get_s3().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.AWS_TRANSCRIBE_BUCKET,
            "Key": s3_key,
            "ContentType": "audio/webm",
        },
        ExpiresIn=300,  # 5 minutes
    )
    return PresignResponse(upload_url=upload_url, s3_key=s3_key, expires_in=300)


@router.post("/image-presign", response_model=PresignResponse)
async def get_image_presigned_upload_url(
    content_type: str = Query(default="image/jpeg"),
    current_user: User = Depends(get_current_user),
) -> PresignResponse:
    """Return a pre-signed S3 URL for direct image upload from the client."""
    # Enforce monthly image input limit — reset counter when the month changes
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    usage = current_user.usage
    if usage.image_inputs_month != current_month:
        usage.image_inputs_this_month = 0
        usage.image_inputs_month = current_month
        await current_user.update(Set({
            User.usage: usage,
        }))

    if not is_admin(current_user) and usage.image_inputs_this_month >= current_user.limits.max_image_inputs_per_month:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly image input limit reached ({current_user.limits.max_image_inputs_per_month}). Upgrade to Pro for more.",
        )

    # Increment before issuing the presign URL so the slot is reserved
    usage.image_inputs_this_month += 1
    await current_user.update(Set({User.usage: usage}))

    ext = _IMAGE_CONTENT_TYPES.get(content_type, "jpg")
    s3_key = f"images/{current_user.id}/{uuid.uuid4()}.{ext}"
    upload_url = _get_s3().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.AWS_TRANSCRIBE_BUCKET,
            "Key": s3_key,
            "ContentType": content_type,
        },
        ExpiresIn=300,  # 5 minutes
    )
    return PresignResponse(upload_url=upload_url, s3_key=s3_key, expires_in=300)
