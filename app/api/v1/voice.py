import uuid

import boto3
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.config import settings
from app.models.user import User

router = APIRouter(prefix="/voice", tags=["voice"])

_s3_client = None


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
