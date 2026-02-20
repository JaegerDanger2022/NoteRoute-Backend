import asyncio
import json
from dataclasses import dataclass

import boto3

from app.config import settings

_bedrock_client = None


@dataclass
class CustomBedrockCreds:
    """User-supplied AWS Bedrock credentials for BYOLLM."""
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str


def _get_client(custom: CustomBedrockCreds | None = None):
    global _bedrock_client
    if custom:
        # Always create a fresh client for custom creds — no caching
        return boto3.client(
            "bedrock-runtime",
            region_name=custom.aws_region,
            aws_access_key_id=custom.aws_access_key_id,
            aws_secret_access_key=custom.aws_secret_access_key,
        )
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    return _bedrock_client


def _invoke_embedding(text: str, custom: CustomBedrockCreds | None = None) -> list[float]:
    client = _get_client(custom)
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    response = client.invoke_model(
        modelId=settings.BEDROCK_EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


async def embed_text(text: str, custom: CustomBedrockCreds | None = None) -> list[float]:
    """Generate a 1024-dim embedding vector via AWS Bedrock Titan Text Embeddings V2."""
    return await asyncio.to_thread(_invoke_embedding, text, custom)
