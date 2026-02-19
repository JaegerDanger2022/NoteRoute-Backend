import asyncio
import json

import boto3

from app.config import settings

_bedrock_client = None


def _get_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    return _bedrock_client


def _invoke_embedding(text: str) -> list[float]:
    client = _get_client()
    body = json.dumps({"inputText": text, "dimensions": 1536, "normalize": True})
    response = client.invoke_model(
        modelId=settings.BEDROCK_EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


async def embed_text(text: str) -> list[float]:
    """Generate a 1536-dim embedding vector via AWS Bedrock Titan Text Embeddings V2."""
    return await asyncio.to_thread(_invoke_embedding, text)
