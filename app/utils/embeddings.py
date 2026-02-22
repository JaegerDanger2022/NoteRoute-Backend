import asyncio
import json
from dataclasses import dataclass

import boto3

from app.config import settings

_bedrock_client = None

_NOVA_EMBED_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"


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


def _invoke_embedding(
    text: str,
    custom: CustomBedrockCreds | None = None,
    use_nova: bool = False,
) -> list[float]:
    client = _get_client(custom)
    if use_nova:
        # Nova 2 multimodal embeddings: structured schema, returns embeddings array
        body = json.dumps({
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 1024,
                "text": {"truncationMode": "END", "value": text},
            },
        })
        model_id = _NOVA_EMBED_MODEL_ID
    else:
        # Titan embed: explicit dimensions + normalize
        body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
        model_id = settings.BEDROCK_EMBED_MODEL_ID
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    if use_nova:
        return result["embeddings"][0]["embedding"]
    return result["embedding"]


async def embed_text(text: str, custom: CustomBedrockCreds | None = None) -> list[float]:
    """Generate a 1024-dim embedding vector via AWS Bedrock (Titan or Nova per GlobalConfig)."""
    from app.models.global_config import GlobalConfig
    config = await GlobalConfig.get_config()
    return await asyncio.to_thread(_invoke_embedding, text, custom, config.use_nova)
