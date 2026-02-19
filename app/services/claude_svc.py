import asyncio
import json
import logging

import boto3

from app.config import settings

logger = logging.getLogger(__name__)

_HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

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


def _infer_slot_metadata_sync(slot_name: str, source_name: str, provider: str) -> dict:
    """
    Call Claude Haiku to infer a description and tags for a slot based on its name.
    Returns {"description": str, "tags": list[str]}.
    """
    client = _get_client()

    prompt = f"""You are helping categorize a note-taking destination slot.

Slot name: "{slot_name}"
Source: "{source_name}" ({provider})

Respond with a JSON object only — no explanation, no markdown:
{{
  "description": "<1-2 sentence description of what kind of notes belong here>",
  "tags": ["<tag1>", "<tag2>", ...]
}}

Rules:
- description: infer from the name what topic/purpose this destination serves
- tags: 2-5 lowercase single-word tags from this list where relevant: work, personal, meetings, ideas, journal, research, tasks, notes, finance, health, travel, learning, projects, clients
- Only include tags that genuinely fit — do not pad with unrelated tags
- If the name is ambiguous, lean toward general tags"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
    })

    response = client.invoke_model(
        modelId=_HAIKU_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    text = result["content"][0]["text"].strip()

    # Parse JSON from response
    parsed = json.loads(text)
    return {
        "description": str(parsed.get("description", "")).strip(),
        "tags": [str(t).lower().strip() for t in parsed.get("tags", []) if t][:5],
    }


async def infer_slot_metadata(slot_name: str, source_name: str, provider: str) -> dict:
    """Async wrapper — returns {"description": str, "tags": list[str]}."""
    return await asyncio.to_thread(
        _infer_slot_metadata_sync, slot_name, source_name, provider
    )
