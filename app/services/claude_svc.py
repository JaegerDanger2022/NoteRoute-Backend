import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Literal, Optional

import boto3
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_NOVA_LITE_MODEL_ID = "amazon.nova-lite-v1:0"

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


@dataclass
class CustomLLMCreds:
    provider: Literal["openai", "anthropic"]
    api_key: str


# ---------------------------------------------------------------------------
# Provider-specific helpers (sync, called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _call_openai_sync(prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    """Call OpenAI chat completions API synchronously."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def _call_anthropic_sync(prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    """Call Anthropic messages API synchronously."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Core sync functions (routed by provider)
# ---------------------------------------------------------------------------

def _infer_slot_metadata_sync(
    slot_name: str,
    source_name: str,
    provider: str,
    custom: Optional[CustomLLMCreds] = None,
    use_nova: bool = False,
) -> dict:
    """
    Call an LLM to infer a description and tags for a slot based on its name.
    Routes to OpenAI / Anthropic direct if custom creds provided, otherwise Bedrock.
    Returns {"description": str, "tags": list[str]}.
    """
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

    if custom:
        if custom.provider == "openai":
            text = _call_openai_sync(prompt, custom.api_key, "gpt-4o-mini", 256)
        else:
            text = _call_anthropic_sync(prompt, custom.api_key, "claude-haiku-4-5-20251001", 256)
    else:
        client = _get_client()
        if use_nova:
            body = json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 256},
            })
            response = client.invoke_model(
                modelId=_NOVA_LITE_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            text = result["output"]["message"]["content"][0]["text"].strip()
        else:
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

    text = _strip_fences(text)
    parsed = json.loads(text)
    return {
        "description": str(parsed.get("description", "")).strip(),
        "tags": [str(t).lower().strip() for t in parsed.get("tags", []) if t][:5],
    }


def _summarize_slot_content_sync(
    slot_name: str,
    raw_content: str,
    custom: Optional[CustomLLMCreds] = None,
    use_nova: bool = False,
) -> str:
    """Summarize raw resource content into a concise indexing-ready passage."""
    truncated = raw_content[:100000]
    prompt = f"""You are preparing a knowledge index entry.

Slot name: "{slot_name}"
Resource content:
---
{truncated}
---

Write a comprehensive summary (500-700 words) of what this resource is about and what kind of notes or information typically belong here. Cover the topic, purpose, key themes, recurring subjects, notable structure, categories, and any specific terminology or concepts present. The richer this summary, the better the search experience for the user. Output plain text only, no markdown."""

    if custom:
        if custom.provider == "openai":
            return _call_openai_sync(prompt, custom.api_key, "gpt-4o-mini", 1024)
        else:
            return _call_anthropic_sync(prompt, custom.api_key, "claude-haiku-4-5-20251001", 1024)
    else:
        client = _get_client()
        if use_nova:
            body = json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 1024},
            })
            response = client.invoke_model(
                modelId=_NOVA_LITE_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            return result["output"]["message"]["content"][0]["text"].strip()
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(
            modelId=_HAIKU_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------

async def infer_slot_metadata(
    slot_name: str,
    source_name: str,
    provider: str,
    custom: Optional[CustomLLMCreds] = None,
) -> dict:
    """Async wrapper — returns {"description": str, "tags": list[str]}."""
    from app.models.global_config import GlobalConfig
    config = await GlobalConfig.get_config()
    return await asyncio.to_thread(
        _infer_slot_metadata_sync, slot_name, source_name, provider, custom, config.use_nova
    )


async def summarize_slot_content(
    slot_name: str,
    raw_content: str,
    custom: Optional[CustomLLMCreds] = None,
) -> str:
    """Async wrapper — returns a plain-text summary of resource content."""
    from app.models.global_config import GlobalConfig
    config = await GlobalConfig.get_config()
    return await asyncio.to_thread(
        _summarize_slot_content_sync, slot_name, raw_content, custom, config.use_nova
    )


_ROUTING_SUMMARY_MAX_CHARS = 1500


def _generate_routing_summary_sync(
    slot_name: str,
    description: str,
    content_sample: str,
    tags: list[str],
    provider: str,
    custom: Optional[CustomLLMCreds] = None,
    use_nova: bool = False,
) -> str:
    """Generate a routing-focused summary that captures purpose + named entities.

    Used at routing time so the LLM can make context-aware decisions rather than
    relying on semantic similarity alone. Capped at 1500 characters so 50 slots
    fit comfortably in a single LLM context window (~75 000 chars to spare).
    """
    context_parts = [f'Name: "{slot_name}"', f'Provider: {provider}']
    if tags:
        context_parts.append(f'Tags: {", ".join(tags)}')
    if description and description.strip() and description.strip() != slot_name.strip():
        context_parts.append(f'Description: {description}')
    if content_sample and content_sample.strip():
        context_parts.append(f'Content sample: {content_sample[:2000]}')
    context = "\n".join(context_parts)

    prompt = f"""You are building a routing index for a knowledge management system.

Slot context:
{context}

Write a concise routing summary for this slot. Cover:
1. What kind of notes or information belong here (purpose and scope)
2. Named entities: people, organisations, places, project names, or product names that are associated with this slot — extract from the name, description, and content sample
3. Topics, themes, or recurring subjects

Rules:
- Plain text only, no markdown or bullet points
- Maximum 1500 characters — be concise but information-dense
- If no named entities can be inferred, omit that section rather than guessing
- Do not repeat the slot name verbatim as the first sentence"""

    if custom:
        if custom.provider == "openai":
            text = _call_openai_sync(prompt, custom.api_key, "gpt-4o-mini", 400)
        else:
            text = _call_anthropic_sync(prompt, custom.api_key, "claude-haiku-4-5-20251001", 400)
    else:
        client = _get_client()
        if use_nova:
            body = json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 400},
            })
            response = client.invoke_model(
                modelId=_NOVA_LITE_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            text = result["output"]["message"]["content"][0]["text"].strip()
        else:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 400,
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

    return text[:_ROUTING_SUMMARY_MAX_CHARS]


async def generate_routing_summary(
    slot_name: str,
    description: str,
    content_sample: str,
    tags: list[str],
    provider: str,
    custom: Optional[CustomLLMCreds] = None,
) -> str:
    """Async wrapper — returns a routing summary capped at 1500 chars."""
    from app.models.global_config import GlobalConfig
    config = await GlobalConfig.get_config()
    return await asyncio.to_thread(
        _generate_routing_summary_sync,
        slot_name, description, content_sample, tags, provider, custom, config.use_nova,
    )
