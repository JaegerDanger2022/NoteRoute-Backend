import logging
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.models.slot import KnowledgeSlot

logger = logging.getLogger(__name__)

_NS_SUMMARY = "slot-summary"
_NS_CONTENT = "slot-content"
_DIMENSION = 1024  # Titan Text Embeddings V2

_shared_index = None


@dataclass
class CustomIndexCreds:
    """Decrypted credentials for a user's own Pinecone index."""
    pinecone_api_key: str
    index_name: str


def init_vector_store() -> None:
    global _shared_index
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    name = settings.PINECONE_INDEX_NAME

    existing = [i.name for i in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Created Pinecone index: %s", name)
    else:
        logger.info("Pinecone index already exists: %s", name)

    _shared_index = pc.Index(name)
    logger.info("Pinecone index ready: %s", name)


def _get_shared_index():
    if _shared_index is None:
        raise RuntimeError("vector store not initialized — call init_vector_store() at startup")
    return _shared_index


def _get_index(custom: CustomIndexCreds | None = None):
    """Return the appropriate Pinecone index — user's own or shared."""
    if custom:
        pc = Pinecone(api_key=custom.pinecone_api_key)
        return pc.Index(custom.index_name)
    return _get_shared_index()


def provision_custom_index(creds: CustomIndexCreds) -> None:
    """Create the user's own Pinecone index if it doesn't exist yet."""
    pc = Pinecone(api_key=creds.pinecone_api_key)
    existing_indexes = pc.list_indexes()
    logger.info("Pinecone list_indexes response type=%s value=%s", type(existing_indexes), existing_indexes)
    existing = [i.name for i in existing_indexes]
    logger.info("Existing index names: %s", existing)
    if creds.index_name not in existing:
        pc.create_index(
            name=creds.index_name,
            dimension=_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Provisioned custom Pinecone index: %s", creds.index_name)
    else:
        logger.info("Custom Pinecone index already exists: %s", creds.index_name)


def check_custom_index_exists(creds: CustomIndexCreds) -> str:
    """Check whether the user's custom index is reachable.

    Returns:
        "exists"      — index found and API key valid
        "not_found"   — API key valid but index no longer exists
        "key_invalid" — API key rejected (401/403)
    """
    try:
        pc = Pinecone(api_key=creds.pinecone_api_key)
        existing = [i.name for i in pc.list_indexes()]
        return "exists" if creds.index_name in existing else "not_found"
    except Exception as exc:
        msg = str(exc).lower()
        if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg or "invalid api key" in msg:
            return "key_invalid"
        # Network errors etc. — treat conservatively as key problem (don't mark index deleted)
        return "key_invalid"


_CHUNK_CHARS = 6000   # ~1500 tokens — wide enough for ~25 cards, narrow enough to preserve minority topics
_CHUNK_OVERLAP = 400  # ~100 tokens of overlap so card at a boundary appears in both chunks


def _split_chunks(text: str, size: int = _CHUNK_CHARS, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks. Returns at least one chunk."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def upsert_slot(
    slot: KnowledgeSlot,
    summary_vector: list[float],
    content_vector: list[float],
    custom: CustomIndexCreds | None = None,
) -> None:
    """Upsert summary vector (single) and content vector (single, no raw content)."""
    idx = _get_index(custom)
    slot_id = str(slot.id)
    metadata = {
        "user_id": str(slot.user_id),
        "source_id": str(slot.source_id),
    }
    idx.upsert(vectors=[{"id": slot_id, "values": summary_vector, "metadata": metadata}], namespace=_NS_SUMMARY)
    idx.upsert(vectors=[{"id": slot_id, "values": content_vector, "metadata": metadata}], namespace=_NS_CONTENT)


def upsert_slot_content_chunks(
    slot: KnowledgeSlot,
    summary_vector: list[float],
    chunk_vectors: list[list[float]],
    custom: CustomIndexCreds | None = None,
) -> None:
    """Upsert summary vector (single) + one content vector per chunk.

    Chunk IDs are stored as '{slot_id}#0', '{slot_id}#1', … so the search node
    can strip the suffix to recover the parent slot_id.
    Content chunks carry has_content=True so the search node can weight them higher.
    """
    idx = _get_index(custom)
    slot_id = str(slot.id)
    base_meta = {
        "user_id": str(slot.user_id),
        "source_id": str(slot.source_id),
    }
    content_meta = {**base_meta, "has_content": True}
    idx.upsert(vectors=[{"id": slot_id, "values": summary_vector, "metadata": base_meta}], namespace=_NS_SUMMARY)
    chunk_upserts = [
        {"id": f"{slot_id}#{i}", "values": vec, "metadata": content_meta}
        for i, vec in enumerate(chunk_vectors)
    ]
    idx.upsert(vectors=chunk_upserts, namespace=_NS_CONTENT)


def search_slots(
    summary_query_vec: list[float],
    content_query_vec: list[float],
    user_id: str,
    source_id: str,
    top_k: int = 10,
    custom: CustomIndexCreds | None = None,
) -> list[dict]:
    idx = _get_index(custom)
    metadata_filter = {"user_id": {"$eq": user_id}, "source_id": {"$eq": source_id}}

    summary_resp = idx.query(
        vector=summary_query_vec,
        top_k=top_k,
        filter=metadata_filter,
        namespace=_NS_SUMMARY,
        include_values=False,
    )
    content_resp = idx.query(
        vector=content_query_vec,
        top_k=top_k,
        filter=metadata_filter,
        namespace=_NS_CONTENT,
        include_values=False,
    )

    scores: dict[str, float] = {}
    for match in summary_resp.get("matches", []):
        scores[match["id"]] = match["score"]
    for match in content_resp.get("matches", []):
        sid = match["id"]
        scores[sid] = max(scores.get(sid, 0.0), match["score"])

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"slot_id": sid, "score": score} for sid, score in ranked]


def resolve_delete_creds(
    index_name: str | None,
    encrypted_api_key: str | None,
) -> CustomIndexCreds | None:
    """Return the CustomIndexCreds needed to delete a vector originally stored in *index_name*.

    Uses the encrypted API key stored on the slot at index time — so deletion works
    regardless of the user's current custom index configuration.

    Returns None when the slot lives on the shared index (no key needed).
    """
    if not index_name or index_name == settings.PINECONE_INDEX_NAME:
        return None  # shared index — use default
    if encrypted_api_key:
        from app.core.security import decrypt_token
        try:
            return CustomIndexCreds(
                pinecone_api_key=decrypt_token(encrypted_api_key),
                index_name=index_name,
            )
        except Exception:
            logger.warning(
                "Could not decrypt stored API key for index '%s' — stale vector may remain.",
                index_name,
            )
    else:
        logger.warning(
            "No stored API key for index '%s' — stale vector may remain.",
            index_name,
        )
    return None


def delete_slot(slot_id: str, custom: CustomIndexCreds | None = None, chunk_count: int = 0) -> None:
    """Delete the summary vector and all content chunks for a slot.

    chunk_count should be the value stored on KnowledgeSlot.chunk_count (set at index time).
    When > 0 we reconstruct the chunk IDs deterministically (slot_id#0 … slot_id#N-1),
    avoiding list_paginated which only works on serverless indexes.
    For legacy slots where chunk_count == 0 we fall back to list_paginated with a
    broad except so a failure there never blocks the rest of the deletion.
    """
    idx = _get_index(custom)
    # Summary namespace: always a single vector with the bare slot_id
    idx.delete(ids=[slot_id], namespace=_NS_SUMMARY)
    logger.info("Deleted vector %s from namespace %s", slot_id, _NS_SUMMARY)
    # Content namespace: delete the single-vector form first (covers non-chunked slots)
    idx.delete(ids=[slot_id], namespace=_NS_CONTENT)

    if chunk_count > 0:
        # Deterministic: reconstruct exactly the IDs that were stored
        chunk_ids = [f"{slot_id}#{i}" for i in range(chunk_count)]
        idx.delete(ids=chunk_ids, namespace=_NS_CONTENT)
        logger.info("Deleted %d content chunks for slot %s", chunk_count, slot_id)
    else:
        # Legacy fallback: list_paginated only works on serverless indexes
        try:
            chunk_ids = []
            for page in idx.list_paginated(prefix=f"{slot_id}#", namespace=_NS_CONTENT):
                for v in (page.vectors or []):
                    chunk_ids.append(v.id if hasattr(v, "id") else v)
            if chunk_ids:
                idx.delete(ids=chunk_ids, namespace=_NS_CONTENT)
                logger.info("Deleted %d legacy content chunks for slot %s", len(chunk_ids), slot_id)
        except Exception:
            logger.warning("list_paginated unsupported or failed for slot %s — chunks may remain", slot_id)
