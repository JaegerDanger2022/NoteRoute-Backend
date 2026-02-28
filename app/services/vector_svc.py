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


def check_custom_index_exists(creds: CustomIndexCreds) -> bool:
    """Return True if the user's index still exists in Pinecone."""
    try:
        pc = Pinecone(api_key=creds.pinecone_api_key)
        existing = [i.name for i in pc.list_indexes()]
        return creds.index_name in existing
    except Exception:
        return False


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
    """
    idx = _get_index(custom)
    slot_id = str(slot.id)
    base_meta = {
        "user_id": str(slot.user_id),
        "source_id": str(slot.source_id),
    }
    idx.upsert(vectors=[{"id": slot_id, "values": summary_vector, "metadata": base_meta}], namespace=_NS_SUMMARY)
    chunk_upserts = [
        {"id": f"{slot_id}#{i}", "values": vec, "metadata": base_meta}
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
    current_custom: CustomIndexCreds | None,
) -> CustomIndexCreds | None:
    """Return the CustomIndexCreds needed to delete a vector originally stored in *index_name*.

    Logic:
    - If the slot's index_name matches the user's current custom index, use those creds.
    - Otherwise (shared index, empty, or a rotated/deleted custom index) use None (shared).
      For rotated/deleted custom indexes we can't recover the old key — log a warning so the
      operator knows a stale vector may remain.
    """
    if current_custom and index_name == current_custom.index_name:
        return current_custom  # slot is in the user's active custom index
    if index_name and index_name != settings.PINECONE_INDEX_NAME:
        # Recorded a custom index name but it doesn't match the current one — key is gone
        logger.warning(
            "Cannot delete from index '%s' — user's current custom index is '%s'. "
            "Stale vector may remain in the old index.",
            index_name,
            current_custom.index_name if current_custom else "none",
        )
    return None  # shared index (or fallback)


def delete_slot(slot_id: str, custom: CustomIndexCreds | None = None) -> None:
    """Delete the summary vector and all content chunks for a slot."""
    idx = _get_index(custom)
    # Summary namespace: always a single vector with the bare slot_id
    idx.delete(ids=[slot_id], namespace=_NS_SUMMARY)
    logger.info("Deleted vector %s from namespace %s", slot_id, _NS_SUMMARY)
    # Content namespace: may be a single vector (slot_id) or chunks (slot_id#0, slot_id#1, …).
    # Delete the bare id first, then sweep chunks by prefix via list_paginated.
    idx.delete(ids=[slot_id], namespace=_NS_CONTENT)
    try:
        chunk_ids: list[str] = []
        for page in idx.list_paginated(prefix=f"{slot_id}#", namespace=_NS_CONTENT):
            chunk_ids.extend(page.vectors if page.vectors else [])
        if chunk_ids:
            idx.delete(ids=chunk_ids, namespace=_NS_CONTENT)
            logger.info("Deleted %d content chunks for slot %s", len(chunk_ids), slot_id)
    except Exception:
        logger.warning("Chunk cleanup via list_paginated failed for slot %s — chunks may remain", slot_id)
