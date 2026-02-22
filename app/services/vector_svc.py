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

    existing = [i["name"] for i in pc.list_indexes()]
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
    existing = [i["name"] for i in pc.list_indexes()]
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
        existing = [i["name"] for i in pc.list_indexes()]
        return creds.index_name in existing
    except Exception:
        return False


def upsert_slot(
    slot: KnowledgeSlot,
    summary_vector: list[float],
    content_vector: list[float],
    custom: CustomIndexCreds | None = None,
) -> None:
    idx = _get_index(custom)
    slot_id = str(slot.id)
    metadata = {
        "user_id": str(slot.user_id),
        "source_id": str(slot.source_id),
    }
    idx.upsert(vectors=[{"id": slot_id, "values": summary_vector, "metadata": metadata}], namespace=_NS_SUMMARY)
    idx.upsert(vectors=[{"id": slot_id, "values": content_vector, "metadata": metadata}], namespace=_NS_CONTENT)


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


def delete_slot(slot_id: str, custom: CustomIndexCreds | None = None) -> None:
    idx = _get_index(custom)
    for ns in (_NS_SUMMARY, _NS_CONTENT):
        idx.delete(ids=[slot_id], namespace=ns)
        logger.info("Deleted vector %s from namespace %s", slot_id, ns)
