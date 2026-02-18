# Phase 2: Pinecone vector operations
# Implemented when embedding service (utils/embeddings.py) is ready.

from app.models.slot import KnowledgeSlot


async def upsert_slot(slot: KnowledgeSlot, summary_vector: list[float], content_vector: list[float]) -> None:
    """Upsert summary and content vectors for a slot into Pinecone."""
    raise NotImplementedError("Pinecone service not yet implemented (Phase 2)")


async def search_slots(
    summary_query_vec: list[float],
    content_query_vec: list[float],
    user_id: str,
    top_k: int = 10,
) -> list[dict]:
    """Two-layer vector search. Filters by user_id metadata. Returns merged + ranked candidates."""
    raise NotImplementedError("Pinecone service not yet implemented (Phase 2)")


async def delete_slot(slot_id: str, user_id: str) -> None:
    """Delete both summary and content vectors for a slot."""
    raise NotImplementedError("Pinecone service not yet implemented (Phase 2)")
