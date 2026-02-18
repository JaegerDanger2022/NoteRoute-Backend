import asyncio
import logging

import asyncpg
from pgvector.asyncpg import register_vector

from app.models.slot import KnowledgeSlot

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

_INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS slot_vectors (
    slot_id   TEXT        NOT NULL,
    user_id   TEXT        NOT NULL,
    slot_type TEXT        NOT NULL,  -- 'summary' or 'content'
    embedding vector(1536),
    PRIMARY KEY (slot_id, slot_type)
);

CREATE INDEX IF NOT EXISTS slot_vectors_user_idx
    ON slot_vectors (user_id);

CREATE INDEX IF NOT EXISTS slot_vectors_embedding_idx
    ON slot_vectors USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


async def init_pgvector(postgres_url: str) -> None:
    global _pool
    _pool = await asyncpg.create_pool(postgres_url, min_size=2, max_size=10)
    async with _pool.acquire() as conn:
        await register_vector(conn)
        await conn.execute(_INIT_SQL)
    logger.info("pgvector initialized")


async def close_pgvector() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("pgvector pool not initialized — call init_pgvector() at startup")
    return _pool


async def upsert_slot(
    slot: KnowledgeSlot,
    summary_vector: list[float],
    content_vector: list[float],
) -> None:
    slot_id = str(slot.id)
    user_id = str(slot.user_id)
    pool = _get_pool()
    async with pool.acquire() as conn:
        await register_vector(conn)
        await conn.executemany(
            """
            INSERT INTO slot_vectors (slot_id, user_id, slot_type, embedding)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (slot_id, slot_type)
            DO UPDATE SET embedding = EXCLUDED.embedding, user_id = EXCLUDED.user_id
            """,
            [
                (slot_id, user_id, "summary", summary_vector),
                (slot_id, user_id, "content", content_vector),
            ],
        )


async def search_slots(
    summary_query_vec: list[float],
    content_query_vec: list[float],
    user_id: str,
    top_k: int = 10,
) -> list[dict]:
    pool = _get_pool()
    async with pool.acquire() as conn:
        await register_vector(conn)

        summary_rows = await conn.fetch(
            """
            SELECT slot_id, 1 - (embedding <=> $1) AS score
            FROM slot_vectors
            WHERE user_id = $2 AND slot_type = 'summary'
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            summary_query_vec,
            user_id,
            top_k,
        )

        content_rows = await conn.fetch(
            """
            SELECT slot_id, 1 - (embedding <=> $1) AS score
            FROM slot_vectors
            WHERE user_id = $2 AND slot_type = 'content'
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            content_query_vec,
            user_id,
            top_k,
        )

    # Merge scores: max of summary and content score per slot
    scores: dict[str, float] = {}
    for row in summary_rows:
        scores[row["slot_id"]] = row["score"]
    for row in content_rows:
        sid = row["slot_id"]
        scores[sid] = max(scores.get(sid, 0.0), row["score"])

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"slot_id": sid, "score": score} for sid, score in ranked]


async def delete_slot(slot_id: str, user_id: str) -> None:
    pool = _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM slot_vectors WHERE slot_id = $1 AND user_id = $2",
            slot_id,
            user_id,
        )
