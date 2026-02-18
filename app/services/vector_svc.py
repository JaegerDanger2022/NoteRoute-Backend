import logging

import boto3
from botocore.exceptions import ClientError

from app.config import settings
from app.models.slot import KnowledgeSlot

logger = logging.getLogger(__name__)

_client = None

_INDEX_SUMMARY = "slot-summary"
_INDEX_CONTENT = "slot-content"
_DIMENSION = 1536  # amazon.titan-embed-text-v2:0
_INDEXES = [_INDEX_SUMMARY, _INDEX_CONTENT]


def init_vector_store() -> None:
    global _client
    _client = boto3.client(
        "s3vectors",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )
    bucket = settings.AWS_VECTOR_BUCKET_NAME
    for index_name in _INDEXES:
        try:
            _client.get_index(vectorBucketName=bucket, indexName=index_name)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NotFoundException", "ResourceNotFoundException", "NoSuchKey"):
                try:
                    _client.create_index(
                        vectorBucketName=bucket,
                        indexName=index_name,
                        dataType="float32",
                        dimension=_DIMENSION,
                        distanceMetric="cosine",
                    )
                    logger.info("Created S3 Vectors index: %s", index_name)
                except ClientError as ce:
                    if ce.response["Error"]["Code"] != "ConflictException":
                        raise
                    logger.info("S3 Vectors index already exists (race): %s", index_name)
            else:
                raise
    logger.info("S3 Vectors initialized (bucket=%s)", bucket)


def _get_client():
    if _client is None:
        raise RuntimeError("vector store not initialized — call init_vector_store() at startup")
    return _client


def upsert_slot(
    slot: KnowledgeSlot,
    summary_vector: list[float],
    content_vector: list[float],
) -> None:
    client = _get_client()
    bucket = settings.AWS_VECTOR_BUCKET_NAME
    slot_id = str(slot.id)
    user_id = str(slot.user_id)
    metadata = {"user_id": user_id, "slot_type": slot.slot_type}

    client.put_vectors(
        vectorBucketName=bucket,
        indexName=_INDEX_SUMMARY,
        vectors=[{"key": slot_id, "data": {"float32": summary_vector}, "metadata": metadata}],
    )
    client.put_vectors(
        vectorBucketName=bucket,
        indexName=_INDEX_CONTENT,
        vectors=[{"key": slot_id, "data": {"float32": content_vector}, "metadata": metadata}],
    )


def search_slots(
    summary_query_vec: list[float],
    content_query_vec: list[float],
    user_id: str,
    top_k: int = 10,
) -> list[dict]:
    client = _get_client()
    bucket = settings.AWS_VECTOR_BUCKET_NAME

    summary_resp = client.query_vectors(
        vectorBucketName=bucket,
        indexName=_INDEX_SUMMARY,
        queryVector={"float32": summary_query_vec},
        topK=top_k,
        filter={"metadata": {"user_id": {"$eq": user_id}}},
        returnDistance=True,
    )
    content_resp = client.query_vectors(
        vectorBucketName=bucket,
        indexName=_INDEX_CONTENT,
        queryVector={"float32": content_query_vec},
        topK=top_k,
        filter={"metadata": {"user_id": {"$eq": user_id}}},
        returnDistance=True,
    )

    # Convert cosine distance to similarity score (1 - distance)
    scores: dict[str, float] = {}
    for item in summary_resp.get("vectors", []):
        scores[item["key"]] = 1.0 - item.get("distance", 1.0)
    for item in content_resp.get("vectors", []):
        sid = item["key"]
        score = 1.0 - item.get("distance", 1.0)
        scores[sid] = max(scores.get(sid, 0.0), score)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"slot_id": sid, "score": score} for sid, score in ranked]


def delete_slot(slot_id: str) -> None:
    client = _get_client()
    bucket = settings.AWS_VECTOR_BUCKET_NAME
    for index_name in _INDEXES:
        try:
            client.delete_vectors(
                vectorBucketName=bucket,
                indexName=index_name,
                keys=[slot_id],
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
