import hashlib

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.clients.llm_client import embeddings
from app.clients.mongo_client import db
from app.models.document import DocumentChunk

collection = db["rag_chunks"]

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_vec, b_vec = np.array(a), np.array(b)
    return float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


async def store_chunk(text: str, source: str, user_id: str | None = None) -> bool:
    """Embeds and stores one chunk. Returns False (and stores nothing) if
    this exact text already exists in this scope (global, or this user's)."""
    content_hash = _hash_text(text)
    existing = await collection.find_one({"content_hash": content_hash, "user_id": user_id})
    if existing:
        return False

    vector = embeddings.embed_query(text)
    chunk = DocumentChunk(
        text=text, source=source, embedding=vector, content_hash=content_hash, user_id=user_id
    )
    await collection.insert_one(chunk.model_dump())
    return True


async def store_document(text: str, source: str, user_id: str | None = None) -> dict:
    """Splits a longer document into chunks and stores each, skipping any
    that already exist (by content hash) in this scope."""
    stored = skipped = 0
    for chunk_text in _splitter.split_text(text):
        if await store_chunk(chunk_text, source, user_id):
            stored += 1
        else:
            skipped += 1
    return {"chunks_stored": stored, "chunks_skipped_duplicate": skipped}


async def retrieve_relevant_chunks(
    query: str, user_id: str | None, top_k: int = 3, min_similarity: float = 0.5
) -> list[DocumentChunk]:
    query_vector = embeddings.embed_query(query)

    # a user can retrieve from shared/global docs (user_id: null) plus their own uploads
    mongo_filter = {"$or": [{"user_id": None}, {"user_id": user_id}]}
    docs = [DocumentChunk(**doc) async for doc in collection.find(mongo_filter)]
    if not docs:
        return []

    scored = [(_cosine_similarity(query_vector, doc.embedding), doc) for doc in docs]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [doc for score, doc in scored[:top_k] if score >= min_similarity]
