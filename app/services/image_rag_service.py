import base64
import hashlib
import io

import numpy as np
from PIL import Image

from app.clients.clip_client import clip_model
from app.clients.mongo_client import db
from app.models.image_document import ImageChunk

collection = db["image_chunks"]

# Cross-modal (text-vs-image) cosine similarity from CLIP runs much lower in
# absolute terms than same-modality text-vs-text similarity from a dedicated
# text embedding model — 0.5 (our text-RAG threshold) would filter out even
# genuine matches here. Calibrated empirically instead of guessed: on a small
# real test, unrelated text-image pairs (including a totally unrelated query)
# scored ~0.15-0.26, genuine matches scored ~0.32-0.33. 0.25 sits between them.
IMAGE_MIN_SIMILARITY = 0.25


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_vec, b_vec = np.array(a), np.array(b)
    return float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def store_image(image_bytes: bytes, source: str, user_id: str | None = None) -> bool:
    """Embeds and stores one image. Returns False if this exact image
    already exists in this scope (dedup by raw byte hash)."""
    content_hash = _hash_bytes(image_bytes)
    existing = await collection.find_one({"content_hash": content_hash, "user_id": user_id})
    if existing:
        return False

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    vector = clip_model.encode(image).tolist()

    chunk = ImageChunk(
        image_base64=base64.b64encode(image_bytes).decode(),
        source=source,
        embedding=vector,
        content_hash=content_hash,
        user_id=user_id,
    )
    await collection.insert_one(chunk.model_dump())
    return True


async def retrieve_relevant_images(
    query_text: str, user_id: str | None, top_k: int = 3, min_similarity: float = IMAGE_MIN_SIMILARITY
) -> list[ImageChunk]:
    # CLIP's text encoder — same call as image encoding, lands in the same space
    query_vector = clip_model.encode(query_text).tolist()

    mongo_filter = {"$or": [{"user_id": None}, {"user_id": user_id}]}
    docs = [ImageChunk(**doc) async for doc in collection.find(mongo_filter)]
    if not docs:
        return []

    scored = [(_cosine_similarity(query_vector, doc.embedding), doc) for doc in docs]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [doc for score, doc in scored[:top_k] if score >= min_similarity]
