from pydantic import BaseModel


class ImageChunk(BaseModel):
    image_base64: str
    source: str
    embedding: list[float]
    content_hash: str
    user_id: str | None = None  # None = shared/global, otherwise a personal upload
