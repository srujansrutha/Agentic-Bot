from pydantic import BaseModel


class DocumentChunk(BaseModel):
    text: str
    source: str
    embedding: list[float]
    content_hash: str
    user_id: str | None = None  # None = shared/global doc, otherwise a personal upload
