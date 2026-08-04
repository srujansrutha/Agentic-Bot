from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Conversation(BaseModel):
    user_id: str
    thread_id: str
    title: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
