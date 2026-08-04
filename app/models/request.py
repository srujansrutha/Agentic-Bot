
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_id: str


class CreateConversationRequest(BaseModel):
    user_id: str
    title: str | None = None