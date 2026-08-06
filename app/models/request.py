
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_id: str
    use_retrieval: bool = False


class CreateConversationRequest(BaseModel):
    user_id: str
    title: str | None = None


class UploadDocumentRequest(BaseModel):
    user_id: str
    title: str
    text: str