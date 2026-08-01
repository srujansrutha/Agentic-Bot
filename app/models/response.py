from pydantic import BaseModel

class ChatResponse(BaseModel):
    thread_id: str
    response : str