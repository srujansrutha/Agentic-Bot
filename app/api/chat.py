import uuid

from fastapi import APIRouter, HTTPException

from app.services.agent_service import chatbot
from app.services.conversation_service import create_conversation, list_conversations, user_owns_thread
from app.models.request import ChatRequest, CreateConversationRequest
from app.models.response import ChatResponse
from app.models.conversation import Conversation

router = APIRouter()


@router.get('/health')
def health():
    return {'Status': 'OK'}


@router.post('/conversations', response_model=Conversation)
async def start_conversation(request: CreateConversationRequest):
    thread_id = str(uuid.uuid4())
    return await create_conversation(request.user_id, thread_id, request.title)


@router.get('/conversations', response_model=list[Conversation])
async def get_conversations(user_id: str):
    return await list_conversations(user_id)


@router.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not await user_owns_thread(request.user_id, request.thread_id):
        raise HTTPException(status_code=403, detail="thread_id does not belong to user_id")

    input = {"messages": [{'role': 'user', 'content': request.message}]}
    config = {"configurable": {"thread_id": request.thread_id}}

    result = await chatbot.ainvoke(input, config)

    return ChatResponse(
        response=result["messages"][-1].content,
        thread_id=request.thread_id
    )
