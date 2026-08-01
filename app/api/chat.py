from fastapi import APIRouter
from app.services.agent_service import chatbot
from app.models.request import ChatRequest
from app.models.response import ChatResponse

router = APIRouter()

@router.get('/health')
def health():
    return {'Status': 'OK'}

@router.post('/chat',response_model=ChatResponse)
def chat(request:ChatRequest):
    input = {"messages":[{'role':'user','content':request.message}]}

    config = {"configurable":{"thread_id":request.thread_id}}

    result = chatbot.invoke(input, config)

    return ChatResponse(
        response=result["messages"][-1].content,
        thread_id=request.thread_id
    )
        
    