import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.agent_service import chatbot, get_conversation_messages, delete_conversation_history
from app.services.conversation_service import (
    create_conversation,
    delete_conversation,
    list_conversations,
    user_owns_thread,
)
from app.services.rag_service import store_document
from app.services.image_rag_service import store_image
from app.services.document_extraction import extract_content
from app.models.request import ChatRequest, CreateConversationRequest, UploadDocumentRequest
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


_ROLE_MAP = {"human": "user", "ai": "assistant"}  # LangChain message.type -> Gradio chat role


@router.get('/conversations/{thread_id}/messages')
async def get_conversation_history(thread_id: str, user_id: str):
    if not await user_owns_thread(user_id, thread_id):
        raise HTTPException(status_code=403, detail="thread_id does not belong to user_id")

    messages = await get_conversation_messages(thread_id)
    return [
        {"role": _ROLE_MAP[m.type], "content": m.content}
        for m in messages
        if m.type in _ROLE_MAP  # defensively drop anything that isn't a real chat turn
    ]


@router.delete('/conversations/{thread_id}')
async def delete_conversation_route(thread_id: str, user_id: str):
    if not await user_owns_thread(user_id, thread_id):
        raise HTTPException(status_code=403, detail="thread_id does not belong to user_id")

    deleted = await delete_conversation(user_id, thread_id)
    await delete_conversation_history(thread_id)

    return {"deleted": deleted}


@router.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not await user_owns_thread(request.user_id, request.thread_id):
        raise HTTPException(status_code=403, detail="thread_id does not belong to user_id")

    input = {
        "messages": [{'role': 'user', 'content': request.message}],
        "use_retrieval": request.use_retrieval,
        "user_id": request.user_id,
    }
    config = {"configurable": {"thread_id": request.thread_id}}

    result = await chatbot.ainvoke(input, config)

    return ChatResponse(
        response=result["messages"][-1].content,
        thread_id=request.thread_id
    )


@router.post('/documents')
async def upload_document(request: UploadDocumentRequest):
    return await store_document(request.text, request.title, request.user_id)


@router.post('/documents/upload')
async def upload_document_file(
    user_id: str = Form(...),
    source: str = Form(...),
    file: UploadFile = File(...),
):
    file_bytes = await file.read()
    try:
        text, images = extract_content(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    text_result = await store_document(text, source, user_id)

    images_stored = 0
    for image_bytes in images:
        if await store_image(image_bytes, source, user_id):
            images_stored += 1

    return {
        **text_result,
        "images_found": len(images),
        "images_stored": images_stored,
        "images_skipped_duplicate": len(images) - images_stored,
    }


@router.post('/images')
async def upload_image(
    user_id: str = Form(...),
    source: str = Form(...),
    file: UploadFile = File(...),
):
    # Multipart form data, not JSON — a real file upload can't be a Pydantic
    # JSON body the way every other endpoint here is. user_id/source travel
    # as form fields alongside the file in the same request.
    image_bytes = await file.read()
    was_new = await store_image(image_bytes, source, user_id)
    return {"stored": was_new}
