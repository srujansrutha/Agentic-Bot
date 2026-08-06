import asyncio
import threading

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import aiosqlite

from app.config import settings
from app.clients.llm_client import llm
from app.services.rag_service import retrieve_relevant_chunks

load_dotenv()

# ── Dedicated async loop for blocking async calls at module load time ──────────
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop (fire and forget)."""
    return _submit_async(coro)


# ── Checkpointer (SQLite-backed, persists conversation memory) ─────────────────
async def _init_checkpointer():
    conn = await aiosqlite.connect(database=settings.db_path)
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())


# ── Graph ─────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str  # no reducer — each turn's retrieval simply replaces the last one
    use_retrieval: bool
    user_id: str


async def retrieve_node(state: ChatState):
    latest_message = state["messages"][-1]
    chunks = await retrieve_relevant_chunks(latest_message.content, state.get("user_id"), top_k=3)
    context = "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in chunks)
    return {"context": context}


def chat_node(state: ChatState):
    messages = state["messages"]
    context = state.get("context", "")

    if context:
        instruction = SystemMessage(
            content=(
                "Use the following retrieved context to answer the user's question "
                "if it's relevant. Ignore it if it isn't.\n\n" + context
            )
        )
        prompt_messages = [instruction] + messages
    else:
        prompt_messages = messages

    response = llm.invoke(prompt_messages)
    return {"messages": [response]}


def route_after_start(state: ChatState) -> str:
    return "retrieve" if state.get("use_retrieval") else "chatbot"


graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("chatbot", chat_node)
graph.add_conditional_edges(START, route_after_start, {"retrieve": "retrieve", "chatbot": "chatbot"})
graph.add_edge("retrieve", "chatbot")
graph.add_edge("chatbot", END)

chatbot = graph.compile(checkpointer=checkpointer)


async def get_conversation_messages(thread_id: str) -> list[BaseMessage]:
    """Reads persisted turns for a thread straight from the checkpointer.
    An unknown thread_id isn't an error here — it just has no history yet
    (aget_state returns an empty snapshot, not a KeyError)."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await chatbot.aget_state(config)
    return snapshot.values.get("messages", [])


async def delete_conversation_history(thread_id: str) -> None:
    """Purges every checkpoint/write LangGraph has stored for this thread.
    Without this, deleting the Mongo ownership record only hides the
    conversation from the list — the actual message history would still
    sit in the SQLite checkpointer forever."""
    await checkpointer.adelete_thread(thread_id)
