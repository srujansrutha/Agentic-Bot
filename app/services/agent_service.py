import asyncio
import threading
import os

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import aiosqlite

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
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOllama(model=os.getenv("MODEL_NAME"),keep_alive=-1)


# ── Graph ─────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chatbot", chat_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

chatbot = graph.compile(checkpointer=checkpointer)
