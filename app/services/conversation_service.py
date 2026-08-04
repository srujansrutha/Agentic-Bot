import json

from app.clients.mongo_client import db
from app.clients.redis_client import client as redis_client
from app.models.conversation import Conversation

collection = db["conversations"]

CACHE_TTL_SECONDS = 60


async def create_conversation(user_id: str, thread_id: str, title: str | None = None) -> Conversation:
    conversation = Conversation(user_id=user_id, thread_id=thread_id, title=title)
    await collection.insert_one(conversation.model_dump())
    await redis_client.delete(f"conversations:{user_id}")
    return conversation


async def list_conversations(user_id: str) -> list[Conversation]:
    cache_key = f"conversations:{user_id}"
    cached = await redis_client.get(cache_key)
    if cached is not None:
        return [Conversation(**doc) for doc in json.loads(cached)]

    cursor = collection.find({"user_id": user_id})
    conversations = [Conversation(**doc) async for doc in cursor]

    payload = json.dumps([c.model_dump(mode="json") for c in conversations])
    await redis_client.set(cache_key, payload, ex=CACHE_TTL_SECONDS)

    return conversations


async def user_owns_thread(user_id: str, thread_id: str) -> bool:
    cache_key = f"owns:{user_id}:{thread_id}"
    if await redis_client.get(cache_key):
        return True

    doc = await collection.find_one({"user_id": user_id, "thread_id": thread_id})
    owns = doc is not None

    if owns:
        await redis_client.set(cache_key, "1", ex=CACHE_TTL_SECONDS)

    return owns
