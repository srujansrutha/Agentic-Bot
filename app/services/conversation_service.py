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


async def delete_conversation(user_id: str, thread_id: str) -> bool:
    """Removes the ownership record and invalidates both caches that
    mention this thread. Returns False if there was nothing to delete.
    Doesn't touch the LangGraph checkpoint history — that's a separate
    system, purged separately by delete_conversation_history."""
    result = await collection.delete_one({"user_id": user_id, "thread_id": thread_id})

    # Both caches must go, not just the list one: owns:{user}:{thread} caches
    # "yes" for up to CACHE_TTL_SECONDS, which would let /chat keep writing
    # to a conversation we just told the user we deleted.
    await redis_client.delete(f"conversations:{user_id}")
    await redis_client.delete(f"owns:{user_id}:{thread_id}")

    return result.deleted_count > 0


async def user_owns_thread(user_id: str, thread_id: str) -> bool:
    cache_key = f"owns:{user_id}:{thread_id}"
    if await redis_client.get(cache_key):
        return True

    doc = await collection.find_one({"user_id": user_id, "thread_id": thread_id})
    owns = doc is not None

    if owns:
        await redis_client.set(cache_key, "1", ex=CACHE_TTL_SECONDS)

    return owns
