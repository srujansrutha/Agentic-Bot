from pymongo import AsyncMongoClient

from app.config import settings

client = AsyncMongoClient(settings.mongo_uri)
db = client.get_database(settings.mongo_db_name)


async def ping() -> bool:
    try:
        await client.admin.command("ping")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(ping()))