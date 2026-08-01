from fastapi import FastAPI
from app.api.chat import router


app = FastAPI(title="srujans assistance")


app.include_router(router)