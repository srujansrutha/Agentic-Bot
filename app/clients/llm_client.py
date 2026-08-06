from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import settings

llm = ChatOllama(model=settings.model_name, keep_alive=-1)
embeddings = OllamaEmbeddings(model=settings.embedding_model)
