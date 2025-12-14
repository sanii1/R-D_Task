from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings

def get_llm():
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0
    )

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
