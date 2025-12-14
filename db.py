import weaviate
from app.core.config import settings

def get_weaviate_client():
    return weaviate.Client(
        url=settings.WEAVIATE_URL
    )
