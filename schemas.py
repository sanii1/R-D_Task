from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    use_hyde: bool = False

class ChatResponse(BaseModel):
    answer: str
    context: List[str]
    processing_time: float
