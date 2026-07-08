from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    stream: bool = False  


class Source(BaseModel):
    title: Optional[str]
    source: Optional[str]
    confidence: Optional[float]


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    retrieved_count: int