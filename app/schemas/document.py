from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    BOOK = "book"
    ARTICLE = "article"
    REPORT = "report"
    NOTE = "note"
    RESEARCH = "research"
    OTHER = "other"
    

class DocumentMetadata(BaseModel):
    title: str = Field(..., description="Document title")
    summary: str = Field(..., description="LLM-generated summary")
    document_type: DocumentType
    author: Optional[str] = None
    language: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence of metadata extraction",
    )