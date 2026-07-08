from datetime import datetime
from typing import List
from langchain.agents.factory import _extract_metadata
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from app.config.settings import settings
from app.schemas.document import DocumentMetadata
from langchain_core.output_parsers import PydanticOutputParser


class MetadataExtractor:
    
    def __init__(self):
        self.llm = ChatGroq(model=settings.LLM_MODEL, temperature=0.3, api_key=settings.GROQ_API_KEY)
        self.parser = PydanticOutputParser(pydantic_object=DocumentMetadata)


    async def enrich_documents(self, docs: List[Document]) -> List[Document]:
        enriched = []

        for i in docs:
            try:
                metadata_dict = _extract_metadata(i)
                i.metadata.update(metadata_dict)
                enriched.append(i)
            except Exception as e:
                print(f"Metadata extraction failed for {i.metadata.get('source')}: {e}")
                fallback = {
                    "title": i.metadata.get("source", "Unknown Document").split("/")[-1],
                    "summary": "Metadata extraction failed.",
                    "document_type": "OTHER",
                    "author": None,
                    "language": "en",
                    "tags": ["unknown"],
                    "keywords": [],
                    "confidence": 0.3,
                    "extracted_at": datetime.now().isoformat()
                }
                i.metadata.update(fallback)
                enriched.append(i)
        
        return enriched