import asyncio
from datetime import datetime
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from app.config.settings import settings
from app.schemas.document import DocumentMetadata


class MetadataExtractor:
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.LLM_MODEL, 
            temperature=0.0, 
            api_key=settings.GROQ_API_KEY
        )
        self.structured_llm = self.llm.with_structured_output(DocumentMetadata)


    async def extract_single_document_metadata(self, doc: Document) -> Document:
        prompt = f"""You are an expert document metadata extractor.
        Extract structured metadata for the following document text fragment.

        Document Content (Snippet):
        {doc.page_content[:1500]}
        """
        try:
            metadata: DocumentMetadata = await self.structured_llm.ainvoke(prompt)
            doc.metadata.update(metadata.model_dump())
            doc.metadata["extracted_at"] = datetime.now().isoformat()
        except Exception as e:
            print(f"Metadata extraction failed for {doc.metadata.get('source')}: {e}")
            fallback = {
                "title": doc.metadata.get("source", "Unknown Document").split("/")[-1],
                "summary": "Metadata extraction failed.",
                "document_type": "other",
                "author": None,
                "language": "en",
                "tags": ["unknown"],
                "keywords": [],
                "confidence": 0.3,
                "extracted_at": datetime.now().isoformat()
            }
            doc.metadata.update(fallback)
        return doc


    async def enrich_documents(self, docs: List[Document]) -> List[Document]:
        tasks = [self.extract_single_document_metadata(doc) for doc in docs]
        return list(await asyncio.gather(*tasks))