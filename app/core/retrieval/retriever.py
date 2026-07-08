from typing import Dict, List, Optional
from collections import defaultdict
from langchain_core.documents import Document
from app.core.storage.vector_store import VectorStore
from app.core.storage.document_store import DocumentStore
from app.core.retrieval.reranker import BGEReranker


class AdvancedRetriever:
    
    def __init__(self, vector_store: VectorStore, document_store: DocumentStore, reranker: Optional[BGEReranker] = None):
        self.vector_store = vector_store
        self.document_store = document_store
        self.reranker = reranker or BGEReranker()


    async def retrieve(self, query: str, k: int = 6) -> List[Document]:
        base_retriever = await self.vector_store.as_retriever(k=k * 3)
        docs = await base_retriever.ainvoke(query)

        if not docs:
            return []

        reranked = await self.reranker.rerank(query, docs)
        unique_docs = self._deduplicate_documents(reranked, max_per_source=2)
        return unique_docs[:k]

    
    def _deduplicate_documents(self, docs: List[Document], max_per_source: int = 2) -> List[Document]:
        seen_sources: Dict[str, int] = defaultdict(int)
        unique_docs = []

        for doc in docs:
            source = doc.metadata.get("source") or doc.metadata.get("title", "unknown")
            
            seen_sources[source] += 1
            
            if seen_sources[source] <= max_per_source:
                unique_docs.append(doc)

        return unique_docs
    

    async def retrieve_with_parent(self, query: str) -> List[Document]:
        chunks = await self.retrieve(query, k=8)
        parent_ids = list({
            chunk.metadata.get("parent_id") 
            for chunk in chunks 
            if chunk.metadata.get("parent_id")
        })
        
        if parent_ids:
            return await self.document_store.mget(parent_ids)
        
        return chunks