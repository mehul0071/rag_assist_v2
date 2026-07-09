import logging
from typing import Dict, List, Optional
from collections import defaultdict
from langchain_core.documents import Document
from app.core.storage.vector_store import VectorStore
from app.core.storage.document_store import DocumentStore
from app.core.retrieval.reranker import BGEReranker

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    
    def __init__(self, vector_store: VectorStore, document_store: DocumentStore, reranker: Optional[BGEReranker] = None):
        self.vector_store = vector_store
        self.document_store = document_store
        self.reranker = reranker or BGEReranker()


    async def retrieve(self, query: str, k: int = 6) -> List[Document]:
        try:
            base_retriever = await self.vector_store.as_retriever(k=k * 3)
            dense_docs = await base_retriever.ainvoke(query)
        except Exception as e:
            logger.error("Dense retrieval failed: %s", e)
            dense_docs = []

        try:
            sparse_docs = await self.vector_store.sparse_search(query, k=k * 3)
        except Exception as e:
            logger.error("Sparse retrieval failed: %s", e)
            sparse_docs = []

        fused_docs = self._reciprocal_rank_fusion(dense_docs, sparse_docs, limit=k * 3)
        logger.info("RRF: combined dense (%d docs) and sparse (%d docs) to retrieve %d docs", len(dense_docs), len(sparse_docs), len(fused_docs))

        if not fused_docs:
            return []

        reranked = await self.reranker.rerank(query, fused_docs)
        
        unique_docs = self._deduplicate_documents(reranked, max_per_source=2)
        return unique_docs[:k]

    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
        rrf_k: int = 60,
        limit: int = 12
    ) -> List[Document]:
        rrf_scores = {}
        doc_map = {}

        def add_ranks(results):
            for rank, doc in enumerate(results):
                source = doc.metadata.get("source", "")
                title = doc.metadata.get("title", "")
                key = f"{doc.page_content}::{source}::{title}"
                
                if key not in doc_map:
                    doc_map[key] = doc
                
                if key not in rrf_scores:
                    rrf_scores[key] = 0.0
                
                rrf_scores[key] += 1.0 / (rrf_k + rank)

        add_ranks(dense_results)
        add_ranks(sparse_results)

        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys[:limit]]

    
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
            parents = await self.document_store.mget(parent_ids)
            return [p for p in parents if p is not None]
        
        return chunks