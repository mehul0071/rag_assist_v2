import asyncio
import logging
import concurrent.futures
from typing import List, Optional
import httpx
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from app.config.settings import settings
from app.core.retrieval.base import BaseReranker

logger = logging.getLogger(__name__)


class BGEReranker(BaseReranker):

    _local_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="local_rerank_worker"
    )


    def __init__(self):
        self.provider = settings.RERANKER_PROVIDER.lower().strip()
        self.cross_encoder = None
        self.reranker = None
        
        if self.provider == "local":
            self.cross_encoder = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)
            self.reranker = CrossEncoderReranker(model=self.cross_encoder, top_n=settings.RERANK_TOP_K)
            logger.info("Local BGE Reranker initialized with model: %s", settings.RERANKER_MODEL)
        else:
            logger.info("Production Reranker initialized using remote provider: %s", self.provider)


    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        if self.provider == "cohere":
            return await self._rerank_cohere(query, documents)
        elif self.provider == "tei":
            return await self._rerank_tei(query, documents)
        else:
            return await self._rerank_local(query, documents)


    async def _rerank_local(self, query: str, documents: List[Document]) -> List[Document]:
        if self.reranker is None:
            self.cross_encoder = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)
            self.reranker = CrossEncoderReranker(model=self.cross_encoder, top_n=settings.RERANK_TOP_K)

        logger.debug("Reranking %d documents locally for query: %s", len(documents), query[:80])
        
        loop = asyncio.get_running_loop()
        reranked = await loop.run_in_executor(
            self._local_executor,
            self.reranker.compress_documents,
            documents,
            query,
        )
        return reranked


    async def _rerank_cohere(self, query: str, documents: List[Document]) -> List[Document]:
        api_key = settings.COHERE_API_KEY
        if not api_key:
            logger.warning("Cohere API key not set, falling back to local reranking.")
            return await self._rerank_local(query, documents)

        logger.debug("Reranking %d documents via Cohere API", len(documents))
        
        doc_texts = [doc.page_content for doc in documents]
        payload = {
            "model": "rerank-english-v3.0",
            "query": query,
            "documents": doc_texts,
            "top_n": settings.RERANK_TOP_K
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post("https://api.cohere.com/v1/rerank", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                reranked_docs = []
                for res in results:
                    idx = res["index"]
                    relevance_score = res["relevance_score"]
                    doc = documents[idx]
                    doc.metadata["rerank_score"] = relevance_score
                    reranked_docs.append(doc)
                return reranked_docs
        except Exception as e:
            logger.error("Cohere Rerank failed with error: %s. Falling back to local.", e)
            return await self._rerank_local(query, documents)


    async def _rerank_tei(self, query: str, documents: List[Document]) -> List[Document]:
        tei_url = settings.TEI_API_URL
        if not tei_url:
            logger.warning("TEI API URL not set, falling back to local reranking.")
            return await self._rerank_local(query, documents)

        logger.debug("Reranking %d documents via Hugging Face TEI API", len(documents))
        
        doc_texts = [doc.page_content for doc in documents]
        payload = {
            "query": query,
            "texts": doc_texts
        }
        
        try:
            url = f"{tei_url.rstrip('/')}/rerank"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                results = response.json()
                
                results = sorted(results, key=lambda x: x["score"], reverse=True)[:settings.RERANK_TOP_K]
                
                reranked_docs = []
                for res in results:
                    idx = res["index"]
                    score = res["score"]
                    doc = documents[idx]
                    doc.metadata["rerank_score"] = score
                    reranked_docs.append(doc)
                return reranked_docs
        except Exception as e:
            logger.error("TEI Rerank failed with error: %s. Falling back to local.", e)
            return await self._rerank_local(query, documents)