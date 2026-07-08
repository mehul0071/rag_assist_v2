import asyncio
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from app.config.settings import settings
from app.core.retrieval.base import BaseReranker

logger = logging.getLogger(__name__)


class BGEReranker(BaseReranker):

    def __init__(self):
        self.cross_encoder = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)
        self.reranker = CrossEncoderReranker(model=self.cross_encoder, top_n=settings.RERANK_TOP_K,)
        logger.info("BGE Reranker initialized with model: %s", settings.RERANKER_MODEL)


    async def rerank(self, query: str, documents: List[Document],) -> List[Document]:
        if not documents:
            return []

        logger.debug("Reranking %d documents for query: %s", len(documents), query[:80])
        reranked = await asyncio.to_thread(
            self.reranker.compress_documents,
            documents,
            query,
        )

        logger.debug("Reranking complete. Returning top %d documents.", len(reranked))
        return reranked