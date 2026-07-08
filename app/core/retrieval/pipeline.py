from typing import List, Optional
from langchain_core.documents import Document
import logging
from app.config.settings import settings
from .retriever import AdvancedRetriever
from .reranker import BGEReranker
from .query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)
        

class RetrievalPipeline:
    
    def __init__(
        self,
        retriever: Optional[AdvancedRetriever] = None,
        reranker: Optional[BGEReranker] = None,
        rewriter: Optional[QueryRewriter] = None
    ):
        self.retriever = retriever
        self.reranker = reranker or BGEReranker()
        self.rewriter = rewriter or QueryRewriter()
        self.enable_rewrite = settings.ENABLE_QUERY_REWRITE
        logger.info("RetrievalPipeline initialized")


    async def search(self, query: str) -> List[Document]:
        try:
            final_query = await self.rewriter.rewrite(query)
            
            docs = await self.retriever.retrieve(final_query, k=12)
            
            if docs and self.reranker:
                docs = await self.reranker.rerank(final_query, docs)
            
            logger.info(f"Pipeline returned {len(docs)} docs for query: {query[:60]}...")
            return docs
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return await self.retriever.retrieve(query, k=6)