from functools import lru_cache
from app.core.context.context_builder import ContextBuilder
from app.core.retrieval.pipeline import RetrievalPipeline
from app.core.storage.vector_store import VectorStore
from app.core.storage.document_store import DocumentStore
from app.core.retrieval.reranker import BGEReranker
from app.core.retrieval.retriever import AdvancedRetriever
from app.core.ingestion.pipeline import IngestionPipeline
from app.services.rag_service import RAGService
from app.services.conversation_service import ConversationService
from app.core.conversation.repository import ConversationRepository
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends


class Container:

    def __init__(self):
        self._vector_store = None
        self._document_store = None
        self._reranker = None
        self._retriever = None
        self._ingestion_pipeline = None
        self._rag_service = None

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def document_store(self) -> DocumentStore:
        if self._document_store is None:
            self._document_store = DocumentStore()
        return self._document_store

    @property
    def reranker(self) -> BGEReranker:
        if self._reranker is None:
            self._reranker = BGEReranker()
        return self._reranker

    @property
    def retriever(self) -> AdvancedRetriever:
        if self._retriever is None:
            self._retriever = AdvancedRetriever(
                vector_store=self.vector_store,
                document_store=self.document_store,
                reranker=self.reranker
            )
        return self._retriever

    @property
    def ingestion_pipeline(self) -> IngestionPipeline:
        if self._ingestion_pipeline is None:
            self._ingestion_pipeline = IngestionPipeline(
                vector_store=self.vector_store,
                document_store=self.document_store
            )
        return self._ingestion_pipeline

    @property
    def context_builder(self) -> ContextBuilder:
        if not hasattr(self, '_context_builder'):
            self._context_builder = ContextBuilder()
        return self._context_builder

    @property
    def rag_service(self) -> RAGService:
        if self._rag_service is None:
            self._rag_service = RAGService(
                retriever=self.retriever,
                ingestion_pipeline=self.ingestion_pipeline,
                conversation_service=self.get_conversation_service(),
                context_builder=self.context_builder
            )
        return self._rag_service
    
    @property
    def retrieval_pipeline(self) -> RetrievalPipeline:
        if not hasattr(self, '_retrieval_pipeline'):
            self._retrieval_pipeline = RetrievalPipeline(
                retriever=self.retriever,
                reranker=self.reranker
            )
        return self._retrieval_pipeline
    
    def get_conversation_repository(self, db: AsyncSession = Depends(get_db)):
        return ConversationRepository(db)

    def get_conversation_service(self, db: AsyncSession = Depends(get_db)):
        repo = self.get_conversation_repository(db)
        return ConversationService(repository=repo)

container = Container()

@lru_cache
def get_container() -> Container:
    return container