import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from app.services.rag_service import RAGService
from app.core.retrieval.retriever import AdvancedRetriever
from app.core.ingestion.pipeline import IngestionPipeline
from langchain_core.documents import Document


@pytest.fixture
def mock_db():
    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    db.execute = AsyncMock(return_value=mock_result)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_rag_query_greeting(mock_db):
    retrieval_pipeline = MagicMock()
    retrieval_pipeline.search = AsyncMock(return_value=[])

    llm_service = MagicMock()
    llm_service.generate_text = AsyncMock(return_value="Hello! How can I help you today?")

    retriever = MagicMock(spec=AdvancedRetriever)
    ingestion_pipeline = MagicMock(spec=IngestionPipeline)
    
    rag_service = RAGService(
        retriever=retriever,
        ingestion_pipeline=ingestion_pipeline,
        llm_service=llm_service,
        retrieval_pipeline=retrieval_pipeline
    )
    
    result = await rag_service.query(
        question="Hello, how are you?",
        db=mock_db,
        conversation_id=None
    )
    
    assert "hello" in result["answer"].lower()
    assert result["retrieved_count"] == 0


@pytest.mark.asyncio
async def test_rag_query_knowledge(mock_db):
    mock_docs = [
        Document(page_content="Supervised learning is label-based.", metadata={"title": "ML Basics", "source": "doc1.txt"})
    ]
    retrieval_pipeline = MagicMock()
    retrieval_pipeline.search = AsyncMock(return_value=mock_docs)

    llm_service = MagicMock()
    llm_service.generate_text = AsyncMock(return_value="Supervised learning uses labels.")

    retriever = MagicMock(spec=AdvancedRetriever)
    ingestion_pipeline = MagicMock(spec=IngestionPipeline)
    
    rag_service = RAGService(
        retriever=retriever,
        ingestion_pipeline=ingestion_pipeline,
        llm_service=llm_service,
        retrieval_pipeline=retrieval_pipeline
    )
    
    result = await rag_service.query(
        question="What is supervised learning?",
        db=mock_db,
        conversation_id=str(uuid4())
    )
    
    assert "supervised" in result["answer"].lower()
    assert result["retrieved_count"] == 1
    assert len(result["sources"]) == 1
    assert result["sources"][0]["title"] == "ML Basics"
