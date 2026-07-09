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


def test_reciprocal_rank_fusion():
    retriever = AdvancedRetriever(vector_store=MagicMock(), document_store=MagicMock(), reranker=MagicMock())
    
    doc1 = Document(page_content="Apple is a fruit", metadata={"source": "fruit.txt", "title": "Apple"})
    doc2 = Document(page_content="Banana is yellow", metadata={"source": "fruit.txt", "title": "Banana"})
    doc3 = Document(page_content="Orange is orange", metadata={"source": "fruit.txt", "title": "Orange"})
    
    dense_results = [doc1, doc2]
    sparse_results = [doc3, doc1]
    
    fused = retriever._reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=60, limit=3)
    
    assert len(fused) == 3
    assert fused[0].page_content == "Apple is a fruit"
    assert fused[1].page_content == "Orange is orange"
    assert fused[2].page_content == "Banana is yellow"


def test_token_budget_manager():
    from app.core.retrieval.token_budget import TokenBudgetManager
    budget_manager = TokenBudgetManager(model_name="llama-3.3-70b-versatile")
    text = "Hello world!"
    tokens = budget_manager.count_tokens(text)
    assert tokens > 0
    
    docs = [
        Document(page_content="This is the first document."),
        Document(page_content="This is the second document.")
    ]
    selected, total = budget_manager.select_documents(query="test", docs=docs, chat_history="")
    assert len(selected) == 2


@pytest.mark.asyncio
async def test_document_store_database():
    from app.core.storage.document_store import DocumentStore
    from unittest.mock import patch, AsyncMock
    
    mock_session = AsyncMock()
    mock_session.merge = AsyncMock()
    mock_session.commit = AsyncMock()
    
    mock_parent = MagicMock()
    mock_parent.id = "doc_1"
    mock_parent.page_content = "Parent page content"
    mock_parent.metadata_json = {"title": "Doc Title"}
    
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mock_parent])))
    mock_session.execute = AsyncMock(return_value=mock_result)
    
    with patch("app.core.storage.document_store.AsyncSessionLocal") as mock_session_local:
        mock_session_local.return_value.__aenter__.return_value = mock_session
        
        store = DocumentStore(store_type="database")
        
        doc = Document(page_content="Parent page content", metadata={"title": "Doc Title"})
        await store.mset([("doc_1", doc)])
        mock_session.merge.assert_called_once()
        mock_session.commit.assert_called_once()
        
        results = await store.mget(["doc_1"])
        assert len(results) == 1
        assert results[0].page_content == "Parent page content"
        assert results[0].metadata["title"] == "Doc Title"


@pytest.mark.asyncio
async def test_reranker_cohere_and_tei():
    from app.core.retrieval.reranker import BGEReranker
    from unittest.mock import patch, MagicMock
    import httpx
    
    # 1. Test Cohere Rerank API Mocking
    with patch("app.core.retrieval.reranker.settings") as mock_settings:
        mock_settings.RERANKER_PROVIDER = "cohere"
        mock_settings.COHERE_API_KEY = "mock_cohere_key"
        mock_settings.RERANK_TOP_K = 2
        
        reranker = BGEReranker()
        
        docs = [
            Document(page_content="Apple is a fruit", metadata={}),
            Document(page_content="Banana is yellow", metadata={})
        ]
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "results": [
                {"index": 1, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.85}
            ]
        })
        
        with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
            reranked = await reranker.rerank("query text", docs)
            
            assert len(reranked) == 2
            assert reranked[0].page_content == "Banana is yellow"
            assert reranked[0].metadata["rerank_score"] == 0.99
            assert reranked[1].page_content == "Apple is a fruit"
            assert reranked[1].metadata["rerank_score"] == 0.85
            mock_post.assert_called_once()
            
    # 2. Test TEI Rerank API Mocking
    with patch("app.core.retrieval.reranker.settings") as mock_settings:
        mock_settings.RERANKER_PROVIDER = "tei"
        mock_settings.TEI_API_URL = "http://tei-server.local"
        mock_settings.RERANK_TOP_K = 2
        
        reranker = BGEReranker()
        
        docs = [
            Document(page_content="Apple is a fruit", metadata={}),
            Document(page_content="Banana is yellow", metadata={})
        ]
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=[
            {"index": 0, "score": 0.95},
            {"index": 1, "score": 0.70}
        ])
        
        with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
            reranked = await reranker.rerank("query text", docs)
            
            assert len(reranked) == 2
            assert reranked[0].page_content == "Apple is a fruit"
            assert reranked[0].metadata["rerank_score"] == 0.95
            assert reranked[1].page_content == "Banana is yellow"
            assert reranked[1].metadata["rerank_score"] == 0.70
            mock_post.assert_called_once()
