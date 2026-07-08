import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from app.main import app
from app.core.database import get_db
from app.core.container import get_container


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.fixture
def mock_container():
    container = MagicMock()
    
    rag_service = MagicMock()
    rag_service.query = AsyncMock(return_value={
        "answer": "This is a mock answer from the evaluator.",
        "sources": [{"title": "Mock doc", "source": "mock_file.txt", "confidence": 0.9}],
        "retrieved_count": 1,
        "conversation_id": "12345678-1234-5678-1234-567812345678",
        "metadata": {}
    })
    
    async def mock_stream(*args, **kwargs):
        yield {"token": "Mock"}
        yield {"token": " response"}
        yield {
            "done": True,
            "sources": [{"title": "Mock doc", "source": "mock_file.txt", "confidence": 0.9}],
            "full_answer": "Mock response",
            "metadata": {"retrieved_count": 1, "used_tokens": 100}
        }
        
    rag_service.query_stream = mock_stream
    container.rag_service = rag_service
    
    conv_service = MagicMock()
    conv_service.create_conversation = AsyncMock(return_value="87654321-8765-4321-8765-432187654321")
    container.get_conversation_service = MagicMock(return_value=conv_service)
    
    return container


def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "healthy"}


def test_root():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "RAG Assistant" in res.json()["message"]


def test_query_endpoint(mock_db, mock_container):
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_container] = lambda: mock_container
    
    with TestClient(app) as client:
        payload = {"question": "What is machine learning?"}
        res = client.post("/api/query/", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "mock answer" in data["answer"].lower()
        assert data["retrieved_count"] == 1
        
    app.dependency_overrides.clear()


def test_query_stream_endpoint(mock_db, mock_container):
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_container] = lambda: mock_container
    
    with TestClient(app) as client:
        payload = {"question": "Explain deep learning."}
        res = client.post("/api/query/stream", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        content = res.text
        assert "Mock" in content
        assert "done" in content
        
    app.dependency_overrides.clear()


def test_conversations_new_endpoint(mock_db, mock_container):
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_container] = lambda: mock_container
    
    with TestClient(app) as client:
        res = client.post("/api/conversations/new")
        assert res.status_code == 200
        assert "conversation_id" in res.json()
        
    app.dependency_overrides.clear()
