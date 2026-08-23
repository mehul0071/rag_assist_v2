# 🧠 Enterprise Self-Correcting RAG Platform

> **Production-grade Retrieval-Augmented Generation system** with cyclic agentic self-correction, hybrid search, rolling memory, and semantic caching — built to demonstrate senior-level Generative AI engineering.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Cyclic_Agents-FF6B35?style=flat)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PGVector-336791?style=flat&logo=postgresql&logoColor=white)](https://pgvector.github.io)
[![Redis](https://img.shields.io/badge/Redis-Semantic_Cache-DC382D?style=flat&logo=redis&logoColor=white)](https://redis.io)
[![Tests](https://img.shields.io/badge/Tests-16_Passing-brightgreen?style=flat&logo=pytest)](./tests)
[![Observability](https://img.shields.io/badge/Observability-Prometheus_+_LangSmith-orange?style=flat)](https://smith.langchain.com)

---

## 📌 Project Overview

This platform goes well beyond a typical LLM wrapper. It implements a **stateful, cyclic agentic workflow** that can detect low-relevance retrievals, self-correct by rewriting queries, check generated answers for hallucinations, and fallback gracefully — all autonomously, without human intervention.

**Core architectural philosophy:** *Every component is designed for horizontal scalability, observability, and cost-efficiency in a production environment.*

### Why This Project Stands Out

| Capability | What It Demonstrates |
|---|---|
| **Cyclic CRAG Agents (LangGraph)** | Agentic orchestration beyond basic chains — self-healing, stateful loops |
| **Hybrid Search + RRF** | Dense (PGVector) + Sparse (FTS) fusion — not just basic similarity search |
| **BGE Cross-Encoder Reranking** | Production-grade multi-step retrieval pipeline, async-safe via ThreadPoolExecutor |
| **Hallucination & Groundedness Grader** | LLM-as-a-judge evaluation integrated into the pipeline at inference time |
| **Semantic Cache (NumPy + Redis)** | Sub-10ms cached responses, saving LLM API costs at scale |
| **Rolling Memory Compilation** | Token-aware summarization with persistent user facts extraction |
| **Full Observability Stack** | Prometheus metrics + LangSmith tracing + Ragas/DeepEval evaluation |
| **16-test Async Test Suite** | Production-level testing: mocked LangGraph cycles, async retrievers, Redis cache |

---

## 🏗️ System Architecture

The core is a **stateful, cyclic LangGraph workflow** with two self-correction loops: one for retrieval quality and one for generation groundedness.

```mermaid
graph TD
    A([User Query]) --> B{Semantic Cache Check}
    B -->|Cache Hit ≤10ms| Z([Return Cached Response])
    B -->|Cache Miss| C[Advanced Planner / Intent Classifier]

    C -->|Greeting / Chitchat| D[Direct LLM Generator]
    C -->|Knowledge Query| E[Query Rewriter / Expansion]

    E --> F[Hybrid Retriever]
    F --> F1[Dense Search: PGVector]
    F --> F2[Sparse Search: PostgreSQL FTS]
    F1 --> G[Reciprocal Rank Fusion - RRF]
    F2 --> G
    G --> H[BGE Cross-Encoder Reranker]

    H --> I{Document Grader Gate}
    I -->|Low Relevance — Retry < 2| E
    I -->|High Relevance OR Retry Maxed| J[Context Builder + Token Budget]

    J --> K[LLM Answer Generator]
    K --> L{Hallucination + Utility Grader}
    L -->|Grading Failed — Retry < 2| E
    L -->|Grading Passed OR Retry Maxed| M[Memory Pipeline]

    M --> M1[Extract & Persist User Facts]
    M --> M2[Rolling Chat Summarization]
    M1 --> N[Write-Through Semantic Cache]
    M2 --> N
    D --> N
    N --> Z
```

---

## 🌟 Key Engineering Features

### 1. 🔀 Hybrid Retrieval + Reciprocal Rank Fusion (RRF)

Implements a **two-stage retrieval** combining:

- **Dense Search:** High-dimensional vector search with `sentence-transformers` embeddings stored in **PostgreSQL PGVector**
- **Sparse Search:** Full-Text Search using PostgreSQL native `to_tsvector` / `ts_rank` indexes for exact keyword matching
- **Fusion:** Merged using a **score-agnostic RRF algorithm** that rewards documents appearing at the top of both ranking systems, without being vulnerable to score magnitude differences

> **Why this matters:** Hybrid search outperforms pure vector search by up to 30% on domain-specific corpora where keyword precision matters — a key metric in production RAG systems.

---

### 2. 🤖 Cyclic CRAG Agent Loops (LangGraph)

Built using **LangGraph stateful cyclic graphs**, the system implements **Corrective RAG (CRAG)**:

- **Document Grader Gate:** Each retrieved chunk is independently graded (LLM-as-a-judge with Pydantic structured output) before being passed downstream
- **Query Rewriter Loop:** If relevance grading fails, the original query is automatically rephrased and retrieval is re-attempted (max 2 cycles)
- **Hallucination + Utility Grader:** After generation, the answer is checked for groundedness against retrieved context AND whether it actually addresses the user's question
- **Controlled Fallback:** If max retries are hit, the system falls back gracefully to the best available context rather than failing

```python
# Structured output graders using Pydantic — no brittle string parsing
class GradeHallucination(BaseModel):
    binary_score: str = Field(
        description="Is the answer grounded in the retrieved documents? 'yes' or 'no'."
    )

structured_grader = llm.with_structured_output(GradeHallucination)
grade = await structured_grader.ainvoke([SystemMessage(...), HumanMessage(...)])
```

---

### 3. ⚡ Write-Through Semantic Cache (Redis + NumPy)

Implements a **two-layer semantic cache** to dramatically reduce API costs and response latency:

- **Pre-load:** On server startup, all cached query embeddings are loaded from Redis into a **local NumPy array in memory**
- **In-memory cosine similarity:** Incoming queries are compared against the NumPy array — no Redis round-trip needed for the matching step
- **Write-Through:** On cache miss, the result is embedded and written back to both the NumPy array and Redis atomically

**Verified Performance Results:**

| Query Type | Latency | Notes |
|---|---|---|
| Cache Miss (First Run) | ~29.57s | Full RAG + LLM pipeline |
| Exact Cache Hit | **8.2ms** | Served directly from Redis |
| Semantic Cache Hit (Rephrased) | **8.1ms** | Matched via cosine similarity |
| Cache Miss (Different Topic) | ~23.74s | Runs standard RAG |

---

### 4. 🧬 Multi-Tiered Reranking (GIL & ThreadPool Isolated)

- **Primary:** BGE Cross-Encoder model re-scores all RRF-fused candidate chunks for semantic precision
- **Async-Safe:** Cross-Encoder inference (PyTorch/CPU) runs inside a class-level `ThreadPoolExecutor` — completely isolated from the FastAPI async event loop, preventing GIL contention
- **Remote Fallback Support:** Plug-in async clients for **Cohere Rerank API** and **Hugging Face TEI** with automatic fallback to the local BGE model

```python
# Non-blocking local model inference in FastAPI async context
executor = ThreadPoolExecutor(max_workers=2)
scores = await asyncio.get_event_loop().run_in_executor(
    executor, self._run_cross_encoder_sync, pairs
)
```

---

### 5. 🧠 Stateful Rolling Memory + User Fact Extraction

Implements a **two-tier persistent memory system**:

- **Token-Aware Summarization:** Uses `tiktoken` to measure conversation history token count. When history approaches LLM context limits, older turns are summarized via LLM and stored in the database — protecting context window budget automatically
- **Structured Fact Extraction:** Uses Pydantic `FactUpdate` models with LLM structured outputs to extract user preferences and facts from each conversation turn. Checks for contradictions before persisting to a `user_facts` database profile — enabling personalized responses across sessions

---

### 6. 📊 Full Observability + Evaluation Stack

| Layer | Tool | What's Measured |
|---|---|---|
| **Metrics** | Prometheus + OpenTelemetry | Retrieval latency, LLM latency, input/output token counts, request rates |
| **Tracing** | LangSmith | Full LangGraph execution trace — every node, every LLM call |
| **RAG Evaluation** | Ragas + DeepEval | Faithfulness, context precision, answer relevancy |
| **Benchmark Dataset** | Custom 51-item QA set | ML, Deep Learning, NLP, and RAG architecture questions |

---

## 🧪 Test Suite

16 robust tests covering the most critical system paths:

```bash
PYTHONPATH=. pytest
```

```
======================== 16 passed, 8 warnings in 6.00s ========================
```

**Test Coverage Includes:**
- ✅ RRF ranking correctness and score fusion
- ✅ Async API reranker client (mocked Cohere + HuggingFace TEI)
- ✅ Redis semantic cache match/miss logic
- ✅ LangGraph cyclic loop simulation (mocked state transitions)
- ✅ Rolling memory pruning and token budget enforcement
- ✅ Database async CRUD operations for documents and conversations

---

## 📂 Project Structure

```
rag_assist_v2/
├── app/
│   ├── config/              # Pydantic-Settings environment config
│   ├── core/
│   │   ├── cache/           # Redis semantic cache (NumPy cosine similarity)
│   │   ├── context/         # Context builder + token budget manager
│   │   ├── graph/
│   │   │   ├── graph.py     # LangGraph state machine definition
│   │   │   ├── nodes.py     # All agent nodes (planner, retriever, graders)
│   │   │   └── state.py     # TypedDict GraphState schema
│   │   ├── memory/          # Rolling summarization + user facts extractor
│   │   ├── observability/   # Prometheus metrics + LangSmith tracing setup
│   │   ├── planner/         # Advanced intent classifier (structured output)
│   │   ├── prompts/         # Centralized prompt manager
│   │   └── retrieval/
│   │       ├── retriever.py      # Hybrid dense+sparse retriever
│   │       ├── reranker.py       # BGE Cross-Encoder + async remote clients
│   │       ├── pipeline.py       # Retrieval orchestration pipeline
│   │       ├── query_rewriter.py # LLM-powered query rewriting
│   │       └── token_budget.py   # Token-aware context window manager
│   ├── models/              # SQLAlchemy async ORM models
│   ├── repositories/        # Data access layer (PGVector + parent docs)
│   ├── routers/             # FastAPI route handlers (documents, query, eval)
│   ├── schemas/             # Pydantic request/response schemas
│   ├── services/            # Orchestration layer (RAGService)
│   └── main.py              # FastAPI lifespan, middleware, router registration
├── alembic/                 # Database migration history
├── evaluation_datasets/     # 51-item RAG evaluation benchmark
├── knowledge_base/          # Source documents for RAG ingestion
├── tests/
│   ├── test_rag.py          # 14 unit tests for core RAG components
│   └── test_routers.py      # 2 API integration tests
└── requirements.txt
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL with `pgvector` extension enabled
- Redis (running on port `6379`)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd rag_assist_v2

# 2. Create and activate a virtual environment
python3 -m venv myenv
source myenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Environment Configuration (`.env`)

```ini
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/rag_assistant

# Cache
REDIS_URL=redis://localhost:6379/0

# LLM Providers
GROQ_API_KEY=your_groq_api_key

# Optional: Remote Reranking (falls back to local BGE if not set)
COHERE_API_KEY=your_cohere_api_key

# Observability
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
```

### Run Database Migrations

```bash
alembic upgrade head
```

### Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

API documentation available at: `http://localhost:8080/docs`

---

## 🛣️ API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/ingest` | Ingest documents into the knowledge base |
| `POST` | `/conversations` | Create a new conversation session |
| `POST` | `/query` | Submit a query through the full CRAG pipeline |
| `GET` | `/conversations/{id}/history` | Retrieve conversation history |
| `POST` | `/evaluation/run` | Run Ragas/DeepEval evaluation suite |
| `GET` | `/metrics` | Prometheus metrics endpoint |
| `GET` | `/health` | Health check |

---

## 🧰 Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI + Uvicorn | Async REST API with middleware and lifespan hooks |
| **Agentic Orchestration** | LangGraph | Stateful cyclic agent graphs |
| **LLM Integration** | LangChain + Groq | LLM invocation, structured outputs, prompt management |
| **Vector Search** | PostgreSQL + PGVector | Dense vector similarity search |
| **Full-Text Search** | PostgreSQL FTS | Sparse lexical search (`to_tsvector`) |
| **Reranking** | BGE Cross-Encoder (local) + Cohere (remote) | Multi-tier candidate reranking |
| **Embedding** | sentence-transformers | Local embedding generation |
| **Semantic Cache** | Redis + NumPy | Sub-10ms query cache with cosine similarity |
| **Database ORM** | SQLAlchemy (async) + Asyncpg | Async database access |
| **Migrations** | Alembic | Schema versioning and migration management |
| **Observability** | Prometheus + OpenTelemetry + LangSmith | Metrics, tracing, cost tracking |
| **Evaluation** | Ragas + DeepEval | Faithfulness, relevance, and hallucination scoring |
| **Testing** | Pytest + AsyncMock | Async unit + integration tests |
| **Token Counting** | tiktoken | Precise context window management |

---

## 🧭 Design Decisions

### Why LangGraph over a simple LangChain chain?
Standard LangChain chains are acyclic — they process inputs in one pass. This system requires **cyclic execution**: if the grader rejects a retrieval, the graph must loop back to the retriever. LangGraph's explicit state machine model makes these retry loops transparent, debuggable, and testable.

### Why PostgreSQL for both vectors and relational data?
Maintaining PGVector (dense) and FTS (sparse) in a single PostgreSQL instance eliminates cross-service coordination overhead. Both retrieval paths share the same ACID transaction guarantees, and Alembic manages the schema lifecycle — no separate vector DB deployment needed.

### Why NumPy for in-memory cosine similarity over Redis vector search?
Redis vector search (RedisSearch/RediSearch) requires a specific Redis Stack build and adds query serialization overhead. Pre-loading cached embeddings into a NumPy array allows **microsecond-level matching without any network call** — effectively a zero-cost in-process cache for the most frequent queries.

### Why ThreadPoolExecutor for the Cross-Encoder?
PyTorch models release the GIL during compute-heavy operations but not during model loading or result processing. Wrapping all Cross-Encoder calls in a class-level `ThreadPoolExecutor` and `run_in_executor` ensures that local model inference never blocks the FastAPI event loop — critical for maintaining API responsiveness under concurrent load.