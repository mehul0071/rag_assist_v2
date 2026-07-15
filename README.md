# 🚀 Enterprise Self-Correcting RAG Platform

A production-grade, highly optimized, and stateful Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, and **PostgreSQL/Redis**. 

Designed to demonstrate advanced Generative AI engineering patterns (e.g., hybrid search, cyclic self-correction agent loops, rolling window memory compilation, and custom semantic caches) targeting high-throughput and sub-10ms cached response requirements.

---

## 🏗️ System Architecture

The platform uses a stateful, cyclic workflow orchestrated by **LangGraph** to evaluate search relevance, self-correct queries, and grade generated outputs:

```mermaid
graph TD
    A[User Query] --> B{Advanced Planner / Classifier}
    B -->|Greeting| C[Direct LLM Generator]
    B -->|Knowledge Query| D[Query Rewriter / Expansion]
    D --> E[Hybrid Retrieval dense PGVector + sparse FTS]
    E --> F[Reciprocal Rank Fusion RRF]
    F --> G[BGE Cross-Encoder Reranking]
    G --> H{Document Grader Gate}
    H -->|Low Relevance / Retry < 2| D
    H -->|High Relevance / Retry Maxed| I[Context Builder]
    I --> J[LLM Answer Generator]
    J --> K{Hallucination & Groundedness Grader}
    K -->|Failed check / Retry < 2| D
    K -->|Passed check / Retry Maxed| L[Extract & Update User Facts]
    L --> M[Rolling Chat Summarization]
    M --> N[Return Response & Cache in Redis]
    C --> N
```

---

## 🌟 Key Engineering Features

### 1. Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
* **Dense search**: High-dimensional vector search utilizing PostgreSQL **PGVector** and `sentence-transformers`.
* **Sparse search**: Lexical search utilizing PostgreSQL Full-Text Search (FTS) indexes (`to_tsvector` and `ts_rank`).
* **Fusion**: Merged rank scores using a score-agnostic **RRF** algorithm, combining conceptual similarity and keyword specificity.

### 2. Multi-Tiered Reranking (GIL & Threadpool Isolated)
* Uses a **BGE Cross-Encoder** model to re-score candidate chunks.
* Local PyTorch models are run inside a class-level `ThreadPoolExecutor` to prevent blocking the FastAPI event loop under concurrency.
* Out-of-the-box support for async remote endpoints (**Cohere Rerank API** and **Hugging Face TEI**) with automatic local model fallback.

### 3. Cyclic Orchestration & Corrective RAG (CRAG)
* Stateful agent loops built using **LangGraph**.
* Includes a structured **AdvancedPlanner** (using Pydantic parsing) that classifies intent, focus areas, and web search targets.
* Employs document relevance grading, query rephrasing loopbacks, and answer groundedness/hallucination checks to guarantee response reliability.

### 4. Database-Backed Document Store & Rolling Memory
* Stateless server design storing parent metadata documents in a relational PostgreSQL table (`parent_documents`), enabling horizontal scaling.
* **Rolling Summarization**: Evaluates conversation history tokens using `tiktoken`. When history exceeds limits, older messages are summarized, saved to the database, and pruned to protect LLM context windows.
* **User Memory**: Uses structured outputs (`FactUpdate` Pydantic models) to extract user preferences, check contradictions, and persist them in a `user_facts` profile database table.

### 5. Write-Through Semantic Cache (Redis + NumPy)
* Connects to a standard local/remote Redis server.
* Pre-loads cached query vectors into a local NumPy array at startup.
* Performs microsecond-level **in-memory Cosine Similarity checks** using **NumPy** on incoming queries.
* Serves exact and semantically similar queries in **under 10ms**, bypassing the RAG/LLM pipeline completely and saving massive API costs.

### 6. Observability & Evaluation Rig
* **Metrics**: Prometheus instrumentation exposing system latencies, request counts, and token counters.
* **Observability**: LangSmith tracing configuration.
* **Evaluation**: Built-in runners for **Ragas** and **DeepEval** to compute faithfulness, context precision, and answer relevancy.
* **Testing Dataset**: A 51-item benchmark covering machine learning, deep learning, NLP, and RAG architectures.

---

## ⚡ Quick Start

### ⚙️ Prerequisites
Ensure you have the following running locally:
* **Python 3.12+**
* **PostgreSQL** (with `pgvector` extension enabled)
* **Redis** (running on port `6379`)

### 📦 Installation
1. Clone the repository and navigate to the directory:
   ```bash
   git clone <repo-url>
   cd rag_assistant
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in `.env`:
   ```ini
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/rag_assistant
   REDIS_URL=redis://localhost:6379/0
   GROQ_API_KEY=your_groq_api_key
   COHERE_API_KEY=your_optional_cohere_key
   ```

---

## 🧪 Verification & Testing

### 1. Run Automated Test Suite
The project features 16 robust unit tests covering RRF rankings, database storage, async API reranking clients, memory pruning, Redis semantic matching, and LangGraph cyclic loops:
```bash
PYTHONPATH=. pytest
```
Output:
```bash
======================== 16 passed, 8 warnings in 6.00s ========================
```

### 2. Verify Redis Semantic Cache Latency
Verify semantic hits and response speeds using the custom cache test script:
```bash
PYTHONPATH=. python verify_redis_cache.py
```
Output metrics:
* **Query 1 (Cache Miss - First run)**: `Time taken: 29.57s` (processed complete RAG/LLM graph).
* **Query 2 (Exact Cache Hit)**: `Time taken: 0.0082s` (**8.2 milliseconds**, served from Redis).
* **Query 3 (Semantic Cache Hit - Rephrased)**: `Time taken: 0.0081s` (**8.1 milliseconds**, matched via cosine similarity, bypassed LLM).
* **Query 4 (Cache Miss - Different Topic)**: `Time taken: 23.74s` (runs standard RAG).

---

## 📂 Project Structure

```
├── app/
│   ├── config/          # Environment variables & settings
│   ├── core/
│   │   ├── cache/       # Redis Semantic Cache (NumPy similarity)
│   │   ├── context/     # RAG context builders & sources
│   │   ├── graph/       # LangGraph state & node connections
│   │   ├── memory/      # Rolling summarization & user facts
│   │   ├── planner/     # Advanced Query Classifier
│   │   ├── retrieval/   # Dense+Sparse retrieval, RRF, & Rerankers
│   │   └── storage/     # PGVector & parent document repositories
│   ├── models/          # SQL Alchemy database schemas
│   ├── services/        # Orchestration layer (RAGService)
│   └── main.py          # FastAPI app routers
├── tests/               # Test suites (16 tests)
├── requirements.txt     # Python packages
└── verify_redis_cache.py # Cache verification script
```