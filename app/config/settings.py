from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project
    PROJECT_NAME: str = "RAG Assistant"
    VERSION: str = "0.1.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/rag_assistant"
    COLLECTION_NAME: str = "knowledge_base"

    # LLM
    GROQ_API_KEY: str
    # LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_MODEL: str = "llama-3.3-70b-versatile" 

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Paths
    KNOWLEDGE_BASE: str = "./knowledge_base"
    METADATA_CACHE_DIR: str = "./metadata_cache"

    # Retrieval Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    RETRIEVER_K: int = 6

    # Token Limits
    MAX_TOKENS_TOTAL: int = 12000
    MAX_HISTORY_TOKENS: int = 4000

    # API
    API_V1_STR: str = "/api/v1"

    # Reranker Settings
    RERANKER_PROVIDER: str = "local"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_TOP_K: int = 5
    COHERE_API_KEY: str | None = None
    TEI_API_URL: str | None = None

    ENABLE_QUERY_REWRITE: bool = True

settings = Settings()

