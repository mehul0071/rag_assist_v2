from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from sqlalchemy import text
from app.config.settings import settings
from app.core.database import AsyncSessionLocal


class VectorStore:

    def __init__(self, embeddings: Optional[HuggingFaceEmbeddings] = None):
        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.collection_name = settings.COLLECTION_NAME
        self.connection_string = settings.DATABASE_URL
        self._vectorstore: Optional[PGVector] = None

    async def get_vectorstore(self) -> PGVector:
        if self._vectorstore is None:
            self._vectorstore = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
                async_mode=True,
                create_extension=False,
            )
        return self._vectorstore
    
    async def add_documents(self, documents: List[Document]) -> None:
        vectorstore = await self.get_vectorstore()
        await vectorstore.aadd_documents(documents)
        print(f"Added {len(documents)} documents to vector store")

    async def as_retriever(self, k: int = None, filters: Optional[dict] = None):
        k = k or settings.RETRIEVER_K
        vectorstore = await self.get_vectorstore()
        return vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": filters}
        )

    async def sparse_search(self, query: str, k: int = 10) -> List[Document]:
        """PostgreSQL Full-Text Search on the langchain_pg_embedding table"""
        if not query:
            return []
            
        async with AsyncSessionLocal() as session:
            sql = text("""
                SELECT emb.document, emb.cmetadata, emb.id
                FROM langchain_pg_embedding emb
                JOIN langchain_pg_collection col ON emb.collection_id = col.uuid
                WHERE col.name = :collection_name
                  AND to_tsvector('english', emb.document) @@ plainto_tsquery('english', :query)
                ORDER BY ts_rank(to_tsvector('english', emb.document), plainto_tsquery('english', :query)) DESC
                LIMIT :limit
            """)
            
            result = await session.execute(
                sql,
                {
                    "collection_name": self.collection_name,
                    "query": query,
                    "limit": k
                }
            )
            
            documents = []
            for row in result.all():
                doc = Document(
                    page_content=row[0],
                    metadata=row[1] or {}
                )
                if "id" not in doc.metadata:
                    doc.metadata["uuid"] = row[2]
                documents.append(doc)
            return documents