from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from app.config.settings import settings


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
        print(f"Added {len(documents)}, documents to vector store")


    async def as_retriever(self, k:int = None, filters: Optional[dict] = None):
        k = k or settings.RETRIEVER_K
        vectorstore = await self.get_vectorstore()
        return vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": filters}
        )