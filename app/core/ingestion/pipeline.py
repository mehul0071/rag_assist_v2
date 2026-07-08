from app.config.settings import settings
from app.core.storage.document_store import DocumentStore
from app.core.storage.vector_store import VectorStore
from .metadata_extractor import MetadataExtractor
from .loader import DocumentLoader
from .splitter import DocumentSplitter


class IngestionPipeline:
    
    def __init__(
        self,
        vector_store: VectorStore,
        document_store: DocumentStore,
        loader: DocumentLoader = None,
        metadata_extractor: MetadataExtractor = None,
        splitter: DocumentSplitter = None
    ):
        self.vector_store = vector_store
        self.document_store = document_store
        self.loader = loader or DocumentLoader()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.splitter = splitter or DocumentSplitter()


    async def ingest_folder(self, folder_path: str = None) -> int:
        folder_path = folder_path or settings.KNOWLEDGE_BASE
        raw_docs = await self.loader.load_folder(folder_path)
        print(f"Loaded {len(raw_docs)}, raw documents")
        enriched_docs = await self.metadata_extractor.enrich_documents(raw_docs)
        print(f"============1=========")
        parent_docs, child_docs = self.splitter.split_parent_child_documents(enriched_docs)
        
        parent_id_pairs = [(doc.metadata["doc_id"], doc) for doc in parent_docs]
        await self.document_store.mset(parent_id_pairs)
        
        await self.vector_store.add_documents(child_docs)
        
        print(f"Ingestion completed: {len(parent_docs)} parent docs, {len(child_docs)} chunks")
        return len(child_docs)