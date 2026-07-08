from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.settings import settings


class DocumentSplitter:
    
    def __init__(self):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )


    def split_parent_child_documents(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        parent_docs = []
        child_docs = []

        for i in docs:
            parent_chunks = self.parent_splitter.split_documents([i])

            for parent in parent_chunks:
                parent.metadata["doc_id"] = f"doc_{len(parent_docs)}"
                parent.metadata["is_parent"] = True
                parent_docs.append(parent)
                
                children = self.child_splitter.split_documents([parent])
                for child in children:
                    child.metadata.update({
                        "parent_id": parent.metadata["doc_id"],
                        "is_child": True
                    })
                    child_docs.append(child)
        
        return parent_docs, child_docs