from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from app.config.settings import settings


class DocumentLoader:

    async def load_folder(self, folder_path:str) -> List[Document]:

        documents = []

        loaders = [
            DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
            DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
            DirectoryLoader(folder_path, glob="**/*.md", loader_cls=TextLoader, show_progress=True),
        ]

        for loader in loaders:
            try:
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading with {loader}: {e}")
        
        return documents