import json
import os
from typing import List, Optional
from langchain_core.documents import Document
from app.config.settings import settings


class DocumentStore:

    def __init__(self, store_type: str = "file"):
        self.store_type = store_type
        self.base_dir = settings.METADATA_CACHE_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    
    async def mset(self, key_value_pairs: List[tuple[str, Document]]) -> None:
        for k, doc in key_value_pairs:
            if self.store_type == "file":
                path = os.path.join(self.base_dir, f"{k}.json")
                data = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

    
    async def mget(self, keys: List[str]) -> List[Optional[Document]]:
        documents = []
        for k in keys:
            path = os.path.join(self.base_dir, f"{k}.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                doc = Document(
                    page_content=data["page_content"],
                    metadata=data["metadata"]
                )
                documents.append(doc)
            else:
                documents.append(None)
        return documents
    

    async def mdelete(self, keys: List[str]) -> None:
        for key in keys:
            path = os.path.join(self.base_dir, f"{key}.json")
            if os.path.exists(path):
                os.remove(path)

    
