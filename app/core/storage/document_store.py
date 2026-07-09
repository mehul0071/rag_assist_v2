import json
import os
from typing import List, Optional
from langchain_core.documents import Document
from sqlalchemy import select
from app.config.settings import settings
from app.core.database import AsyncSessionLocal
from app.models.parent_documents import ParentDocument


class DocumentStore:

    def __init__(self, store_type: str = "database"):
        self.store_type = store_type
        self.base_dir = settings.METADATA_CACHE_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    async def mset(self, key_value_pairs: List[tuple[str, Document]]) -> None:
        if self.store_type == "file":
            for k, doc in key_value_pairs:
                path = os.path.join(self.base_dir, f"{k}.json")
                data = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            async with AsyncSessionLocal() as session:
                for k, doc in key_value_pairs:
                    parent_doc = ParentDocument(
                        id=k,
                        page_content=doc.page_content,
                        metadata_json=doc.metadata
                    )
                    await session.merge(parent_doc)
                await session.commit()

    async def mget(self, keys: List[str]) -> List[Optional[Document]]:
        if self.store_type == "file":
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
        else:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ParentDocument).where(ParentDocument.id.in_(keys))
                )
                db_docs = {d.id: d for d in result.scalars().all()}
                
                documents = []
                for k in keys:
                    db_doc = db_docs.get(k)
                    if db_doc:
                        documents.append(Document(
                            page_content=db_doc.page_content,
                            metadata=db_doc.metadata_json or {}
                        ))
                    else:
                        documents.append(None)
                return documents

    async def mdelete(self, keys: List[str]) -> None:
        if self.store_type == "file":
            for key in keys:
                path = os.path.join(self.base_dir, f"{key}.json")
                if os.path.exists(path):
                    os.remove(path)
        else:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ParentDocument).where(ParentDocument.id.in_(keys))
                )
                for db_doc in result.scalars().all():
                    await session.delete(db_doc)
                await session.commit()

    
