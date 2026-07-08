from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: List[Document],) -> List[Document]:
        raise NotImplementedError