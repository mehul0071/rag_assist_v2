from collections import defaultdict
from typing import Any
from langchain_core.documents import Document
from app.core.retrieval.token_budget import TokenBudgetManager


class ContextBuilder:

    MAX_CHUNKS_PER_SOURCE = 2

    def __init__(self):
        self.token_manager = TokenBudgetManager()

    def build_context(
        self,
        query: str,
        retrieved_docs: list[Document],
        chat_history: str = "",
    ) -> dict[str, Any]:

        if not retrieved_docs:
            return self._empty_context(query, chat_history)
        
        grouped = defaultdict(list)

        for doc in retrieved_docs:
            source = (
                doc.metadata.get("source")
                or doc.metadata.get("title")
                or "unknown"
            )

            grouped[source].append(doc)

        candidate_docs = []

        for docs in grouped.values():
            candidate_docs.extend(docs[: self.MAX_CHUNKS_PER_SOURCE])

        selected_docs, used_tokens = self.token_manager.select_documents(
            query=query,
            docs=candidate_docs,
            chat_history=chat_history,
        )

        context_parts = []

        for i, doc in enumerate(selected_docs, start=1):
            title = doc.metadata.get("title", "Document")
            context_parts.append(
                f"""
                Source {i} — {title}
                {doc.page_content.strip()}
                """
                )

        formatted_context = "\n\n---\n\n".join(context_parts)

        sources = []
        seen = set()

        for doc in selected_docs:

            source = (
                doc.metadata.get("source")
                or doc.metadata.get("title")
            )

            if source in seen:
                continue
            seen.add(source)
            sources.append(
                {
                    "title": doc.metadata.get("title"),
                    "source": source,
                    "page": doc.metadata.get("page"),
                    "confidence": doc.metadata.get("confidence"),
                }
            )

        return {
            "query": query,
            "chat_history": chat_history,
            "formatted_context": formatted_context,
            "context": formatted_context,
            "selected_documents": selected_docs,
            "sources": sources,
            "num_sources": len(sources),
            "total_chunks_used": len(selected_docs),
            "estimated_tokens": used_tokens,
            "remaining_tokens": (
                self.token_manager.max_input_tokens
                - used_tokens
            ),
            "truncated": len(selected_docs) < len(candidate_docs),
        }


    def _empty_context(
        self,
        query: str,
        chat_history: str,
    ) -> dict[str, Any]:

        used_tokens = (
            self.token_manager.count_tokens(query)
            + self.token_manager.count_tokens(chat_history)
        )

        return {
            "query": query,
            "chat_history": chat_history,
            "formatted_context": "No relevant documents found.",
            "context": "",
            "selected_documents": [],
            "sources": [],
            "num_sources": 0,
            "total_chunks_used": 0,
            "estimated_tokens": used_tokens,
            "remaining_tokens": (
                self.token_manager.max_input_tokens
                - used_tokens
            ),
            "truncated": False,
        }