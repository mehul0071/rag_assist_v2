from typing import List
from langchain_core.documents import Document
from app.config.llm import llm_config


class TokenBudgetManager:

    SYSTEM_PROMPT_BUDGET = 1200
    SAFETY_MARGIN = 800

    def __init__(self, model_name: str | None = None):
        self.model_spec = llm_config.get(model_name)

        self.max_input_tokens = (
            self.model_spec.context_window
            - self.model_spec.response_tokens
            - self.SYSTEM_PROMPT_BUDGET
            - self.SAFETY_MARGIN
        )

        if self.max_input_tokens <= 0:
            raise ValueError("Invalid token budget configuration.")


    def count_tokens(self, text: str) -> int:
        if not text:
            return 0

        return len(text) // 4 + 25


    def select_documents(
        self,
        query: str,
        docs: List[Document],
        chat_history: str = "",
    ) -> tuple[List[Document], int]:

        selected_docs = []
        total_tokens = (
            self.count_tokens(query)
            + self.count_tokens(chat_history)
        )

        for doc in docs:
            doc_tokens = self.count_tokens(doc.page_content)

            if total_tokens + doc_tokens > self.max_input_tokens:
                continue

            selected_docs.append(doc)
            total_tokens += doc_tokens

        return selected_docs, total_tokens