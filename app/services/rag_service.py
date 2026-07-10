from typing import Dict, Any, Optional
from uuid import UUID
import time
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.retrieval.pipeline import RetrievalPipeline
from app.core.retrieval.retriever import AdvancedRetriever
from app.core.ingestion.pipeline import IngestionPipeline
from app.services.conversation_service import ConversationService
from app.core.context.context_builder import ContextBuilder
from app.core.prompts.prompt_manager import PromptManager
from app.services.llm_service import LLMService
from app.core.graph.graph import create_rag_graph
from app.core.observability.metrics import rag_requests_total, rag_latency
from app.core.memory.manager import MemoryManager


class RAGService:

    def __init__(
        self,
        retriever: AdvancedRetriever,
        ingestion_pipeline: IngestionPipeline,
        conversation_service: Optional[ConversationService] = None,
        context_builder: Optional[ContextBuilder] = None,
        prompt_manager: Optional[PromptManager] = None,
        llm_service: Optional[LLMService] = None,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
        memory_manager: Optional[Any] = None
    ):
        self.retriever = retriever
        self.ingestion_pipeline = ingestion_pipeline
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(retriever=retriever)
        self.conversation_service = conversation_service
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_manager = prompt_manager or PromptManager()
        self.llm_service = llm_service or LLMService()
        self.memory_manager = memory_manager or MemoryManager()
        self.graph = create_rag_graph(self)


    async def ingest_folder(self, folder_path: str = None) -> Dict[str, Any]:
        return await self.ingestion_pipeline.ingest_folder(folder_path)


    async def query(
        self, 
        question: str, 
        db: AsyncSession,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        rag_requests_total.inc()
        start_time = time.perf_counter()
        
        from app.core.conversation.repository import ConversationRepository
        conversation_repo = ConversationRepository(db)
        conversation_service = ConversationService(repository=conversation_repo)

        chat_history_text = ""
        if conversation_id:
            chat_history_text = await self.memory_manager.get_history_with_summary(
                UUID(conversation_id), conversation_service
            )

        result = await self.graph.ainvoke({
            "question": question,
            "conversation_id": conversation_id,
            "chat_history": chat_history_text,
            "intent": "knowledge",
            "retrieved_docs": [],
            "answer": None,
            "sources": [],
            "metadata": {}
        })

        if conversation_id:
            await conversation_service.add_message(
                UUID(conversation_id), "user", question
            )
            await conversation_service.add_message(
                UUID(conversation_id), "assistant", result.get("answer", "")
            )

        duration = time.perf_counter() - start_time
        rag_latency.observe(duration)

        return {
            "answer": result.get("answer", "Sorry, I couldn't generate a response."),
            "sources": result.get("sources", []),
            "retrieved_count": len(result.get("retrieved_docs", [])),
            "conversation_id": str(conversation_id) if conversation_id else None,
            "metadata": result.get("metadata", {})
        }

    async def query_stream(
        self,
        question: str,
        db: AsyncSession,
        conversation_id: Optional[str] = None
    ):
        rag_requests_total.inc()
        start_time = time.perf_counter()

        from app.core.conversation.repository import ConversationRepository
        conversation_repo = ConversationRepository(db)
        conversation_service = ConversationService(repository=conversation_repo)

        chat_history_text = ""
        if conversation_id:
            chat_history_text = await self.memory_manager.get_history_with_summary(
                UUID(conversation_id), conversation_service
            )

        retrieved_docs = await self.retrieval_pipeline.search(question)

        context_data = self.context_builder.build_context(
            query=question,
            retrieved_docs=retrieved_docs,
            chat_history=chat_history_text
        )

        prompt_template = self.prompt_manager.get_rag_prompt()
        prompt_value = prompt_template.format(
            chat_history=chat_history_text,
            context=context_data.get("formatted_context", ""),
            question=question
        )

        if conversation_id:
            await conversation_service.add_message(
                UUID(conversation_id), "user", question
            )

        from langchain_core.messages import HumanMessage
        import asyncio
        full_answer = ""
        async for chunk in self.llm_service.stream_generate([HumanMessage(content=prompt_value)]):
            if chunk:
                full_answer += chunk
                yield {"token": chunk}
                await asyncio.sleep(0.015)

        if conversation_id:
            await conversation_service.add_message(
                UUID(conversation_id), "assistant", full_answer
            )

        duration = time.perf_counter() - start_time
        rag_latency.observe(duration)

        yield {
            "done": True,
            "sources": context_data.get("sources", []),
            "full_answer": full_answer,
            "metadata": {
                "retrieved_count": len(retrieved_docs),
                "used_tokens": context_data.get("estimated_tokens")
            }
        }

    async def create_conversation(self, db: AsyncSession) -> str:
        from app.core.conversation.repository import ConversationRepository
        repo = ConversationRepository(db)
        service = ConversationService(repository=repo)
        return str(await service.create_conversation())