import time
import asyncio
import logging
from uuid import UUID
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage
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
from app.core.planner.planner import AdvancedPlanner
from app.core.cache.semantic_cache import RedisSemanticCache
from app.core.conversation.repository import ConversationRepository
from app.core.conversation.repository import ConversationRepository


logger = logging.getLogger(__name__)


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
        memory_manager: Optional[Any] = None,
        planner: Optional[Any] = None,
        semantic_cache: Optional[Any] = "DEFAULT"
    ):
        self.retriever = retriever
        self.ingestion_pipeline = ingestion_pipeline
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(retriever=retriever)
        self.conversation_service = conversation_service
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_manager = prompt_manager or PromptManager()
        self.llm_service = llm_service or LLMService()
        self.memory_manager = memory_manager or MemoryManager()
        self.planner = planner or AdvancedPlanner(llm_service=self.llm_service)
        
        if semantic_cache == "DEFAULT":
            embeddings = None
            if hasattr(self.retriever, "vector_store") and hasattr(self.retriever.vector_store, "embeddings"):
                embeddings = self.retriever.vector_store.embeddings
            self.semantic_cache = RedisSemanticCache(embeddings=embeddings)
        else:
            self.semantic_cache = semantic_cache
            
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
        
        if self.semantic_cache:
            cache_hit = await self.semantic_cache.get(question)
            if cache_hit:
                cached_answer, cached_sources = cache_hit
                duration = time.perf_counter() - start_time
                rag_latency.observe(duration)
                logger.info("Semantic cache HIT served in %.4fs", duration)
                return {
                    "answer": cached_answer,
                    "sources": cached_sources,
                    "retrieved_count": len(cached_sources),
                    "conversation_id": str(conversation_id) if conversation_id else None,
                    "metadata": {"cached": True, "cache_hit_latency": f"{duration:.4f}s"}
                }

        conversation_repo = ConversationRepository(db)
        conversation_service = ConversationService(repository=conversation_repo)

        chat_history_text = ""
        user_facts_text = "No profile details or preferences recorded yet."
        if conversation_id:
            chat_history_text = await self.memory_manager.get_history_with_summary(
                UUID(conversation_id), conversation_service
            )
            user_facts_text = await self.memory_manager.user_facts.get_user_facts_text(
                UUID(conversation_id), db
            )

        result = await self.graph.ainvoke({
            "question": question,
            "conversation_id": conversation_id,
            "chat_history": chat_history_text,
            "user_facts": user_facts_text,
            "intent": "knowledge",
            "retrieved_docs": [],
            "answer": None,
            "sources": [],
            "metadata": {}
        })

        answer = result.get("answer", "Sorry, I couldn't generate a response.")
        sources = result.get("sources", [])

        if self.semantic_cache and result.get("intent") != "greeting" and result.get("answer"):
            await self.semantic_cache.set(question, answer, sources)

        if conversation_id:
            await conversation_service.add_message(
                UUID(conversation_id), "user", question
            )
            await conversation_service.add_message(
                UUID(conversation_id), "assistant", answer
            )
            new_turn = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            await self.memory_manager.user_facts.extract_and_update_facts(
                UUID(conversation_id), db, new_turn
            )

        duration = time.perf_counter() - start_time
        rag_latency.observe(duration)

        return {
            "answer": answer,
            "sources": sources,
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

        if self.semantic_cache:
            cache_hit = await self.semantic_cache.get(question)
            if cache_hit:
                cached_answer, cached_sources = cache_hit
                duration = time.perf_counter() - start_time
                rag_latency.observe(duration)
                logger.info("Semantic cache HIT (streaming) served in %.4fs", duration)
                
                words = cached_answer.split(" ")
                for i, word in enumerate(words):
                    suffix = " " if i < len(words) - 1 else ""
                    yield {"token": word + suffix}
                    await asyncio.sleep(0.015)
                    
                yield {
                    "done": True,
                    "sources": cached_sources,
                    "full_answer": cached_answer,
                    "metadata": {"cached": True, "cache_hit_latency": f"{duration:.4f}s"}
                }
                return

        conversation_repo = ConversationRepository(db)
        conversation_service = ConversationService(repository=conversation_repo)

        chat_history_text = ""
        user_facts_text = "No profile details or preferences recorded yet."
        if conversation_id:
            chat_history_text = await self.memory_manager.get_history_with_summary(
                UUID(conversation_id), conversation_service
            )
            user_facts_text = await self.memory_manager.user_facts.get_user_facts_text(
                UUID(conversation_id), db
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
            user_facts=user_facts_text,
            context=context_data.get("formatted_context", ""),
            question=question
        )

        if conversation_id:
            await conversation_service.add_message(
                UUID(conversation_id), "user", question
            )
        
        full_answer = ""
        async for chunk in self.llm_service.stream_generate([HumanMessage(content=prompt_value)]):
            if chunk:
                full_answer += chunk
                yield {"token": chunk}
                await asyncio.sleep(0.015)

        if self.semantic_cache and full_answer:
            await self.semantic_cache.set(question, full_answer, context_data.get("sources", []))

        if conversation_id:
            await conversation_service.add_message(
                UUID(conversation_id), "assistant", full_answer
            )
            new_turn = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_answer}
            ]
            await self.memory_manager.user_facts.extract_and_update_facts(
                UUID(conversation_id), db, new_turn
            )

        duration = time.perf_counter() - start_time
        rag_latency.observe(duration)

        yield {
            "done": True,
            "sources": context_data.get("sources", []),
            "full_answer": full_answer,
            "metadata": {
                "retrieved_count": len(retrieved_docs),
                "used_tokens": context_data.get("estimated_tokens"),
                "latency_seconds": duration
            }
        }

    async def create_conversation(self, db: AsyncSession) -> str:
        from app.core.conversation.repository import ConversationRepository
        repo = ConversationRepository(db)
        service = ConversationService(repository=repo)
        return str(await service.create_conversation())