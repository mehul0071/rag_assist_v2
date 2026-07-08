from typing import Dict, Any, Optional
from uuid import UUID
from app.core.retrieval.pipeline import RetrievalPipeline
from app.core.retrieval.retriever import AdvancedRetriever
from app.core.ingestion.pipeline import IngestionPipeline
from app.services.conversation_service import ConversationService
from app.core.context.context_builder import ContextBuilder
from app.core.prompts.prompt_manager import PromptManager
from app.services.llm_service import LLMService
from app.core.graph.graph import create_rag_graph


class RAGService:

    def __init__(
        self,
        retriever: AdvancedRetriever,
        ingestion_pipeline: IngestionPipeline,
        conversation_service: ConversationService,
        context_builder: ContextBuilder = None,
        prompt_manager: PromptManager = None,
        llm_service: LLMService = None,
        retrieval_pipeline: RetrievalPipeline = None
    ):
        self.retriever = retriever
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(retriever=retriever)
        self.retriever = retriever
        self.ingestion_pipeline = ingestion_pipeline
        self.retrieval_pipeline = RetrievalPipeline(retriever=retriever)
        self.conversation_service = conversation_service
        self.context_builder = ContextBuilder()
        self.prompt_manager = prompt_manager or PromptManager()
        self.llm_service = llm_service or LLMService()
        self.graph = create_rag_graph(self)


    async def ingest_folder(self, folder_path: str = None) -> Dict[str, Any]:
        return await self.ingestion_pipeline.ingest_folder(folder_path)


    async def query(
        self, 
        question: str, 
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        
        chat_history_text = ""
        if conversation_id:
            history = await self.conversation_service.get_history(UUID(conversation_id))
            chat_history_text = self.conversation_service.get_history_text(history)

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
            await self.conversation_service.add_message(
                UUID(conversation_id), "user", question
            )
            await self.conversation_service.add_message(
                UUID(conversation_id), "assistant", result.get("answer", "")
            )

        return {
            "answer": result.get("answer", "Sorry, I couldn't generate a response."),
            "sources": result.get("sources", []),
            "retrieved_count": len(result.get("retrieved_docs", [])),
            "conversation_id": str(conversation_id) if conversation_id else None,
            "metadata": result.get("metadata", {})
        }

    async def create_conversation(self) -> str:
        return str(await self.conversation_service.create_conversation())