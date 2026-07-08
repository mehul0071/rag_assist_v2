import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage
from app.core.container import get_container
from app.core.database import get_db
from app.core.conversation.repository import ConversationRepository
from app.schemas.query import QueryRequest, QueryResponse
import logging


router = APIRouter(prefix="/api/query", tags=["Query"])

logger = logger = logging.getLogger(__name__)


@router.post("/stream")
async def ask_question_stream(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    container = Depends(get_container)
):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")

    conversation_repo = ConversationRepository(db)

    async def event_generator():
        full_answer = ""
        try:
            chat_history = ""
            if request.conversation_id:
                messages = await conversation_repo.get_messages(request.conversation_id, limit=6)
                chat_history = "\n\n".join([
                    f"{m.role.capitalize()}: {m.content}" for m in messages
                ])

            # === Use same pipeline as LangGraph ===
            retrieved_docs = await container.rag_service.retrieval_pipeline.search(request.question)

            context_data = container.rag_service.context_builder.build_context(
                query=request.question,
                retrieved_docs=retrieved_docs,
                chat_history=chat_history
            )

            prompt_template = container.rag_service.prompt_manager.get_rag_prompt()
            prompt_value = prompt_template.format(
                chat_history=chat_history,
                context=context_data.get("formatted_context", ""),
                question=request.question
            )

            if request.conversation_id:
                await conversation_repo.add_message(
                    conversation_id=request.conversation_id,
                    role="user",
                    content=request.question
                )

            async for chunk in container.rag_service.llm_service.stream_generate([HumanMessage(content=prompt_value)]):
                if chunk:
                    full_answer += chunk
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                    await asyncio.sleep(0.015)

            if request.conversation_id:
                await conversation_repo.add_message(
                    conversation_id=request.conversation_id,
                    role="assistant",
                    content=full_answer
                )

            yield f"data: {json.dumps({
                'done': True, 
                'sources': context_data.get('sources', []),
                'full_answer': full_answer,
                'metadata': {
                    'retrieved_count': len(retrieved_docs),
                    'used_tokens': context_data.get('estimated_tokens')
                }
            })}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    container = Depends(get_container)
):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")

    conversation_service = container.get_conversation_service(db)
    container.rag_service.conversation_service = conversation_service

    result = await container.rag_service.query(
        question=request.question,
        conversation_id=request.conversation_id
    )

    return QueryResponse(**result)