import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.container import get_container
from app.core.database import get_db
from app.schemas.query import QueryRequest, QueryResponse
import logging


router = APIRouter(prefix="/api/query", tags=["Query"])

logger = logging.getLogger(__name__)


@router.post("/stream")
async def ask_question_stream(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    container = Depends(get_container)
):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")

    async def event_generator():
        try:
            async for chunk in container.rag_service.query_stream(
                question=request.question,
                db=db,
                conversation_id=request.conversation_id
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
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

    result = await container.rag_service.query(
        question=request.question,
        db=db,
        conversation_id=request.conversation_id
    )

    return QueryResponse(**result)