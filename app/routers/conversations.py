from fastapi import APIRouter, Depends, HTTPException
from uuid import UUID
from app.core.database import get_db
from app.core.container import get_container
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.conversation.repository import ConversationRepository
from app.services.conversation_service import ConversationService
from app.schemas.conversation import ConversationResponse, ConversationHistoryResponse

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


@router.post("/new")
async def create_new_conversation(db: AsyncSession = Depends(get_db)):
    repo = ConversationRepository(db)
    service = ConversationService(repository=repo)
    
    conversation_id = await service.create_conversation()
    return {"conversation_id": str(conversation_id)}


@router.get("/all_conversation", response_model=list[ConversationResponse])
async def get_all_conv(db: AsyncSession = Depends(get_db)):
    repo = ConversationRepository(db)
    return await repo.get_all_conversations()


@router.get("/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    container = Depends(get_container)
):
    service = container.get_conversation_service(db)
    conv = await service.get_conversation_history(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv