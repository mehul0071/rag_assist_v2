from typing import List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.conversations import Conversation
from app.models.messages import Message


class ConversationRepository:
    
    def __init__(self, db: AsyncSession):
        self.db = db


    async def get_conversation(self, conversation_id: UUID):
        result = await self.db.execute(
            select(Conversation).
            where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()
    

    async def get_all_conversations(self):
        result = await self.db.execute(
            select(Conversation)
            .order_by(Conversation.created_at)
        )
        conv = result.scalars().all()
        print(f"============conv==================={conv}")
        return conv


    async def get_messages(self, conversation_id: UUID, limit: int = 10) -> List[Message]:
        result = await self.db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        return result.scalars().all()


    async def create_conversation(self) -> Conversation:
        conversation = Conversation()
        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation


    async def add_message(self, conversation_id: UUID, role: str, content: str):
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content
        )
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        return message


    async def update_summary(self, conversation_id: UUID, summary: str):
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.summary = summary
            await self.db.commit()
            await self.db.refresh(conversation)
        return conversation