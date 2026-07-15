from typing import List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
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
            .options(selectinload(Conversation.messages))
            .order_by(Conversation.created_at)
        )
        conv = result.scalars().all()
        
        for c in conv:
            if not c.title:
                user_msgs = [m.content for m in c.messages if m.role == "user"]
                if user_msgs:
                    first_query = user_msgs[0]
                    words = first_query.split()
                    c.title = " ".join(words[:4]) + ("..." if len(words) > 4 else "")
                else:
                    c.title = "New Conversation"
                    
            if not c.summary:
                c.summary = "No summary available yet."
                
            if c.updated_at is None:
                c.updated_at = c.created_at
                
        conv_dicts = [c.__dict__ for c in conv]
        print(f"============conv==================={conv_dicts}")
        
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


    async def update_title(self, conversation_id: UUID, title: str):
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.title = title
            await self.db.commit()
            await self.db.refresh(conversation)
        return conversation


    async def get_conversation_with_messages(self, conversation_id: UUID) -> Optional[Conversation]:
        result = await self.db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        c = result.scalar_one_or_none()
        
        if c:
            if not c.title:
                user_msgs = [m.content for m in c.messages if m.role == "user"]
                if user_msgs:
                    first_query = user_msgs[0]
                    words = first_query.split()
                    c.title = " ".join(words[:4]) + ("..." if len(words) > 4 else "")
                else:
                    c.title = "New Conversation"
            if not c.summary:
                c.summary = "No summary available yet."
            if c.updated_at is None:
                c.updated_at = c.created_at
                
            c.messages.sort(key=lambda m: m.created_at)
            
        return c