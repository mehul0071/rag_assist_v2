from typing import List, Optional
from uuid import UUID
from app.core.conversation.repository import ConversationRepository


class ConversationService:
    
    def __init__(
        self, 
        repository: ConversationRepository = None
    ):
        self.repository = repository


    async def create_conversation(
        self
    ) -> UUID:
        conversation = await self.repository.create_conversation()
        return conversation.id


    
    async def get_history(
        self, 
        conversation_id: UUID, 
        limit: int = 8
    ) -> List[dict]:
        messages = await self.repository.get_messages(conversation_id, limit)
        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]


    async def add_message(
        self, 
        conversation_id: UUID, 
        role: str, 
        content: str
    ):
        await self.repository.add_message(conversation_id, role, content)


    def get_history_text(
        self, 
        messages: List[dict]
    ) -> str:
        if not messages:
            return ""
        history = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        return "\n\n".join(history)


    async def update_summary(self, conversation_id: UUID, summary: str):
        await self.repository.update_summary(conversation_id, summary)


    async def update_title(self, conversation_id: UUID, title: str):
        await self.repository.update_title(conversation_id, title)


    async def get_summary(self, conversation_id: UUID) -> Optional[str]:
        conv = await self.repository.get_conversation(conversation_id)
        return conv.summary if conv else None


    async def get_conversation_history(self, conversation_id: UUID):
        return await self.repository.get_conversation_with_messages(conversation_id)