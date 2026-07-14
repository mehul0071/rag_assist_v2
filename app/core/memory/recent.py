import logging
from typing import List
from uuid import UUID
from app.services.conversation_service import ConversationService

logger = logging.getLogger(__name__)


class RecentMemoryManager:
    """
    Manages retrieving recent raw messages and active conversation context.
    """

    async def get_recent_messages(
        self,
        conversation_id: UUID,
        conversation_service: ConversationService,
        limit: int = 5
    ) -> List[dict]:
        """
        Fetches the N most recent messages from the database repository.
        """
        messages = await conversation_service.repository.get_messages(
            conversation_id, limit=limit
        )
        return [
            {"id": m.id, "role": m.role, "content": m.content}
            for m in messages
        ]

    async def get_recent_context_text(
        self,
        conversation_id: UUID,
        conversation_service: ConversationService,
        limit: int = 5
    ) -> str:
        """
        Retrieves and formats the N most recent messages as a readable chat transcript.
        """
        recent = await self.get_recent_messages(conversation_id, conversation_service, limit=limit)
        return conversation_service.get_history_text(recent)
