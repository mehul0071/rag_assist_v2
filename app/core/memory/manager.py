import logging
from uuid import UUID
from typing import Optional
from sqlalchemy import delete
from app.config.settings import settings
from app.models.messages import Message
from app.core.memory.summary import ConversationSummarizer
from app.services.conversation_service import ConversationService
from app.core.retrieval.token_budget import TokenBudgetManager

logger = logging.getLogger(__name__)


class MemoryManager:

    def __init__(self, summarizer: Optional[ConversationSummarizer] = None, token_budget: Optional[TokenBudgetManager] = None):
        self.summarizer = summarizer or ConversationSummarizer()
        self.token_budget = token_budget or TokenBudgetManager()


    async def get_history_with_summary(
        self, 
        conversation_id: UUID, 
        conversation_service: ConversationService
    ) -> str:
        """
        Fetches conversation history. If the token count of raw messages 
        exceeds MAX_HISTORY_TOKENS, it triggers a rolling summarization, 
        updates the database conversation summary, prunes summarized raw messages, 
        and returns the formatted history containing the summary + remaining raw messages.
        """
        raw_messages = await conversation_service.repository.get_messages(conversation_id, limit=100)
        if not raw_messages:
            return ""

        history_dicts = [
            {"id": m.id, "role": m.role, "content": m.content}
            for m in raw_messages
        ]

        history_text = conversation_service.get_history_text(history_dicts)
        history_tokens = self.token_budget.count_tokens(history_text)

        existing_summary = await conversation_service.get_summary(conversation_id)

        if history_tokens > settings.MAX_HISTORY_TOKENS and len(raw_messages) > 4:
            logger.info("Chat history tokens (%d) exceed limit (%d). Summarizing older messages...", history_tokens, settings.MAX_HISTORY_TOKENS)
            
            to_summarize = history_dicts[:-4]
            remaining = history_dicts[-4:]

            new_summary = await self.summarizer.summarize(existing_summary, to_summarize)
            await conversation_service.update_summary(conversation_id, new_summary)
            logger.info("Updated conversation summary stored in database.")

            ids_to_delete = [m["id"] for m in to_summarize]
            db = conversation_service.repository.db
            
            await db.execute(
                delete(Message).where(Message.id.in_(ids_to_delete))
            )
            await db.commit()
            logger.info("Deleted %d older messages from database messages table.", len(ids_to_delete))

            recent_text = conversation_service.get_history_text(remaining)
            formatted_history = f"System: Summary of the conversation so far: {new_summary}\n\n{recent_text}"
            return formatted_history
        else:
            raw_text = conversation_service.get_history_text(history_dicts)
            if existing_summary:
                return f"System: Summary of the conversation so far: {existing_summary}\n\n{raw_text}"
            return raw_text
