import logging
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user_facts import UserFact
from app.services.llm_service import LLMService
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class FactUpdate(BaseModel):
    new_facts: List[str] = Field(
        description="New facts to add about the user, extracted from the conversations. Each fact should be a short standalone sentence."
    )
    contradicted_ids: List[str] = Field(
        description="List of IDs of existing facts that are contradicted, outdated, or corrected by the new conversation segment and should be removed."
    )


class UserFactManager:

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()

    async def get_user_facts(self, conversation_id: UUID, db: AsyncSession) -> List[UserFact]:
        result = await db.execute(
            select(UserFact)
            .where(UserFact.conversation_id == conversation_id)
            .order_by(UserFact.created_at.asc())
        )
        return list(result.scalars().all())

    async def get_user_facts_text(self, conversation_id: UUID, db: AsyncSession) -> str:
        facts = await self.get_user_facts(conversation_id, db)
        if not facts:
            return "No profile details or preferences recorded yet."
        return "\n".join(f"- {f.fact}" for f in facts)

    async def extract_and_update_facts(
        self, 
        conversation_id: UUID, 
        db: AsyncSession, 
        new_messages: List[dict]
    ) -> None:
        """
        Extracts new facts from recent messages and checks against existing facts
        for updates or contradictions, applying the changes to the database.
        """
        if not new_messages:
            return

        existing_facts = await self.get_user_facts(conversation_id, db)
        
        # Format existing facts with their IDs so the LLM can reference them for deletion
        existing_facts_text = ""
        if existing_facts:
            existing_facts_text = "\n".join(
                f"ID: {f.id} | Fact: {f.fact}" for f in existing_facts
            )
        else:
            existing_facts_text = "No facts recorded yet."

        # Format new messages segment
        segment_text = "\n".join(
            f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in new_messages
        )

        system_prompt = (
            "You are a user profile memory agent. Your job is to extract facts, preferences, background, "
            "or persistent goals about the User from the latest conversation segment.\n\n"
            "Here are the existing facts recorded about the user:\n"
            f"{existing_facts_text}\n\n"
            "Compare any new facts you find in the new conversation segment with existing ones. "
            "If the user changes their preference or contradicts an existing fact, mark the old fact ID for deletion.\n"
            "Ensure extracted facts are concise, standalone sentences about the user (e.g. 'User prefers Python over JavaScript'). "
            "Do NOT store general Q&A context or transient details."
        )

        user_prompt = f"New conversation segment:\n{segment_text}"

        try:
            structured_llm = self.llm_service.llm.with_structured_output(FactUpdate)
            update = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            new_facts = []
            contradicted_ids = []
            if isinstance(update, FactUpdate):
                new_facts = update.new_facts
                contradicted_ids = update.contradicted_ids

            logger.info(
                f"Extracted user facts update: added {len(new_facts)} new, deleted {len(contradicted_ids)} contradicted."
            )

            # 1. Delete contradicted facts
            if contradicted_ids:
                try:
                    uuid_ids = [UUID(cid.strip()) for cid in contradicted_ids if cid.strip()]
                    if uuid_ids:
                        await db.execute(
                            delete(UserFact).where(
                                UserFact.id.in_(uuid_ids),
                                UserFact.conversation_id == conversation_id
                            )
                        )
                except Exception as ex:
                    logger.warning(f"Error parsing UUIDs for fact deletion: {ex}")

            # 2. Add new facts
            for fact_text in new_facts:
                clean_fact = fact_text.strip()
                if clean_fact:
                    db_fact = UserFact(
                        conversation_id=conversation_id,
                        fact=clean_fact
                    )
                    db.add(db_fact)

            if contradicted_ids or new_facts:
                await db.commit()

        except Exception as e:
            logger.error(f"Error extracting and updating user facts: {e}")
