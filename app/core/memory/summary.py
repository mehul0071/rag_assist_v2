from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_service import LLMService


class ConversationSummarizer:

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm = llm_service or LLMService()


    async def summarize(self, current_summary: Optional[str], new_messages: List[dict]) -> str:
        if not new_messages:
            return current_summary or ""

        formatted_messages = []
        for msg in new_messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            formatted_messages.append(f"{role}: {content}")
        new_messages_text = "\n\n".join(formatted_messages)

        if current_summary:
            prompt = (
                f"You are an AI assistant tasked with updating an ongoing summary of the conversation.\n\n"
                f"Existing summary so far:\n{current_summary}\n\n"
                f"New messages to incorporate:\n{new_messages_text}\n\n"
                f"Write a new, cohesive, and concise summary of the conversation (max 150 words) that integrates the existing summary with the new context. Keep it in third-person narrative. Do NOT start with 'The conversation...'"
            )
        else:
            prompt = (
                f"You are an AI assistant tasked with summarizing the following conversation.\n\n"
                f"Conversation:\n{new_messages_text}\n\n"
                f"Write a cohesive, concise summary of the conversation (max 150 words). Keep it in third-person narrative. Do NOT start with 'The conversation...'"
            )

        messages = [
            SystemMessage(content="You are a precise summarization assistant. Provide clean, direct summaries without preamble."),
            HumanMessage(content=prompt)
        ]
        
        summary = await self.llm.generate_text(messages)
        return summary.strip()
