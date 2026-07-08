from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from app.config.settings import settings


class LLMService:
\
    def __init__(self, model: Optional[BaseChatModel] = None):
        self.llm = model or ChatGroq(
            model=settings.LLM_MODEL,
            temperature=0.2,
            api_key=settings.GROQ_API_KEY,
            max_tokens=1024,
        )


    async def generate(self, messages: List[BaseMessage]) -> AIMessage:
        response: AIMessage = await self.llm.ainvoke(messages)
        return response


    async def generate_text(self, messages: List[BaseMessage]) -> str:
        response = await self.generate(messages)
        return response.content
    

    async def stream_generate(self, messages: List[BaseMessage]):
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content