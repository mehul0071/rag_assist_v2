import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class QueryPlan(BaseModel):
    intent: str = Field(
        description="The primary intent of the query. Must be 'greeting' (for hello/general chitchat) or 'retrieval' (for knowledge questions requiring search)."
    )
    suggested_search_queries: List[str] = Field(
        default_factory=list,
        description="A list of 1-2 optimized search query variations for retrieval (leave empty for greetings)."
    )
    requires_web_search: bool = Field(
        default=False,
        description="True if the query requires external web search (e.g., real-time events, current year facts), False for static knowledge base retrieval."
    )
    focus_topics: List[str] = Field(
        default_factory=list,
        description="Core semantic entities or concepts focused on in the user query."
    )


class AdvancedPlanner:

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()

    async def plan_query(self, query: str, chat_history: str = "") -> QueryPlan:
        """
        Semantically analyzes the query, classifies intent, extracts key focus areas, 
        suggests rewritten search terms, and selects dynamic retrieval strategies.
        """
        try:
            logger.info("AdvancedPlanner analyzing query: '%s'", query[:60])
            
            system_prompt = (
                "You are an expert query analysis planner. Your job is to analyze the user's question, "
                "classify its intent, identify key topics, suggest optimized RAG search variations, and "
                "determine if web search is needed (e.g., for current events or real-time information)."
            )
            
            user_prompt = f"Query: {query}\n\nChat History (if any):\n{chat_history}"
            
            structured_llm = self.llm_service.llm.with_structured_output(QueryPlan)
            plan = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            logger.info("AdvancedPlanner successful: intent=%s, queries=%s, web_search=%s", 
                        plan.intent, plan.suggested_search_queries, plan.requires_web_search)
            return plan
        except Exception as e:
            logger.error("AdvancedPlanner analysis failed: %s. Using fallback query plan.", e)
            q = query.strip().lower()
            greeting_keywords = ["hi", "hello", "hey", "how are you", "good morning", "good afternoon"]
            intent = "greeting" if any(k in q for k in greeting_keywords) else "retrieval"
            return QueryPlan(
                intent=intent,
                suggested_search_queries=[query],
                requires_web_search=False,
                focus_topics=[]
            )
