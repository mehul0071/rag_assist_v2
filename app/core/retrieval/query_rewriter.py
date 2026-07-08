from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class QueryRewriter:

    def __init__(self):
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=0.0,
            api_key=settings.GROQ_API_KEY
        )
        logger.info("QueryRewriter initialized with model: %s", settings.LLM_MODEL)
        

    async def rewrite(self, query: str) -> str:
        if not settings.ENABLE_QUERY_REWRITE:
            return query

        query_lower = query.lower().strip()
        words = len(query_lower.split())

        if words > 10 and not any(k in query_lower for k in ["how", "what", "explain", "difference"]):
            return query

        try:
            prompt = ChatPromptTemplate.from_template(
                """Rewrite the user question to be more effective for retrieval 
                while keeping the exact same meaning.

                Question: {query}
                Rewritten:"""
            )
            chain = prompt | self.llm
            result = await chain.ainvoke({"query": query})
            rewritten = result.content.strip()
            logger.debug(f"Rewrote: '{query[:60]}...' → '{rewritten[:60]}...'")
            return rewritten
        except Exception as e:
            logger.warning(f"Rewrite failed: {e}")
            return query