from langchain_core.messages import HumanMessage
from app.core.graph.state import GraphState
import logging
from datetime import datetime
import time
from app.core.observability.metrics import retrieval_latency, llm_latency, tokens_input, tokens_output

logger = logging.getLogger(__name__)


async def planner_node(state: GraphState, service) -> GraphState:
    try:
        q = state["question"].strip().lower()
        greeting_keywords = ["hi", "hello", "hey", "how are you", "good morning", "good afternoon"]
        
        if any(g in q for g in greeting_keywords):
            intent = "greeting"
        elif any(word in q for word in ["compare", "difference", "vs", "versus", "which is best"]):
            intent = "comparison"
        else:
            intent = "knowledge"

        logger.info(f"Planner → Intent: {intent} for question: {state['question'][:60]}...")

        return {
            **state,
            "intent": intent,
            "metadata": {
                **(state.get("metadata") or {}),
                "intent": intent,
                "processed_at": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Planner failed: {e}")
        return {**state, "intent": "knowledge", "error": str(e)}


async def retrieve_node(state: GraphState, service) -> GraphState:
    """Uses RetrievalPipeline (Rewrite → Retrieve → Rerank)"""
    try:
        logger.info(f"[DEBUG] Using RetrievalPipeline: {type(service.retrieval_pipeline)}")
        start_time = time.perf_counter()
        docs = await service.retrieval_pipeline.search(state["question"])
        duration = time.perf_counter() - start_time
        retrieval_latency.observe(duration)
        logger.info(f"[DEBUG] Retrieved {len(docs)} documents in {duration:.4f}s")
        
        return {
            **state,
            "retrieved_docs": docs or [],
            "metadata": {
                **(state.get("metadata") or {}),
                "retrieved_count": len(docs) if docs else 0,
                "used_rewrite": service.retrieval_pipeline.enable_rewrite
            }
        }
    except Exception as e:
        logger.error(f"Retrieve node failed: {e}")
        return {**state, "retrieved_docs": [], "error": str(e)}


async def generate_node(state: GraphState, service) -> GraphState:
    try:
        if state.get("intent") == "greeting":
            prompt = f"Answer conversationally and friendly:\n\n{state['question']}"
            sources = []
        else:
            context_data = service.context_builder.build_context(
                query=state["question"],
                retrieved_docs=state.get("retrieved_docs", []),
                chat_history=state.get("chat_history", "")
            )
            
            prompt_template = service.prompt_manager.get_rag_prompt()
            prompt = prompt_template.format(
                chat_history=state.get("chat_history", ""),
                context=context_data.get("formatted_context", ""),
                question=state["question"]
            )
            sources = context_data.get("sources", [])
            logger.info(f"[DEBUG] ContextBuilder used. Tokens estimated: {context_data.get('estimated_tokens')}")

        start_time = time.perf_counter()
        answer = await service.llm_service.generate_text([HumanMessage(content=prompt)])
        duration = time.perf_counter() - start_time
        llm_latency.observe(duration)
        logger.info(f"[DEBUG] LLM generated text in {duration:.4f}s")

        input_tok = service.context_builder.token_manager.count_tokens(prompt)
        output_tok = service.context_builder.token_manager.count_tokens(answer)
        tokens_input.inc(input_tok)
        tokens_output.inc(output_tok)

        return {
            **state,
            "answer": answer,
            "sources": sources,
            "metadata": {
                **(state.get("metadata") or {}),
                "generated_at": datetime.now().isoformat(),
                "used_tokens": context_data.get("estimated_tokens") if 'context_data' in locals() else None
            }
        }

    except Exception as e:
        logger.error(f"Generate node failed: {e}")
        return {
            **state,
            "answer": "Sorry, I encountered an error. Please try again.",
            "error": str(e)
        }