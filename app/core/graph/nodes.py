from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.graph.state import GraphState
import logging
from datetime import datetime
import time
from app.core.observability.metrics import retrieval_latency, llm_latency, tokens_input, tokens_output

logger = logging.getLogger(__name__)


class RouteQuery(BaseModel):
    intent: str = Field(
        description="Classify query intent. Must be 'greeting' (chitchat, hello, how are you) or 'retrieval' (questions about concepts, history, data, or facts)."
    )


class GradeDocument(BaseModel):
    binary_score: str = Field(
        description="Is the document chunk relevant to the user's question? Must be 'yes' or 'no'."
    )


class GradeHallucination(BaseModel):
    binary_score: str = Field(
        description="Is the generated answer fully grounded in and supported by the retrieved documents? 'yes' means it is grounded (no hallucinations), 'no' means it contains hallucinations."
    )


class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Does the generated answer address and resolve the user's question? Must be 'yes' or 'no'."
    )


async def planner_node(state: GraphState, service) -> GraphState:
    try:
        rewrite_count = state.get("rewrite_count", 0)
        generation_count = state.get("generation_count", 0)

        # Call the AdvancedPlanner from the container (passed as service)
        plan = await service.planner.plan_query(
            query=state["question"], 
            chat_history=state.get("chat_history", "")
        )
        intent = plan.intent.strip().lower()
        logger.info(f"LLM Planner → Classified intent: {intent} for question: {state['question'][:60]}...")

        return {
            **state,
            "intent": intent,
            "rewrite_count": rewrite_count,
            "generation_count": generation_count,
            "metadata": {
                **(state.get("metadata") or {}),
                "intent": intent,
                "suggested_queries": plan.suggested_search_queries,
                "requires_web_search": plan.requires_web_search,
                "focus_topics": plan.focus_topics,
                "processed_at": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Planner node failed: {e}. Falling back to keyword classification.")
        q = state["question"].strip().lower()
        greeting_keywords = ["hi", "hello", "hey", "how are you", "good morning"]
        intent = "greeting" if any(g in q for g in greeting_keywords) else "retrieval"
        return {
            **state,
            "intent": intent,
            "rewrite_count": state.get("rewrite_count", 0),
            "generation_count": state.get("generation_count", 0),
            "metadata": {
                **(state.get("metadata") or {}),
                "intent": intent,
                "fallback_used": True
            }
        }


async def retrieve_node(state: GraphState, service) -> GraphState:
    try:
        start_time = time.perf_counter()
        docs = await service.retrieval_pipeline.search(state["question"])
        duration = time.perf_counter() - start_time
        retrieval_latency.observe(duration)
        logger.info(f"Retrieve node retrieved {len(docs)} docs in {duration:.4f}s")

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


async def grade_documents_node(state: GraphState, service) -> GraphState:
    try:
        docs = state.get("retrieved_docs", [])
        question = state["question"]
        
        if not docs:
            return {
                **state,
                "metadata": {
                    **(state.get("metadata") or {}),
                    "grade_result": "fail",
                    "graded_relevant_count": 0
                }
            }

        logger.info("Grading %d document chunks...", len(docs))
        structured_grader = service.llm_service.llm.with_structured_output(GradeDocument)
        
        relevant_docs = []
        for idx, doc in enumerate(docs):
            system_prompt = (
                "You are an assessment grader grading document relevance. "
                "Assess whether the document contains information relevant to answer the query. "
                "Respond with a binary grade: 'yes' (relevant) or 'no' (irrelevant)."
            )
            user_prompt = f"Query: {question}\n\nDocument Chunk:\n{doc.page_content}"
            
            grade = await structured_grader.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            is_relevant = grade.binary_score.strip().lower() == "yes"
            logger.debug(f"Document [{idx}] graded relevant: {is_relevant}")
            if is_relevant:
                relevant_docs.append(doc)

        grade_result = "pass" if len(relevant_docs) > 0 else "fail"
        logger.info("Grading complete. Graded %d relevant out of %d. Result: %s", len(relevant_docs), len(docs), grade_result)

        return {
            **state,
            "retrieved_docs": relevant_docs,
            "metadata": {
                **(state.get("metadata") or {}),
                "grade_result": grade_result,
                "graded_relevant_count": len(relevant_docs)
            }
        }
    except Exception as e:
        logger.error(f"Document grading node failed: {e}")
        return {
            **state,
            "metadata": {
                **(state.get("metadata") or {}),
                "grade_result": "pass",
                "grading_error": str(e)
            }
        }


async def rewrite_query_node(state: GraphState, service) -> GraphState:
    try:
        rewrite_count = state.get("rewrite_count", 0) + 1
        original_query = state["question"]
        
        logger.info(f"Rewrite cycle #{rewrite_count} triggered for query: '{original_query[:60]}...'")
        
        rewritten_query = await service.retrieval_pipeline.rewriter.rewrite(original_query)
        logger.info(f"Query rewritten: '{original_query[:60]}' → '{rewritten_query[:60]}'")

        return {
            **state,
            "question": rewritten_query,
            "rewrite_count": rewrite_count,
            "metadata": {
                **(state.get("metadata") or {}),
                "original_question": original_query,
                "rewritten_at_step": rewrite_count
            }
        }
    except Exception as e:
        logger.error(f"Query rewrite node failed: {e}")
        return {
            **state,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "error": str(e)
        }


async def generate_node(state: GraphState, service) -> GraphState:
    try:
        generation_count = state.get("generation_count", 0) + 1
        
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
            logger.info(f"Context built. Estimated tokens: {context_data.get('estimated_tokens')}")

        start_time = time.perf_counter()
        answer = await service.llm_service.generate_text([HumanMessage(content=prompt)])
        duration = time.perf_counter() - start_time
        llm_latency.observe(duration)
        logger.info(f"LLM generated response in {duration:.4f}s")

        input_tok = service.context_builder.token_manager.count_tokens(prompt)
        output_tok = service.context_builder.token_manager.count_tokens(answer)
        tokens_input.inc(input_tok)
        tokens_output.inc(output_tok)

        return {
            **state,
            "answer": answer,
            "sources": sources,
            "generation_count": generation_count,
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
            "answer": "Sorry, I encountered an error during generation.",
            "error": str(e)
        }


async def grade_generation_node(state: GraphState, service) -> GraphState:
    try:
        answer = state.get("answer")
        docs = state.get("retrieved_docs", [])
        question = state["question"]
        
        if state.get("intent") == "greeting" or not docs or not answer:
            return {
                **state,
                "metadata": {
                    **(state.get("metadata") or {}),
                    "generation_grade": "pass"
                }
            }

        logger.info("Grading generation groundedness & usefulness...")
        
        structured_hallucination_grader = service.llm_service.llm.with_structured_output(GradeHallucination)
        system_hallucination_prompt = (
            "You are an assessment grader grading LLM output groundedness. "
            "Determine if the generated answer is fully grounded in and supported by the facts "
            "provided in the retrieved documents. Respond with a binary grade: 'yes' (grounded/no hallucinations) "
            "or 'no' (hallucinated content present)."
        )
        context_text = "\n\n".join([doc.page_content for doc in docs])
        user_hallucination_prompt = f"Answer:\n{answer}\n\nRetrieved Context:\n{context_text}"
        
        hallucination_grade = await structured_hallucination_grader.ainvoke([
            SystemMessage(content=system_hallucination_prompt),
            HumanMessage(content=user_hallucination_prompt)
        ])
        
        is_grounded = hallucination_grade.binary_score.strip().lower() == "yes"
        logger.info(f"Generation groundedness check: {is_grounded}")

        structured_utility_grader = service.llm_service.llm.with_structured_output(GradeAnswer)
        system_utility_prompt = (
            "You are an assessment grader grading LLM utility. "
            "Determine if the generated answer fully and accurately addresses the user's question. "
            "Respond with a binary grade: 'yes' (answers question) or 'no' (does not answer question)."
        )
        user_utility_prompt = f"Question: {question}\n\nAnswer:\n{answer}"
        
        utility_grade = await structured_utility_grader.ainvoke([
            SystemMessage(content=system_utility_prompt),
            HumanMessage(content=user_utility_prompt)
        ])
        
        answers_question = utility_grade.binary_score.strip().lower() == "yes"
        logger.info(f"Generation utility check: {answers_question}")

        generation_grade = "pass" if (is_grounded and answers_question) else "fail"
        logger.info(f"Overall generation grade: {generation_grade}")

        return {
            **state,
            "metadata": {
                **(state.get("metadata") or {}),
                "generation_grade": generation_grade,
                "hallucination_check": "passed" if is_grounded else "failed",
                "utility_check": "passed" if answers_question else "failed"
            }
        }
    except Exception as e:
        logger.error(f"Grade generation node failed: {e}")
        return {
            **state,
            "metadata": {
                **(state.get("metadata") or {}),
                "generation_grade": "pass",
                "grade_generation_error": str(e)
            }
        }