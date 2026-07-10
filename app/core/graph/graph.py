from langgraph.graph import END, StateGraph
from app.core.graph.state import GraphState
from app.core.graph.nodes import (
    planner_node,
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    grade_generation_node,
)


def route_after_planner(state: GraphState) -> str:
    if state.get("intent") == "greeting":
        return "direct"
    else:
        return "retrieve"


def decide_to_generate(state: GraphState) -> str:
    rewrite_count = state.get("rewrite_count", 0)
    metadata = state.get("metadata") or {}
    has_relevant_docs = len(state.get("retrieved_docs", [])) > 0
    grade_result = metadata.get("grade_result", "pass")

    if has_relevant_docs and grade_result == "pass":
        return "generate"
    
    if rewrite_count < 2:
        return "rewrite"
        
    return "generate"


def decide_to_finalize(state: GraphState) -> str:
    rewrite_count = state.get("rewrite_count", 0)
    metadata = state.get("metadata") or {}
    generation_grade = metadata.get("generation_grade", "pass")

    if generation_grade == "pass":
        return "finalize"
        
    if rewrite_count < 2:
        return "retry_rewrite"
        
    return "finalize"


def make_async_node(node_func, container):
    async def wrapper(state: GraphState):
        return await node_func(state, container)
    return wrapper


def create_rag_graph(container):
    graph = StateGraph(GraphState)
    graph.add_node("planner", make_async_node(planner_node, container))
    graph.add_node("retrieve", make_async_node(retrieve_node, container))
    graph.add_node("grade_documents", make_async_node(grade_documents_node, container))
    graph.add_node("rewrite_query", make_async_node(rewrite_query_node, container))
    graph.add_node("generate", make_async_node(generate_node, container))
    graph.add_node("grade_generation", make_async_node(grade_generation_node, container))
    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "direct": "generate",
            "retrieve": "retrieve"
        }
    )

    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite_query"
        }
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "grade_generation")

    graph.add_conditional_edges(
        "grade_generation",
        decide_to_finalize,
        {
            "finalize": END,
            "retry_rewrite": "rewrite_query"
        }
    )

    return graph.compile()