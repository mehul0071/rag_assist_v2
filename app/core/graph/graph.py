from langgraph.graph import END, StateGraph
from app.core.graph.state import GraphState
from app.core.graph.nodes import (
    planner_node,
    retrieve_node,
    generate_node,
)


def route_after_planner(state: GraphState) -> str:
    if state.get("intent") == "greeting":
        return "direct"
    else:
        return "retrieve"


def make_async_node(node_func, container):
    async def wrapper(state: GraphState):
        return await node_func(state, container)
    return wrapper


def create_rag_graph(container):
    graph = StateGraph(GraphState)
    graph.add_node("planner", make_async_node(planner_node, container))
    graph.add_node("retrieve", make_async_node(retrieve_node, container))
    graph.add_node("generate", make_async_node(generate_node, container))

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "direct": "generate",
            "retrieve": "retrieve"
        }
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()