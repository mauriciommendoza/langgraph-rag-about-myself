"""
Graph Builder Module.
Constructs and compiles the StateGraph workflow using the defined nodes and edges.
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.state import GraphState
from src.graph.nodes import web_search, retrieve, grade_documents, generate
from src.graph.edges import route_question, decide_to_generate, grade_generation_v_documents_and_question

def build_workflow() -> CompiledStateGraph:
    """
    Builds and compiles the LangGraph workflow.
    
    Returns:
        CompiledStateGraph: The compiled state graph application ready to be invoked.
    """
    workflow = StateGraph(GraphState)

    # --- Add Nodes
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    # --- Conditional Entry Point
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "retrieve": "retrieve",
        },
    )

    # --- Construct Edges
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )

    workflow.add_edge("websearch", "generate")

    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",  # Retry generation
            "useful": END,                # End of workflow
            "not useful": "websearch",    # Fallback to search
        },
    )

    return workflow.compile()

# The compiled LangGraph application
app = build_workflow()
