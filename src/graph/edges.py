"""
Graph Edges Module.
Contains the conditional routing and edge logic for deciding the next node in the graph.
"""

from typing import Literal

from src.graph.state import GraphState
from src.chains.router import question_router
from src.chains.graders import hallucination_grader, answer_grader

def route_question(state: GraphState) -> Literal["websearch", "retrieve"]:
    """
    Routes the initial question to either the RAG pipeline or web search.
    
    Args:
        state (GraphState): The current graph state.
        
    Returns:
        str: Next node to call ("websearch" or "retrieve").
    """
    print("  [Router] Deciding data source...")
    question = state["question"]
    source = question_router.invoke({"question": question})

    if source.get('datasource') == 'web_search':
        print("  [Router] -> Web Search")
        return "websearch"
    else:
        print("  [Router] -> Vectorstore (RAG)")
        return "retrieve"


def decide_to_generate(state: GraphState) -> Literal["websearch", "generate"]:
    """
    Decides whether to perform generation or to fall back to web search based on document grading.
    
    Args:
        state (GraphState): The current graph state.
        
    Returns:
        str: Next node to call ("websearch" or "generate").
    """
    if state["web_search"] == "Yes":
        print("  [Decision] Not enough relevant docs -> Web Search")
        return "websearch"
    else:
        print("  [Decision] Documents are sufficient -> Generate")
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState) -> Literal["not supported", "useful", "not useful"]:
    """
    Checks if the generated answer is grounded in facts and actually answers the user's question.
    
    Args:
        state (GraphState): The current graph state.
        
    Returns:
        str: Signal indicating the next step ("not supported", "useful", "not useful").
    """
    print("  [Hallucination Check] Verifying answer is grounded in facts...")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    hallucination_grade = score.get('score', 'no')

    if hallucination_grade == "yes":
        print("  [Hallucination Check] Answer is grounded. Checking usefulness...")
        score = answer_grader.invoke({"question": question, "generation": generation})
        answer_grade = score.get('score', 'no')
        
        if answer_grade == "yes":
            print("  [Answer Check] Answer is useful!")
            return "useful"
        else:
            print("  [Answer Check] Answer does not address the question. Retrying...")
            return "not useful"
    else:
        print("  [Hallucination Check] Answer is NOT grounded. Retrying...")
        return "not supported"
