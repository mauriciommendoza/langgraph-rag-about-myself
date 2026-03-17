"""
Graph Edges Module.
Contains the conditional routing and edge logic for deciding the next node in the graph.
"""

from typing import Literal

from src.graph.state import GraphState
from src.config.llm import get_llm
from src.chains.graders import build_hallucination_grader, build_answer_grader

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
    model_name = state.get("model_name", "")

    # Build graders dynamically with the selected model
    llm_instance = get_llm(model_name) if model_name else get_llm()
    h_grader = build_hallucination_grader(llm_instance)
    a_grader = build_answer_grader(llm_instance)

    score = h_grader.invoke({"documents": documents, "generation": generation})
    hallucination_grade = score.get('score', 'no')

    if hallucination_grade == "yes":
        print("  [Hallucination Check] Answer is grounded. Checking usefulness...")
        score = a_grader.invoke({"question": question, "generation": generation})
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
