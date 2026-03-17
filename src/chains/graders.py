"""
Grader Chains Module.
Assembles chains used to grade document relevance, hallucinations, and answer usefulness.
"""

from langchain_core.runnables import Runnable

from src.config.llm import llm
from src.config.prompts import (
    RETRIEVAL_GRADER_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
    ANSWER_GRADER_PROMPT,
)


def build_retrieval_grader(llm_instance=None) -> Runnable:
    """
    Builds a chain that assesses the relevance of a retrieved document to a user's question.

    Args:
        llm_instance: Optional LLM instance. Defaults to the global llm.

    Returns:
        Runnable: A chain outputting a JSON with a {"score": "yes" | "no"} key.
    """
    model = llm_instance or llm
    return RETRIEVAL_GRADER_PROMPT | model.with_structured_output(method="json_mode")


def build_hallucination_grader(llm_instance=None) -> Runnable:
    """
    Builds a chain that checks if a generated answer is hallucinated or grounded in the facts.

    Args:
        llm_instance: Optional LLM instance. Defaults to the global llm.

    Returns:
        Runnable: A chain outputting a JSON with a {"score": "yes" | "no"} key.
    """
    model = llm_instance or llm
    return HALLUCINATION_GRADER_PROMPT | model.with_structured_output(method="json_mode")


def build_answer_grader(llm_instance=None) -> Runnable:
    """
    Builds a chain that determines if the generated answer actually resolves the user's question.

    Args:
        llm_instance: Optional LLM instance. Defaults to the global llm.

    Returns:
        Runnable: A chain outputting a JSON with a {"score": "yes" | "no"} key.
    """
    model = llm_instance or llm
    return ANSWER_GRADER_PROMPT | model.with_structured_output(method="json_mode")


# Global chain instances (used by CLI/API for backward compatibility)
retrieval_grader = build_retrieval_grader()
hallucination_grader = build_hallucination_grader()
answer_grader = build_answer_grader()
