"""
Response Generator Module.
Assembles the RAG chain responsible for generating the final answer.
"""

import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from src.config.llm import llm
from src.config.prompts import RAG_GENERATION_PROMPT


def strip_think_tags(text: str) -> str:
    """
    Removes <think>...</think> blocks from model output.
    Some models (e.g. Qwen 3) include internal reasoning in these tags
    that should not be shown to the user.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def build_rag_chain(llm_instance=None) -> Runnable:
    """
    Builds the Retrieval-Augmented Generation (RAG) chain.

    Args:
        llm_instance: Optional LLM instance. Defaults to the global llm.

    Returns:
        Runnable: A chain that takes a context and question, and returns an answer string.
    """
    model = llm_instance or llm
    return RAG_GENERATION_PROMPT | model | StrOutputParser() | RunnableLambda(strip_think_tags)


# Global chain instance (used by CLI/API for backward compatibility)
rag_chain = build_rag_chain()
