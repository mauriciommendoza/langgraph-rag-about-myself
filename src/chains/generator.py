"""
Response Generator Module.
Assembles the RAG chain responsible for generating the final answer.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.config.llm import llm
from src.config.prompts import RAG_GENERATION_PROMPT


def build_rag_chain() -> Runnable:
    """
    Builds the Retrieval-Augmented Generation (RAG) chain.

    Returns:
        Runnable: A chain that takes a context and question, and returns an answer string.
    """
    return RAG_GENERATION_PROMPT | llm | StrOutputParser()


rag_chain = build_rag_chain()
