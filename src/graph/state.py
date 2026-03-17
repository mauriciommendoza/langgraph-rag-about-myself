"""
Graph State Module.
Defines the state structure that is passed between LangGraph nodes.
"""

from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM generated answer.
        documents: List of retrieved documents.
        model_name: The selected LLM model identifier.
    """
    question: str
    generation: str
    documents: List[Document]
    model_name: str
