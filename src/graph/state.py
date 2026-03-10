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
        web_search: Flag indicating whether to use web search ('Yes' or 'No').
        documents: List of retrieved documents.
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]
