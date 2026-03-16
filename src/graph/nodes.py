"""
Graph Nodes Module.
Contains the executable functions for each Node in the LangGraph workflow.
"""

from typing import Dict, Any
from langchain_core.documents import Document

from src.graph.state import GraphState
from src.retrieval.document_indexer import get_retriever, format_docs
from src.chains.generator import rag_chain
from src.chains.graders import retrieval_grader

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves documents based on the question.

    Args:
        state (GraphState): The current graph state.

    Returns:
        Dict: Updated state dictionary with the retrieved documents.
    """
    print("  [Retrieve] Searching vectorstore for relevant documents...")
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    print(f"  [Retrieve] Found {len(documents)} document(s).")
    return {"documents": documents, "question": question, "generation": ""}


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generates an answer using the retrieved documents as context.

    Args:
        state (GraphState): The current graph state.

    Returns:
        Dict: Updated state dictionary containing the generated answer.
    """
    print("  [Generate] Generating answer from context...")
    question = state["question"]
    documents = state["documents"]

    context = format_docs(documents)
    generation = rag_chain.invoke({"context": context, "question": question})

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
    }


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Filters retrieved documents based on their relevance to the question.

    Args:
        state (GraphState): The current graph state.

    Returns:
        Dict: Updated state with only relevant documents.
    """
    print("  [Grader] Checking document relevance...")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []

    for i, d in enumerate(documents, 1):
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.get('score', 'no')

        if grade.lower() == "yes":
            print(f"    Doc {i}: Relevant")
            filtered_docs.append(d)
        else:
            print(f"    Doc {i}: Not relevant")

    print(f"  [Grader] {len(filtered_docs)}/{len(documents)} documents passed.")
    return {
        "documents": filtered_docs,
        "question": question,
        "generation": state.get("generation", "")
    }
