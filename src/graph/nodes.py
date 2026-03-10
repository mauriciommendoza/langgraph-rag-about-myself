"""
Graph Nodes Module.
Contains the executable functions for each Node in the LangGraph workflow.
"""

from typing import Dict, Any
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from src.graph.state import GraphState
from src.retrieval.document_indexer import retriever, format_docs
from src.chains.generator import rag_chain
from src.chains.graders import retrieval_grader

import os
# Ensure you use a valid Tavily API Key from .env.
web_search_tool = TavilySearch(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))

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
    documents = retriever.invoke(question)
    print(f"  [Retrieve] Found {len(documents)} document(s).")
    return {"documents": documents, "question": question, "generation": "", "web_search": "No"}


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
        "web_search": state.get("web_search", "No")
    }


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Filters retrieved documents based on their relevance to the question.
    Flags if web search is needed.
    
    Args:
        state (GraphState): The current graph state.
        
    Returns:
        Dict: Updated state with only relevant documents and the web_search flag updated.
    """
    print("  [Grader] Checking document relevance...")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search = "No"

    for i, d in enumerate(documents, 1):
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.get('score', 'no')
        
        if grade.lower() == "yes":
            print(f"    Doc {i}: Relevant")
            filtered_docs.append(d)
        else:
            print(f"    Doc {i}: Not relevant")
            web_search = "Yes"

    print(f"  [Grader] {len(filtered_docs)}/{len(documents)} documents passed.")
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
        "generation": state.get("generation", "")
    }


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Executes a web search for the question to fetch additional context.
    
    Args:
        state (GraphState): The current graph state.
        
    Returns:
        Dict: Updated state containing new web documents appended to existing ones.
    """
    print("  [Web Search] Searching the web for additional context...")
    question = state["question"]
    documents = state.get("documents", [])  
    
    docs = web_search_tool.invoke({"query": question})
    
    # TavilySearch returns a dict: {"query": ..., "results": [{"content": ..., "url": ...}, ...]}
    if isinstance(docs, dict):
        results_list = docs.get("results", [])
        web_results = "\n".join([r.get("content", "") for r in results_list])
        result_count = len(results_list)
    elif isinstance(docs, list):
        web_results = "\n".join([d.get("content", str(d)) if isinstance(d, dict) else str(d) for d in docs])
        result_count = len(docs)
    else:
        web_results = str(docs)
        result_count = 1
    
    web_doc = Document(page_content=web_results)
    
    documents.append(web_doc)
    print(f"  [Web Search] Added {result_count} web result(s).")
    return {
        "documents": documents,
        "question": question,
        "web_search": "Yes",
        "generation": state.get("generation", "")
    }
