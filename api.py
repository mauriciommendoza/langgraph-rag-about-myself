"""
FastAPI Backend.
Exposes the LangGraph RAG Agent as a REST API for external integrations.
Run with: uvicorn api:api --reload
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from src.graph.builder import app as langgraph_app

# --- API Setup ---
api = FastAPI(
    title="LangGraph RAG Agent API",
    description="REST API for the LangGraph RAG chatbot with Pinecone retrieval and Tavily web search.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    """Request body schema for the /chat endpoint."""
    question: str


class ChatResponse(BaseModel):
    """Response body schema for the /chat endpoint."""
    question: str
    answer: str
    steps: list[str]


@api.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "ok"}


@api.post("/chat", response_model=ChatResponse)
async def chat(request: QuestionRequest):
    """
    Main chat endpoint.
    Receives a question, runs the full LangGraph pipeline,
    and returns the generated answer along with the steps taken.
    
    Args:
        request (QuestionRequest): JSON body with a 'question' field.
        
    Returns:
        ChatResponse: JSON with the question, answer, and list of nodes visited.
    """
    try:
        steps = []
        generation = ""

        for output in langgraph_app.stream({"question": request.question}):
            for node_name, node_output in output.items():
                steps.append(node_name)
                if node_name == "generate":
                    generation = node_output.get("generation", "")

        if not generation:
            generation = "Could not generate an answer. Try rephrasing the question."

        return ChatResponse(
            question=request.question,
            answer=generation,
            steps=steps,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
