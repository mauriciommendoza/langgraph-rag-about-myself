"""
LLM Configuration module.
Initializes and exports the language model used across the LangGraph nodes.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm() -> ChatGroq:
    """
    Instantiates and returns the Groq Language Model configuration.
    
    Returns:
        ChatGroq: The configured LangChain Groq model object.
    
    Raises:
        ValueError: If GROQ_API_KEY is not found in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please check your .env file.")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key
    )

# Global LLM instance
llm = get_llm()
