"""
LLM Configuration module.
Initializes and exports the language model used across the LangGraph nodes.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Available models for the user to choose from
AVAILABLE_MODELS = {
    "Qwen 3 32B": "qwen/qwen3-32b",
    "GPT OSS 120B": "openai/gpt-oss-120b",
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
}

DEFAULT_MODEL = "qwen/qwen3-32b"


def get_llm(model_name: str = DEFAULT_MODEL) -> ChatGroq:
    """
    Instantiates and returns the Groq Language Model configuration.

    Args:
        model_name (str): The model identifier to use (e.g. 'qwen/qwen3-32b').

    Returns:
        ChatGroq: The configured LangChain Groq model object.

    Raises:
        ValueError: If GROQ_API_KEY is not found in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please check your .env file.")

    return ChatGroq(
        model=model_name,
        temperature=0,
        max_tokens=1024,
        timeout=30,
        max_retries=2,
        api_key=api_key
    )

# Global LLM instance (used by CLI/API for backward compatibility)
llm = get_llm()
