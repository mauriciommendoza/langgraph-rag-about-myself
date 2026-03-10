"""
Configuration settings for the LangGraph RAG application.
This module stores global application settings such as models, chunking parameters, and URLs.
"""

from typing import List

# Embedding model to use for the vector store
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# Text chunking configuration
CHUNK_SIZE: int = 250
CHUNK_OVERLAP: int = 0

# Source URLs to load documents from
URLS_TO_LOAD: List[str] = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
