"""
Configuration settings for the LangGraph RAG application.
This module stores global application settings such as models and chunking parameters.
"""

import os

# Embedding model to use for the vector store
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# Text chunking configuration
CHUNK_SIZE: int = 1500
CHUNK_OVERLAP: int = 200

# Path to local data files (Markdown)
DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
