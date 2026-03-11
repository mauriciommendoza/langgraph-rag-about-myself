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
    "https://www.linkedin.com/in/mauriciommendoza/",
]

import os
# Path to local PDF files
PDF_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
