"""
Document Indexing and Retrieval Module.
Handles loading, chunking, embedding, and vectorizing Markdown documents from the data directory.
"""

import os
import glob
from typing import List
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DATA_DIR
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


class DocumentIndexer:
    """
    A class responsible for loading Markdown documents from a local directory,
    splitting them into chunks, and storing them in a Pinecone vector store.
    """

    def __init__(self, data_dir: str, chunk_size: int, chunk_overlap: int, embedding_model: str):
        """
        Initializes the indexer with data directory and embedding settings.

        Args:
            data_dir (str): Path to directory containing Markdown files.
            chunk_size (int): Max characters for each document chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.
            embedding_model (str): HuggingFace embedding model ID.
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

    def _load_documents(self) -> List[Document]:
        """Loads documents from local Markdown files in the data directory."""
        all_docs = []

        if os.path.exists(self.data_dir):
            print(f"  [Indexer] Scanning '{self.data_dir}' for Markdown files...")
            md_files = glob.glob(os.path.join(self.data_dir, "**/*.md"), recursive=True)
            if md_files:
                for filepath in md_files:
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                            doc = Document(page_content=text_content, metadata={"source": filepath})
                            all_docs.append(doc)
                    except Exception as e:
                        print(f"  [Indexer] Error reading {filepath}: {e}")
                print(f"  [Indexer] Loaded {len(md_files)} Markdown file(s).")
            else:
                print("  [Indexer] No Markdown files found in the directory.")
        else:
            print(f"  [Indexer] Data directory '{self.data_dir}' does not exist.")

        return all_docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """Splits the loaded documents into smaller chunks using a text splitter."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def create_retriever(self):
        """
        Creates a retriever connected to a Pinecone vector store.

        - If the index does NOT exist, it creates the index, loads documents,
          splits them, embeds them, and uploads them to Pinecone.
        - If the index already exists AND has vectors, it simply connects to it
          without re-uploading (avoiding duplicates).
        - If the index exists but is empty, it populates it with document vectors.

        Returns:
            VectorStoreRetriever: A retriever instance connected to the Pinecone vector database.

        Raises:
            ValueError: If PINECONE_API_KEY or PINECONE_INDEX_NAME are not set.
        """
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set. Please check your .env file.")

        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME environment variable is not set. Please check your .env file.")

        pc = Pinecone(api_key=api_key)
        embedding = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # Check if index exists, create if not
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        needs_population = False

        if index_name not in existing_indexes:
            print(f"  [Pinecone] Index '{index_name}' not found. Creating...")
            pc.create_index(
                name=index_name,
                dimension=384,  # Dimensions for sentence-transformers/all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"  [Pinecone] Index '{index_name}' created successfully.")
            needs_population = True
        else:
            # Index exists — check if it already has vectors
            index_stats = pc.Index(index_name).describe_index_stats()
            total_vectors = index_stats.get("total_vector_count", 0)

            if total_vectors > 0:
                print(f"  [Pinecone] Index '{index_name}' has {total_vectors} vectors. Skipping upload.")
            else:
                print(f"  [Pinecone] Index '{index_name}' is empty. Uploading documents...")
                needs_population = True

        if needs_population:
            # Load, split, and upload documents
            docs_list = self._load_documents()
            doc_splits = self._split_documents(docs_list)

            vectorstore = PineconeVectorStore.from_documents(
                documents=doc_splits,
                embedding=embedding,
                index_name=index_name,
            )
        else:
            # Just connect to the existing populated index
            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embedding,
            )

        return vectorstore.as_retriever()


def format_docs(docs: List[Document]) -> str:
    """
    Utility function to format a list of LangChain documents into a single text block.

    Args:
        docs (List[Document]): List of retrieved documents.

    Returns:
        str: Combined content of the documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# --- Lazy-loaded global retriever ---
_retriever_instance = None


def get_retriever():
    """
    Returns the global retriever instance, creating it on first call.
    This avoids blocking network calls at module import time.
    """
    global _retriever_instance
    if _retriever_instance is None:
        indexer = DocumentIndexer(DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL)
        _retriever_instance = indexer.create_retriever()
    return _retriever_instance
