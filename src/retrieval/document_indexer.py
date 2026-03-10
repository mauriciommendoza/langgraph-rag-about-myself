"""
Document Indexing and Retrieval Module.
Handles downloading, chunking, embedding, and vectorizing documents from URLs.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import URLS_TO_LOAD, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Set User-Agent to avoid scraping issues
os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; LangGraph-RAG-Agent/1.0; +http://example.com)"

class DocumentIndexer:
    """
    A class responsible for fetching documents from URLs, splitting them into chunks, 
    and storing them in a Pinecone vector store.
    """
    
    def __init__(self, urls: List[str], chunk_size: int, chunk_overlap: int, embedding_model: str):
        """
        Initializes the indexer with URLs and embedding settings.
        
        Args:
            urls (List[str]): URLs to scrape.
            chunk_size (int): Max characters for each document chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.
            embedding_model (str): HuggingFace embedding model ID.
        """
        self.urls = urls
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
    def _load_documents(self) -> List[Document]:
        """Loads documents from the provided URLs."""
        docs = [WebBaseLoader(url).load() for url in self.urls]
        return [item for sublist in docs for item in sublist]

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
        
        - If the index does NOT exist, it creates the index, downloads documents,
          splits them, embeds them, and uploads them to Pinecone.
        - If the index already exists AND has vectors, it simply connects to it
          without re-uploading (avoiding duplicates).
        - If the index exists but is empty, it populates it with document vectors.
        
        Returns:
            VectorStoreRetriever: A retriever instance connected to the Pinecone vector database.
        """
        index_name = os.getenv("PINECONE_INDEX_NAME", "psyco-oncology-rag")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
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
            # Download, split, and upload documents
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
    return "\\n\\n".join(doc.page_content for doc in docs)

# Instantiate the global retriever
indexer = DocumentIndexer(URLS_TO_LOAD, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL)
retriever = indexer.create_retriever()
