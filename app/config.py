"""
config.py — Central configuration loaded from .env
All settings in one place — nothing is hardcoded anywhere else.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3-8b-8192")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Embeddings
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ChromaDB
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

# Chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
TOP_K_DEFAULT: int = int(os.getenv("TOP_K_DEFAULT", "5"))
TOP_K_HOWTO: int = int(os.getenv("TOP_K_HOWTO", "7"))
TOP_K_CONCEPTUAL: int = int(os.getenv("TOP_K_CONCEPTUAL", "4"))
TOP_K_TROUBLESHOOTING: int = int(os.getenv("TOP_K_TROUBLESHOOTING", "6"))

# Workflow
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))