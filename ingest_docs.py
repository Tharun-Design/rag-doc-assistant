"""
ingest_docs.py — One-time script to load our docs into ChromaDB
Run this once before using the API.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.ingestion import ingest_files

docs = [
    "./docs/fastapi_basics.md",
    "./docs/pydantic_guide.md",
    "./docs/langchain_guide.md",
]

print("Starting ingestion...")
total, sources = ingest_files(docs)
print(f"Done! Ingested {total} chunks from {sources}")