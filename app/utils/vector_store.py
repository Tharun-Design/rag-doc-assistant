"""
vector_store.py — ChromaDB wrapper for storing and retrieving document chunks.
Uses sentence-transformers locally — no API key needed, runs on your PC.
"""
import logging
import os
import hashlib
from typing import List
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from app.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedding_fn = None
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB ready — {self.collection.count()} chunks stored.")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self._embedding_fn is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._embedding_fn = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = self._embedding_fn.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [hashlib.md5(text.encode()).hexdigest()[:16] for text in texts]
        embeddings = self._get_embeddings(texts)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(texts)} chunks to ChromaDB.")
        return len(texts)

    def similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        if self.collection.count() == 0:
            logger.warning("Vector store is empty. Ingest documents first.")
            return []
        query_embedding = self._get_embeddings([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            doc = Document(
                page_content=text,
                metadata={**metadata, "relevance_score": round(1 - distance, 4)},
            )
            documents.append(doc)
        return documents

    def list_documents(self) -> List[dict]:
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["documents", "metadatas"])
        docs = []
        for i, (doc_id, text, meta) in enumerate(
            zip(results["ids"], results["documents"], results["metadatas"])
        ):
            docs.append({
                "id": doc_id,
                "source": meta.get("source", "unknown"),
                "content_preview": text[:150] + "..." if len(text) > 150 else text,
                "chunk_index": meta.get("chunk_index", i),
            })
        return docs

    def get_total_count(self) -> int:
        return self.collection.count()


_vector_store = None

def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store