"""
Node 2: Adaptive Retrieval
- Searches ChromaDB for relevant chunks
- Uses different top-k based on query type
"""
import logging
from langchain_core.documents import Document
from app.models.state import GraphState
from app.utils.vector_store import get_vector_store
from app.config import (
    TOP_K_DEFAULT, TOP_K_HOWTO,
    TOP_K_CONCEPTUAL, TOP_K_TROUBLESHOOTING,
)

logger = logging.getLogger(__name__)

ADAPTIVE_TOP_K = {
    "how-to": TOP_K_HOWTO,
    "conceptual": TOP_K_CONCEPTUAL,
    "troubleshooting": TOP_K_TROUBLESHOOTING,
    "api-reference": TOP_K_DEFAULT,
    "general": TOP_K_DEFAULT,
}


def retrieval_node(state: GraphState) -> GraphState:
    query = state.get("rewritten_query") or state["question"]
    query_type = state.get("query_type", "general")
    top_k = ADAPTIVE_TOP_K.get(query_type, TOP_K_DEFAULT)

    logger.info(f"[Node 2] Retrieval | type='{query_type}' | top_k={top_k}")

    vs = get_vector_store()
    retrieved_docs = vs.similarity_search(query, top_k=top_k)

    for i, doc in enumerate(retrieved_docs):
        logger.info(f"  Retrieved chunk {i+1}: {doc.page_content[:100]}")

    return {
        **state,
        "retrieved_docs": retrieved_docs,
    }