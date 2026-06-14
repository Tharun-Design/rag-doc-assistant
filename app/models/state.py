"""
state.py — LangGraph State Schema
This is the memory of the workflow — all data that flows between nodes.
"""
from typing import TypedDict, List, Optional, Literal
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    rewritten_query: str
    query_type: Literal["how-to", "conceptual", "troubleshooting", "api-reference", "general"]
    retrieved_docs: List[Document]
    graded_docs: List[Document]
    generation: str
    retry_count: int
    hallucination_flag: bool
    sources: List[str]
    error: Optional[str]
    # Conversation memory
    chat_history: List[dict]   # [{"role": "user/assistant", "content": "..."}]
    # Web search fallback
    web_search_used: bool
    regen_count: int