"""
workflow.py — LangGraph StateGraph with:
- 5 core nodes (query analysis, retrieval, grading, generation, hallucination check)
- Web search fallback node (Tavily) when local docs fail
- Conversation memory passed through state

Flow:
  START
    → Node 1: Query Analysis (classify + rewrite, uses chat history)
    → Node 2: Retrieval (search ChromaDB)
    → Node 3: Document Grading (filter relevant chunks)
    → [conditional] relevant?     → Node 4: Generation
                   no relevant?   → rewrite + retry (max 3 times)
                   max retries?   → Web Search Node (Tavily)
                                  → Node 4: Generation
    → Node 5: Hallucination Check
    → [conditional] grounded?     → END
                   hallucinated?  → regenerate → END
"""
import logging
from langgraph.graph import StateGraph, END

from app.models.state import GraphState
from app.nodes.node_query_analysis import query_analysis_node
from app.nodes.node_retrieval import retrieval_node
from app.nodes.node_document_grading import (
    document_grading_node,
    route_after_grading,
)
from app.nodes.node_generation import generation_node
from app.nodes.node_hallucination_check import (
    hallucination_check_node,
    route_after_hallucination_check,
)
from app.nodes.node_web_search import web_search_node
from app.config import MAX_RETRIES

logger = logging.getLogger(__name__)


# ── Rewrite + Retry Node ──────────────────────────────────────────────────────

def rewrite_and_retry_node(state: GraphState) -> GraphState:
    retry_count = state.get("retry_count", 0) + 1
    original = state["question"]

    rewritten = f"detailed explanation of {original} with examples and use cases"

    logger.info(
        f"[Retry Node] Attempt {retry_count}/{MAX_RETRIES} | "
        f"New query: '{rewritten}'"
    )

    return {
        **state,
        "retry_count": retry_count,
        "rewritten_query": rewritten,
        "retrieved_docs": [],
        "graded_docs": [],
    }


# ── No Answer Node ────────────────────────────────────────────────────────────

def no_answer_node(state: GraphState) -> GraphState:
    logger.warning("[No Answer Node] Max retries exceeded — falling back to web search.")
    return {
        **state,
        "generation": (
            "I was unable to find relevant information in the "
            "documentation to answer your question. Please try "
            "rephrasing or check if the topic is covered in the docs."
        ),
        "hallucination_flag": False,
        "error": "No relevant documents found after maximum retries.",
    }


# ── Routing: after grading ────────────────────────────────────────────────────

def route_after_grading_with_web(state: GraphState) -> str:
    """
    Extended routing that sends to web_search instead of no_answer
    when max retries are hit.
    """
    graded_docs = state.get("graded_docs", [])
    retry_count = state.get("retry_count", 0)

    if graded_docs:
        return "generate"
    elif retry_count >= MAX_RETRIES:
        return "web_search"   # fallback to Tavily
    else:
        return "rewrite_retry"


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    graph = StateGraph(GraphState)

    # Add all nodes
    graph.add_node("query_analysis",      query_analysis_node)
    graph.add_node("retrieval",           retrieval_node)
    graph.add_node("document_grading",    document_grading_node)
    graph.add_node("generation",          generation_node)
    graph.add_node("hallucination_check", hallucination_check_node)
    graph.add_node("rewrite_retry",       rewrite_and_retry_node)
    graph.add_node("no_answer",           no_answer_node)
    graph.add_node("web_search",          web_search_node)

    # Entry point
    graph.set_entry_point("query_analysis")

    # Fixed edges
    graph.add_edge("query_analysis",  "retrieval")
    graph.add_edge("retrieval",       "document_grading")
    graph.add_edge("rewrite_retry",   "retrieval")
    graph.add_edge("no_answer",       END)
    graph.add_edge("web_search",      "generation")   # web results → generation

    # Conditional edge after document grading
    graph.add_conditional_edges(
        "document_grading",
        route_after_grading_with_web,
        {
            "generate":      "generation",
            "rewrite_retry": "rewrite_retry",
            "web_search":    "web_search",
        },
    )

    # Generation → hallucination check
    graph.add_edge("generation", "hallucination_check")

    # Conditional edge after hallucination check
    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {
            "return_answer": END,
            "regenerate":    "generation",
        },
    )

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled


# ── Run the Workflow ──────────────────────────────────────────────────────────

def run_workflow(question: str, chat_history: list = None) -> dict:
    """
    Main entry point to run the RAG workflow.
    Accepts optional chat_history for conversation memory.
    """
    app = build_workflow()

    initial_state: GraphState = {
    "question":           question,
    "rewritten_query":    "",
    "query_type":         "general",
    "retrieved_docs":     [],
    "graded_docs":        [],
    "generation":         "",
    "retry_count":        0,
    "hallucination_flag": False,
    "sources":            [],
    "error":              None,
    "chat_history":       chat_history or [],
    "web_search_used":    False,
    "regen_count":        0,        # ← ADD THIS LINE
}

    logger.info(f"Starting workflow for: '{question}' | history_len={len(chat_history or [])}")
    final_state = app.invoke(initial_state)
    logger.info("Workflow complete.")

    return final_state
