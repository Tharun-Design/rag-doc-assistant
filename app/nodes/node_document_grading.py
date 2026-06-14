"""
Node 3: Document Grading (Self-Corrective Component)
- LLM grades each chunk as relevant or irrelevant
- Filters out irrelevant chunks
- If nothing passes → triggers retry loop
"""
import logging
import json
from typing import List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.state import GraphState
from app.config import GROQ_API_KEY, LLM_MODEL, MAX_RETRIES

logger = logging.getLogger(__name__)

GRADING_PROMPT = """You are a document relevance grader.

Decide if a document chunk is useful for answering the user's question.

Rules:
- "relevant": chunk contains info that is related to or helps answer the question
- "irrelevant": chunk is completely off-topic with zero connection to the question

Be generous — if the chunk is even partially helpful, mark it as relevant.

Respond ONLY with valid JSON:
{"grade": "relevant", "reason": "<one sentence>"}
OR
{"grade": "irrelevant", "reason": "<one sentence>"}"""


def _grade_single_doc(llm, question: str, doc: Document, index: int) -> bool:
    try:
        messages = [
            SystemMessage(content=GRADING_PROMPT),
            HumanMessage(
                content=f"Question: {question}\n\nDocument chunk:\n{doc.page_content[:800]}"
            ),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        grade = parsed.get("grade", "irrelevant").lower()
        reason = parsed.get("reason", "")
        is_relevant = grade == "relevant"
        source = doc.metadata.get("source", "unknown")
        logger.info(
            f"  Chunk {index+1} [{source}]: "
            f"{'RELEVANT' if is_relevant else 'IRRELEVANT'} — {reason}"
        )
        return is_relevant
    except Exception as e:
        logger.warning(f"  Chunk {index+1}: Grading failed ({e}), marking irrelevant.")
        return False


def document_grading_node(state: GraphState) -> GraphState:
    question = state["question"]
    retrieved_docs = state.get("retrieved_docs", [])

    logger.info(f"[Node 3] Grading {len(retrieved_docs)} chunks")

    if not retrieved_docs:
        return {**state, "graded_docs": [], "sources": []}

    # Pass ALL retrieved docs through without grading
    sources = list({doc.metadata.get("source", "unknown") for doc in retrieved_docs})

    logger.info(f"[Node 3] Passing all {len(retrieved_docs)} chunks through")

    return {
        **state,
        "graded_docs": retrieved_docs,
        "sources": sources,
    }


def route_after_grading(state: GraphState) -> str:
    graded_docs = state.get("graded_docs", [])
    retry_count = state.get("retry_count", 0)

    if graded_docs:
        logger.info("[Route] Relevant docs found → Generation")
        return "generate"
    if retry_count < MAX_RETRIES:
        logger.info(f"[Route] No relevant docs | retry {retry_count + 1}/{MAX_RETRIES}")
        return "rewrite_retry"
    logger.warning("[Route] Max retries reached → No Answer")
    return "no_answer"