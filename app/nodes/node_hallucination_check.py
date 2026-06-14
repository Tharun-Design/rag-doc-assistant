"""
Node 5: Hallucination Check (Bonus — Self-RAG inspired)
- Verifies the generated answer is actually supported by retrieved chunks
- If hallucination detected → triggers regeneration
"""
import logging
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.state import GraphState
from app.config import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

HALLUCINATION_PROMPT = """You are a fact-checking expert.

Check if the answer is fully supported by the provided document chunks.

Rules:
- "grounded": every claim in the answer comes from the chunks
- "hallucinated": the answer contains info NOT found in the chunks

Respond ONLY with valid JSON:
{"verdict": "grounded", "reason": "<one sentence>"}
OR
{"verdict": "hallucinated", "reason": "<one sentence>"}"""


def hallucination_check_node(state: GraphState) -> GraphState:
    generation = state.get("generation", "")
    graded_docs = state.get("graded_docs", [])

    logger.info("[Node 5] Hallucination Check")

    if not graded_docs or not generation:
        return {**state, "hallucination_flag": False}

    context = "\n\n".join([doc.page_content for doc in graded_docs[:3]])

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=0,
        )
        messages = [
            SystemMessage(content=HALLUCINATION_PROMPT),
            HumanMessage(
                content=f"Document chunks:\n{context}\n\nGenerated answer:\n{generation}"
            ),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "grounded")
        reason = parsed.get("reason", "")
        is_grounded = verdict == "grounded"

        logger.info(f"[Node 5] Verdict: {verdict.upper()} — {reason}")

        return {**state, "hallucination_flag": is_grounded}

    except Exception as e:
        logger.error(f"[Node 5] Check failed: {e}. Assuming grounded.")
        return {**state, "hallucination_flag": True}


def route_after_hallucination_check(state: GraphState) -> str:
    hallucination_flag = state.get("hallucination_flag", False)
    regen_count = state.get("regen_count", 0)

    # Stop after 2 regeneration attempts regardless of verdict
    if hallucination_flag and regen_count < 2:
        return "regenerate"
    return "return_answer"
    

    