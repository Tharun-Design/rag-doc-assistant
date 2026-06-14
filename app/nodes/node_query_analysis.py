"""
Node 1: Query Analysis
- Classifies query type (how-to, conceptual, troubleshooting, api-reference)
- Rewrites query to improve retrieval quality
- Uses conversation history to understand follow-up questions
"""
import logging
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.state import GraphState
from app.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

QUERY_ANALYSIS_PROMPT = """You are a query analysis expert for a technical documentation assistant.

Given a user's question and optional chat history, you must:
1. Classify the query type into ONE of: how-to, conceptual, troubleshooting, api-reference, general
2. Rewrite the query to be more specific and retrieval-friendly
   - If the question uses pronouns like "it", "this", "that" — resolve them using chat history
   - If it's a follow-up question, make it self-contained

Query type definitions:
- how-to: Step-by-step instructions ("How do I...", "How to...")
- conceptual: Understanding concepts ("What is...", "Explain...", "Why...")
- troubleshooting: Fixing errors ("Error...", "Not working...", "Exception...")
- api-reference: Specific API details ("What parameters...", "What does X return...")
- general: Doesn't fit above categories

Respond ONLY with valid JSON:
{
  "query_type": "<type>",
  "rewritten_query": "<improved self-contained query>",
  "reasoning": "<one sentence why>"
}"""


def query_analysis_node(state: GraphState) -> GraphState:
    logger.info(f"[Node 1] Query Analysis | '{state['question']}'")

    # Build chat history context
    chat_history = state.get("chat_history", [])
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]  # last 3 turns
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent
        )

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )

        user_content = f"User question: {state['question']}"
        if history_text:
            user_content = f"Chat history:\n{history_text}\n\n{user_content}"

        messages = [
            SystemMessage(content=QUERY_ANALYSIS_PROMPT),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        query_type = parsed.get("query_type", "general")
        rewritten_query = parsed.get("rewritten_query", state["question"])
        logger.info(f"[Node 1] Type: '{query_type}' | Rewritten: '{rewritten_query}'")
        return {
            **state,
            "query_type": query_type,
            "rewritten_query": rewritten_query,
        }
    except Exception as e:
        logger.error(f"[Node 1] Failed: {e}. Using original question.")
        return {
            **state,
            "query_type": "general",
            "rewritten_query": state["question"],
        }
