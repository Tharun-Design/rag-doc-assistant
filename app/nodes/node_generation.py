"""
Node 4: Answer Generation
- Generates answer using only the graded (relevant) chunks
- Includes source citations in the response
- Formats answer based on query type
- Uses conversation history for contextual answers
- Handles web search results differently from doc chunks
"""
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.state import GraphState
from app.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

GENERATION_PROMPT = """You are a helpful technical documentation assistant.

Answer the user's question using ONLY the provided document chunks.


Rules:
- Do NOT use any outside knowledge — only what is in the chunks
- Do NOT add greetings, introductions, or offers to help further
- Do NOT say things like "Hello!", "I'd be happy to help", "Let me know if..."
- Start directly with the answer content
- Cite sources like this: [Source: filename or URL]
- If the chunks don't fully answer the question, say so honestly
- For how-to questions: use numbered steps
- For conceptual questions: give a clear explanation
- For troubleshooting: give the error cause and fix
- If answering a follow-up question, use the chat history for context"""

WEB_GENERATION_PROMPT = """You are a helpful technical assistant.

The user's question was not found in the local documentation, so web search results are provided.
Answer using ONLY the web search results below.

Rules:
- Be clear and accurate
- Always cite the source URL like this: [Source: URL]
- If results don't answer the question, say so honestly
- For how-to questions: use numbered steps"""


def generation_node(state: GraphState) -> GraphState:
    question = state["question"]
    query_type = state.get("query_type", "general")
    graded_docs = state.get("graded_docs", [])
    chat_history = state.get("chat_history", [])
    web_search_used = state.get("web_search_used", False)

    logger.info(f"[Node 4] Generation | {len(graded_docs)} chunks | type={query_type} | web={web_search_used}")

    if not graded_docs:
        return {
            **state,
            "generation": "I don't have enough information in the documentation to answer this question.",
            "hallucination_flag": False,
        }

    # Build context from docs
    context_parts = []
    for i, doc in enumerate(graded_docs):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Chunk {i+1} | Source: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # Build chat history string (last 6 messages = 3 turns)
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent
        )

    # Choose prompt based on whether we used web search
    system_prompt = WEB_GENERATION_PROMPT if web_search_used else GENERATION_PROMPT

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )

        user_content = f"Question: {question}\n\nDocument chunks:\n{context}"
        if history_text:
            user_content = f"Chat history:\n{history_text}\n\n{user_content}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        generation = response.content.strip()

        logger.info(f"[Node 4] Answer generated ({len(generation)} chars)")

        return {
            **state,
            "generation": generation,
            "hallucination_flag": False,
            "regen_count": state.get("regen_count", 0) + 1,
        }

    except Exception as e:
        logger.error(f"[Node 4] Generation failed: {e}")
        return {
            **state,
            "generation": f"Error generating answer: {str(e)}",
            "hallucination_flag": False,
        }