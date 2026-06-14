"""
Node: Web Search Fallback
- Triggered when ChromaDB retrieval finds no relevant docs after max retries
- Uses Tavily to search the web for an answer
- Feeds web results into generation node as if they were retrieved docs
"""
import logging
import os
from langchain_core.documents import Document
from app.models.state import GraphState

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


def web_search_node(state: GraphState) -> GraphState:
    """
    Uses Tavily to search the web when local docs have no relevant results.
    Converts web results into Document objects so generation node can use them.
    """
    question = state.get("rewritten_query") or state["question"]
    logger.info(f"[Web Search Node] Searching web for: '{question}'")

    if not TAVILY_API_KEY:
        logger.warning("[Web Search Node] No TAVILY_API_KEY set — skipping web search.")
        return {
            **state,
            "graded_docs": [],
            "sources": [],
            "web_search_used": False,
            "error": "Web search unavailable: no API key configured.",
        }

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=question,
            search_depth="basic",
            max_results=4,
        )

        results = response.get("results", [])

        if not results:
            logger.warning("[Web Search Node] No web results found.")
            return {
                **state,
                "graded_docs": [],
                "sources": [],
                "web_search_used": True,
                "error": "No results found via web search.",
            }

        # Convert Tavily results → LangChain Documents
        web_docs = []
        sources = []
        for r in results:
            content = r.get("content", "")
            url = r.get("url", "web")
            title = r.get("title", "Web Result")
            doc = Document(
                page_content=f"{title}\n\n{content}",
                metadata={"source": url, "title": title, "from_web": True},
            )
            web_docs.append(doc)
            sources.append(url)

        logger.info(f"[Web Search Node] Got {len(web_docs)} web results.")

        return {
            **state,
            "graded_docs": web_docs,
            "sources": sources,
            "web_search_used": True,
            "error": None,
        }

    except Exception as e:
        logger.error(f"[Web Search Node] Failed: {e}")
        return {
            **state,
            "graded_docs": [],
            "sources": [],
            "web_search_used": False,
            "error": f"Web search failed: {str(e)}",
        }
