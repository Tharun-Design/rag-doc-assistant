"""
main.py — FastAPI application entry point
Serves the RAG Documentation Assistant with conversation memory support.
"""
import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from app.workflow import run_workflow
from app.utils.ingestion import ingest_files, ingest_urls
from app.utils.vector_store import VectorStore
from app.memory import get_history, add_message, clear_history, list_sessions

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    vs = VectorStore()
    count = vs.get_total_count()
    logger.info(f"ChromaDB ready — {count} chunks stored.")
    yield

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description="Ask questions about technical documentation using RAG + LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    query_type: str
    retry_count: int
    hallucination_checked: bool
    web_search_used: bool
    session_id: str
    error: Optional[str]

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    comment: Optional[str] = None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    vs = VectorStore()
    return {
        "status": "running",
        "total_chunks": vs.get_total_count(),
        "message": "RAG Documentation Assistant is ready.",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    chat_history = get_history(session_id)

    result = run_workflow(
        question=request.question,
        chat_history=chat_history,
    )

    answer = result.get("generation", "No answer generated.")
    sources = result.get("sources", [])

    add_message(session_id, "user", request.question)
    add_message(session_id, "assistant", answer)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        query_type=result.get("query_type", "general"),
        retry_count=result.get("retry_count", 0),
        hallucination_checked=result.get("hallucination_flag", False),
        web_search_used=result.get("web_search_used", False),
        session_id=session_id,
        error=result.get("error"),
    )


@app.post("/ingest")
async def ingest(
    urls: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    total_chunks = 0
    sources = []

    if urls:
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        chunks, srcs = ingest_urls(url_list)
        total_chunks += chunks
        sources.extend(srcs)

    if files:
        import tempfile, os
        for f in files:
            content = await f.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f.filename) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            chunks, srcs = ingest_files([tmp_path])
            total_chunks += chunks
            sources.extend(srcs)
            os.unlink(tmp_path)

    return {"status": "success", "chunks_added": total_chunks, "sources": sources}


@app.get("/documents")
def list_documents():
    vs = VectorStore()
    docs = vs.list_documents()
    return {"total_documents": len(docs), "documents": docs}


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    log_line = (
        f"FEEDBACK | rating={request.rating} | "
        f"question={request.question!r} | "
        f"comment={request.comment!r}\n"
    )
    with open("feedback.log", "a") as f:
        f.write(log_line)
    return {"status": "recorded", "rating": request.rating}


@app.delete("/memory/{session_id}")
def clear_session_memory(session_id: str):
    clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/memory/{session_id}")
def get_session_memory(session_id: str):
    history = get_history(session_id)
    return {"session_id": session_id, "history": history, "turns": len(history) // 2}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)