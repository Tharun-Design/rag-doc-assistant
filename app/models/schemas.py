"""
schemas.py — Pydantic models for FastAPI request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to answer",
        examples=["How do I add authentication in FastAPI?"]
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    query_type: str
    retry_count: int
    hallucination_checked: bool
    error: Optional[str] = None


class IngestURLRequest(BaseModel):
    urls: List[str] = Field(
        ...,
        description="List of URLs to fetch and ingest"
    )


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_added: int
    sources: List[str]


class DocumentInfo(BaseModel):
    id: str
    source: str
    content_preview: str
    chunk_index: int


class DocumentsResponse(BaseModel):
    total_documents: int
    documents: List[DocumentInfo]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["thumbs_up", "thumbs_down"]
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )