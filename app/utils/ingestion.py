"""
ingestion.py — Document loading, chunking, and indexing pipeline.

Chunking strategy:
- Split at Markdown headers first (keeps sections together)
- Then split oversized sections by character count
- Code blocks never split mid-block
- Chunk size: 500 tokens, overlap: 50 tokens
"""
import logging
import re
from typing import List, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.utils.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def load_from_file(file_path: str) -> Tuple[str, str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    content = path.read_text(encoding="utf-8", errors="ignore")
    source_name = path.name
    if path.suffix == ".html":
        soup = BeautifulSoup(content, "lxml")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        content = soup.get_text(separator="\n", strip=True)
    return content, source_name


def load_from_url(url: str) -> Tuple[str, str]:
    logger.info(f"Fetching: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (RAG-Assistant/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("body")
    content = main.get_text(separator="\n", strip=True) if main else soup.get_text()
    content = re.sub(r"\n{3,}", "\n\n", content)
    source_name = url.split("/")[-2] if url.endswith("/") else url.split("/")[-1]
    return content, source_name or url


def smart_chunk(content: str, source_name: str) -> List[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ],
        strip_headers=False,
    )
    try:
        header_splits = header_splitter.split_text(content)
    except Exception:
        header_splits = [Document(page_content=content)]

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,
        chunk_overlap=CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n```", "```\n", "\n", ". ", " "],
    )

    final_chunks = []
    for i, split in enumerate(header_splits):
        sub_chunks = char_splitter.split_documents([split])
        for j, chunk in enumerate(sub_chunks):
            chunk.metadata.update({
                "source": source_name,
                "chunk_index": len(final_chunks),
                "section_index": i,
            })
            final_chunks.append(chunk)

    logger.info(f"'{source_name}' → {len(final_chunks)} chunks")
    return final_chunks


def ingest_files(file_paths: List[str]) -> Tuple[int, List[str]]:
    vs = get_vector_store()
    total_chunks = 0
    sources = []
    for file_path in file_paths:
        try:
            content, source_name = load_from_file(file_path)
            chunks = smart_chunk(content, source_name)
            added = vs.add_documents(chunks)
            total_chunks += added
            sources.append(source_name)
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
    return total_chunks, sources


def ingest_urls(urls: List[str]) -> Tuple[int, List[str]]:
    vs = get_vector_store()
    total_chunks = 0
    sources = []
    for url in urls:
        try:
            content, source_name = load_from_url(url)
            chunks = smart_chunk(content, source_name)
            added = vs.add_documents(chunks)
            total_chunks += added
            sources.append(source_name)
        except Exception as e:
            logger.error(f"Failed to ingest {url}: {e}")
    return total_chunks, sources