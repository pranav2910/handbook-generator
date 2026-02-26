from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Iterable

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    content: str
    page: int


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """
    Extract text per page from a PDF (bytes) using pdfplumber.
    Returns list of (page_number, page_text) with 1-based page numbers.
    """
    pages: list[tuple[int, str]] = []

    # ✅ FIX: pdfplumber needs a seekable stream, not raw bytes
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((idx, text))
            else:
                logger.info("No text extracted from page %d (possibly scanned).", idx)

    return pages


def chunk_pages(pages: Iterable[tuple[int, str]]) -> list[Chunk]:
    """
    Chunk each page separately so every chunk has a reliable page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    out: list[Chunk] = []
    for page_no, page_text in pages:
        pieces = splitter.split_text(page_text)
        for p in pieces:
            p = p.strip()
            if p:
                out.append(Chunk(content=p, page=page_no))
    return out


def build_full_text(pages: Iterable[tuple[int, str]]) -> str:
    """
    Join pages into a single string for LightRAG ingestion.
    """
    return "\n\n".join([t for _, t in pages if t.strip()])
