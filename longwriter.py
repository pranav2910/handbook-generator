from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from typing import Any, Callable, Optional

from config import settings
from db import get_supabase_client
from grok_client import GrokClient

logger = logging.getLogger(__name__)


# ----------------------------
# Utilities
# ----------------------------

def _keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out[:12]


def _chunk_fingerprint(content: str) -> str:
    norm = re.sub(r"\s+", " ", content.strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


# ----------------------------
# Fetch chunks from Supabase
# ----------------------------

def fetch_source_chunks(
    topic: str,
    per_doc: int = 30,
    max_total: int = 120,
) -> list[dict[str, Any]]:

    sb = get_supabase_client()
    table = settings.SUPABASE_TABLE
    keys = _keywords(topic)

    resp = sb.table(table).select("content,metadata").limit(2000).execute()
    rows = resp.data or []

    if not rows:
        return []

    def score(row):
        text = (row.get("content") or "").lower()
        return sum(1 for k in keys if k in text)

    by_doc = defaultdict(list)

    for r in rows:
        md = r.get("metadata") or {}
        doc = md.get("document_name", "unknown.pdf")
        by_doc[doc].append(r)

    for doc in by_doc:
        by_doc[doc].sort(key=score, reverse=True)

    docs = sorted(by_doc.keys())

    picked = []
    seen = set()

    round_idx = 0

    while len(picked) < max_total:
        progressed = False

        for doc in docs:
            lst = by_doc[doc]

            if round_idx < len(lst) and round_idx < per_doc:

                fp = _chunk_fingerprint(lst[round_idx]["content"])

                if fp not in seen:
                    picked.append(lst[round_idx])
                    seen.add(fp)

                progressed = True

                if len(picked) >= max_total:
                    break

        if not progressed:
            break

        round_idx += 1

    logger.info("Fetched %d chunks from Supabase", len(picked))

    return picked


# ----------------------------
# Build source block
# ----------------------------

def build_sources_block(
    chunks: list[dict[str, Any]],
    max_chars_each: int = 1500,
) -> str:

    parts = []

    for i, row in enumerate(chunks, start=1):

        md = row.get("metadata") or {}

        doc = md.get("document_name", "unknown.pdf")
        page = md.get("page", 1)

        content = (row.get("content") or "").strip()
        content = content[:max_chars_each]

        cite = f"[source: {doc}, p.{page}]"

        parts.append(f"SOURCE {i}: {cite}\n{content}")

    return "\n\n".join(parts)


# ----------------------------
# Outline generation
# ----------------------------

def generate_outline(client, topic, sources_block):

    prompt = f"""
You are writing a professional handbook.

TASK:
Create a very detailed outline for a {settings.TARGET_WORDS}+ word handbook on:

{topic}

STRICT REQUIREMENTS:

Include:

Part I, Part II, Part III...
Chapters
Sections

Each line MUST include citation.

Example format:

Part I: Introduction [source: file.pdf, p.1]

Chapter 1: Overview [source: file.pdf, p.2]

Section 1.1: Definition [source: file.pdf, p.3]

Use ONLY sources.

SOURCES:

{sources_block}
"""

    return client.chat(prompt, temperature=0.2)


# ----------------------------
# Extract headings
# ----------------------------

def extract_outline_items(outline_text: str) -> list[str]:

    items = []

    for line in outline_text.splitlines():

        t = line.strip()

        if not t:
            continue

        if (
            t.lower().startswith("part ")
            or t.lower().startswith("chapter ")
            or t.lower().startswith("section ")
        ):
            items.append(t)

    return items


# ----------------------------
# Section generation
# ----------------------------

def generate_section(
    client,
    topic,
    section_title,
    sources_block,
    covered,
):

    prompt = f"""
You are writing ONE handbook section.

TOPIC: {topic}

SECTION: {section_title}

ALREADY COVERED:

{covered}

STRICT RULES:

Write 800–1200 words minimum.

Use ONLY SOURCES.

Every paragraph MUST include citation:

[source: document.pdf, p.X]

Do NOT hallucinate.

Do NOT repeat earlier content.

End with Key takeaways bullets.

SOURCES:

{sources_block}
"""

    return client.chat(prompt, temperature=0.2)


# ----------------------------
# MAIN HANDBOOK GENERATOR
# ----------------------------

def generate_handbook(
    topic: str,
    rag,
    target_words: int = settings.TARGET_WORDS,
    progress_cb: Optional[Callable] = None,
) -> str:

    client = GrokClient()

    chunks = fetch_source_chunks(topic)

    if not chunks:
        return "No chunks found."

    sources_block = build_sources_block(chunks)

    if progress_cb:
        progress_cb(0.05, "Generating outline")

    outline = generate_outline(client, topic, sources_block)

    sections = extract_outline_items(outline)

    logger.info("Outline contains %d sections", len(sections))

    out = [outline, "\n---\n"]

    covered = []

    words = len(" ".join(out).split())

    sec_index = 0

    total_sections = max(len(sections), 1)

    while words < target_words:

        sec = sections[sec_index % total_sections]

        if progress_cb:

            progress = min(0.1 + 0.9 * (words / target_words), 0.99)

            progress_cb(progress, f"Writing {sec}")

        text = generate_section(
            client,
            topic,
            sec,
            sources_block,
            "\n".join(covered[-20:]),
        )

        out.append(text)

        covered.append(sec)

        words = len(" ".join(out).split())

        logger.info("Total words: %d", words)

        sec_index += 1

    if progress_cb:
        progress_cb(1.0, "Completed")

    return "\n\n".join(out)
