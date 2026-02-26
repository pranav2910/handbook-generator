from __future__ import annotations

import hashlib
import logging
import numpy as np
import streamlit as st

from config import settings
from db import get_supabase_client
from ingestion import extract_pages, chunk_pages, build_full_text
from rag_engine import RAGEngine, hash_embedding_np
from longwriter import generate_handbook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Handbook Generator", layout="wide")
st.title(settings.APP_TITLE)


def _file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def embedding_as_list(text: str) -> list[float]:
    # Uses same deterministic embedding as LightRAG wrapper for consistency
    v = hash_embedding_np(text, settings.EMBEDDING_DIM)
    return v.astype(float).tolist()


def save_chunks_to_supabase(chunks, document_name: str) -> int:
    """
    Store chunks in Supabase with:
    - content
    - embedding (vector(1536))
    - metadata containing doc name + page for correct citations
    """
    sb = get_supabase_client()
    rows = []

    for i, ch in enumerate(chunks):
        chunk_id = hashlib.sha1(f"{document_name}|{ch.page}|{i}|{ch.content}".encode("utf-8")).hexdigest()
        rows.append(
            {
                "content": ch.content,
                "embedding": embedding_as_list(ch.content),
                "metadata": {
                    "document_name": document_name,
                    "page": int(ch.page),
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                },
            }
        )

    sb.table(settings.SUPABASE_TABLE).insert(rows).execute()
    return len(rows)


# ---- Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "ingested_hashes" not in st.session_state:
    st.session_state.ingested_hashes = set()

if "messages" not in st.session_state:
    st.session_state.messages = []

rag: RAGEngine = st.session_state.rag


# ---- Sidebar: PDF Upload
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_pdfs = st.file_uploader(
        "Upload 2–3 AI-related PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_pdfs:
        for up in uploaded_pdfs:
            file_bytes = up.getvalue()
            file_id = _file_hash(file_bytes)
            doc_name = up.name

            if file_id in st.session_state.ingested_hashes:
                st.info(f"Already ingested: {doc_name}")
                continue

            with st.spinner(f"Extracting: {doc_name}"):
                pages = extract_pages(file_bytes)

            if not pages:
                st.warning(f"No extractable text found in: {doc_name} (scanned PDF?)")
                continue

            with st.spinner(f"Chunking: {doc_name}"):
                chunks = chunk_pages(pages)

            with st.spinner(f"Saving to Supabase (with embeddings): {doc_name}"):
                n = save_chunks_to_supabase(chunks, doc_name)
                st.success(f"Saved {n} chunks to Supabase: {doc_name}")

            full_text = build_full_text(pages)
            with st.spinner(f"Ingesting into LightRAG: {doc_name}"):
                rag.ingest(full_text)
                st.success(f"Ingested into LightRAG: {doc_name}")

            st.session_state.ingested_hashes.add(file_id)


# ---- Main: Chat UI
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input('Ask about the PDFs, or type: "Create a handbook on <topic>"')

if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    if user_msg.lower().startswith("create a handbook"):
        if not uploaded_pdfs:
            reply = "Please upload 2–3 PDFs first (handbook must be grounded in uploaded content)."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            with st.chat_message("assistant"):
                progress = st.progress(0.0)
                status = st.empty()

                def cb(p: float, msg: str):
                    progress.progress(max(0.0, min(1.0, p)))
                    status.write(msg)

                topic = user_msg.replace("Create a handbook on", "").strip() or "the uploaded documents"
                handbook_md = generate_handbook(
                    topic=topic,
                    rag=rag,
                    target_words=settings.TARGET_WORDS,
                    progress_cb=cb,
                )

                st.markdown(handbook_md)
                st.caption(f"Word count: {len(handbook_md.split())}")

                st.download_button(
                    label="Download handbook (Markdown)",
                    data=handbook_md,
                    file_name="handbook.md",
                    mime="text/markdown",
                )

            st.session_state.messages.append({"role": "assistant", "content": handbook_md})

    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching + answering..."):
                answer = rag.query(user_msg)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
