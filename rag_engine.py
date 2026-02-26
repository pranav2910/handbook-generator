from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from typing import List, Optional

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from config import settings
from grok_client import GrokClient

logger = logging.getLogger(__name__)

_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_THREAD: threading.Thread | None = None
_LOOP_LOCK = threading.Lock()


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    """
    Create one dedicated asyncio loop that runs forever in a background thread.
    Streamlit reruns won't break it.
    """
    global _LOOP, _LOOP_THREAD

    with _LOOP_LOCK:
        if _LOOP is not None and _LOOP.is_running():
            return _LOOP

        _LOOP = asyncio.new_event_loop()

        def _runner(loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        _LOOP_THREAD = threading.Thread(target=_runner, args=(_LOOP,), daemon=True)
        _LOOP_THREAD.start()
        return _LOOP


def _run(coro):
    loop = _ensure_background_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


def hash_embedding_np(text: str, dim: int) -> np.ndarray:
    """
    Deterministic local embedding vector (works offline).
    Produces a normalized np.ndarray (dim,).
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split()[:2000]:
        h = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        vec[idx] += 1.0
    n = np.linalg.norm(vec)
    return vec if n == 0 else (vec / n)


class RAGEngine:
    """
    LightRAG wrapper safe for Streamlit:
    - runs LightRAG async calls on a dedicated background loop thread
    """

    def __init__(self, working_dir: str = "./rag_storage"):
        self._client = GrokClient()

        async def llm_model_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[list] = None,  # ✅ no mutable default
            keyword_extraction: bool = False,
            **kwargs,
        ) -> str:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            return self._client.chat(full_prompt, temperature=0.2)

        async def embedding_func(texts: List[str]) -> np.ndarray:
            embs = [hash_embedding_np(t, settings.EMBEDDING_DIM) for t in texts]
            return np.vstack(embs)

        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=settings.EMBEDDING_DIM,
                max_token_size=8192,
                func=embedding_func,
            ),
        )

        _run(self.rag.initialize_storages())

    def ingest(self, text: str) -> None:
        logger.info("Ingesting document into LightRAG...")
        _run(self.rag.ainsert(text))

    def query(self, question: str) -> str:
        logger.info("Querying LightRAG...")
        param = QueryParam(mode="mix")
        out = _run(self.rag.aquery(question, param=param))
        return out if isinstance(out, str) else str(out)
