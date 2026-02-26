from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # App
    APP_TITLE: str = "📖 AI Handbook Generator"

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Supabase
    SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY: str | None = os.getenv("SUPABASE_SERVICE_KEY")
    SUPABASE_TABLE: str = os.getenv("SUPABASE_TABLE", "document_chunks")

    # Grok (xAI)
    GROK_API_KEY: str | None = os.getenv("GROK_API_KEY")
    GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-4-1-fast-reasoning")
    GROK_ENDPOINT: str = os.getenv("GROK_ENDPOINT", "https://api.x.ai/v1/chat/completions")

    # Handbook
    TARGET_WORDS: int = int(os.getenv("TARGET_WORDS", "20000"))

    # Embeddings (must match Supabase vector dimension)
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))


settings = Settings()
