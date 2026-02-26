from __future__ import annotations

from functools import lru_cache
from supabase import Client, create_client
from config import settings


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Returns a Supabase client using Service Role key (server-side only).
    Cached to avoid recreating client on Streamlit reruns.
    """
    missing = []
    if not settings.SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not settings.SUPABASE_SERVICE_KEY:
        missing.append("SUPABASE_SERVICE_KEY")

    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
