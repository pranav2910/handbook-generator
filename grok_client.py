from __future__ import annotations

import logging
import requests

from config import settings

logger = logging.getLogger(__name__)


class GrokClient:
    def __init__(self) -> None:
        if not settings.GROK_API_KEY:
            raise ValueError("Missing GROK_API_KEY in .env")
        if not settings.GROK_ENDPOINT:
            raise ValueError("Missing GROK_ENDPOINT in .env")

    def chat(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Minimal xAI chat completion call.
        """
        payload = {
            "model": settings.GROK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        resp = requests.post(
            settings.GROK_ENDPOINT,
            headers={
                "Authorization": f"Bearer {settings.GROK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )

        try:
            resp.raise_for_status()
        except Exception:
            logger.error("Grok API error (%s): %s", resp.status_code, resp.text)
            raise

        data = resp.json()
        return data["choices"][0]["message"]["content"]
