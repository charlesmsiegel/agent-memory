"""Embedding provider using OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

import httpx

from .embeddings import normalize

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEmbeddings:
    """Generate embeddings via OpenRouter's OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a normalized vector."""
        results = self._request([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call (native batching)."""
        if not texts:
            return []
        return self._request(texts)

    def _request(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post(
            "/embeddings",
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to preserve input order
        items = sorted(data["data"], key=lambda d: d["index"])
        return [normalize(item["embedding"]) for item in items]
