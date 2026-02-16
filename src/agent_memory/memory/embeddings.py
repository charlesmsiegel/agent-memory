"""Embedding provider protocol, fallback chain, and factory."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface that all embedding providers must satisfy."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector. Returns zero vector if magnitude is zero."""
    arr = np.array(vec, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mag = np.linalg.norm(arr)
    if mag == 0:
        return arr.tolist()
    return (arr / mag).tolist()


import logging

log = logging.getLogger(__name__)


class FallbackChain:
    """Tries providers in order, falling back on any exception."""

    def __init__(self, providers: list[EmbeddingProvider]) -> None:
        if not providers:
            raise ValueError("FallbackChain requires at least one provider")
        self._providers = providers
        self._active: EmbeddingProvider = providers[0]

    @property
    def dimensions(self) -> int:
        return self._providers[0].dimensions

    @property
    def active_provider(self) -> EmbeddingProvider:
        """The provider that last succeeded."""
        return self._active

    def embed(self, text: str) -> list[float]:
        return self._try("embed", text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._try("embed_batch", texts)

    def _try(self, method: str, *args):
        last_err: Exception | None = None
        for provider in self._providers:
            try:
                result = getattr(provider, method)(*args)
                self._active = provider
                return result
            except Exception as exc:
                log.warning("Provider %s.%s failed: %s", type(provider).__name__, method, exc)
                last_err = exc
        raise last_err  # type: ignore[misc]


def resolve_embeddings(cfg: dict) -> EmbeddingProvider:
    """Build an embedding provider (with optional fallback) from config dict."""
    primary = _build_provider(cfg, cfg["provider"])
    fallback_name = cfg.get("fallback")
    if fallback_name:
        fallback = _build_provider(cfg, fallback_name)
        return FallbackChain([primary, fallback])
    return primary


def _build_provider(cfg: dict, name: str) -> EmbeddingProvider:
    if name == "bedrock":
        from .embeddings_bedrock import BedrockEmbeddings
        return BedrockEmbeddings(
            model_id=cfg.get("model_id", "amazon.titan-embed-text-v2:0"),
            region=cfg.get("region", "us-east-1"),
            dimensions=cfg.get("dimensions", 1024),
        )
    if name == "openrouter":
        from .embeddings_openrouter import OpenRouterEmbeddings
        sub = cfg.get("openrouter", {})
        return OpenRouterEmbeddings(
            api_key=sub.get("api_key", ""),
            model=sub.get("model", ""),
            dimensions=sub.get("dimensions", 1024),
            base_url=sub.get("base_url", "https://openrouter.ai/api/v1"),
        )
    if name == "gguf":
        from .embeddings_gguf import LocalGGUFEmbeddings
        sub = cfg.get("gguf", {})
        return LocalGGUFEmbeddings(model_path=sub["model_path"])
    raise ValueError(f"Unknown embedding provider: {name!r}")
