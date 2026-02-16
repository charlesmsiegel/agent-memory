"""Tests for FallbackChain."""

from __future__ import annotations

import pytest

from agent_memory.memory.embeddings import EmbeddingProvider, FallbackChain, normalize


class WorkingProvider:
    def __init__(self, dims: int = 3, tag: str = "working") -> None:
        self._dims = dims
        self.tag = tag

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, text: str) -> list[float]:
        return normalize([1.0] + [0.0] * (self._dims - 1))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class BrokenProvider:
    @property
    def dimensions(self) -> int:
        return 3

    def embed(self, text: str) -> list[float]:
        raise ConnectionError("provider down")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise ConnectionError("provider down")


def test_satisfies_protocol() -> None:
    chain = FallbackChain([WorkingProvider()])
    assert isinstance(chain, EmbeddingProvider)


def test_uses_primary_when_healthy() -> None:
    primary = WorkingProvider(tag="primary")
    fallback = WorkingProvider(tag="fallback")
    chain = FallbackChain([primary, fallback])
    chain.embed("test")
    assert chain.active_provider is primary


def test_falls_back_on_error() -> None:
    broken = BrokenProvider()
    healthy = WorkingProvider(tag="healthy")
    chain = FallbackChain([broken, healthy])
    vec = chain.embed("test")
    assert len(vec) == 3
    assert chain.active_provider is healthy


def test_all_fail_raises_last_error() -> None:
    chain = FallbackChain([BrokenProvider(), BrokenProvider()])
    with pytest.raises(ConnectionError):
        chain.embed("test")


def test_dimensions_from_first_provider() -> None:
    chain = FallbackChain([WorkingProvider(dims=512), WorkingProvider(dims=768)])
    assert chain.dimensions == 512


def test_dimensions_follows_active_after_fallback() -> None:
    """After fallback, dimensions should reflect the active (fallback) provider."""
    chain = FallbackChain([BrokenProvider(), WorkingProvider(dims=768)])
    chain.embed("test")  # triggers fallback
    assert chain.active_provider.dimensions == 768
    assert chain.dimensions == 768


def test_embed_batch_fallback() -> None:
    chain = FallbackChain([BrokenProvider(), WorkingProvider(dims=3)])
    results = chain.embed_batch(["a", "b"])
    assert len(results) == 2


def test_empty_providers_raises() -> None:
    with pytest.raises(ValueError):
        FallbackChain([])
