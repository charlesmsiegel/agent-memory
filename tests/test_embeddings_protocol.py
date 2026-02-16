"""Tests for EmbeddingProvider protocol and helpers."""

from __future__ import annotations

import math

from agent_memory.memory.embeddings import EmbeddingProvider, normalize


class FakeProvider:
    """Minimal provider for protocol verification."""

    def __init__(self, dims: int = 3) -> None:
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, text: str) -> list[float]:
        return normalize([float(len(text))] + [0.0] * (self._dims - 1))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def test_fake_provider_satisfies_protocol() -> None:
    provider: EmbeddingProvider = FakeProvider(dims=4)
    assert provider.dimensions == 4
    vec = provider.embed("hi")
    assert len(vec) == 4
    batch = provider.embed_batch(["a", "bb"])
    assert len(batch) == 2


def test_normalize_unit_vector() -> None:
    vec = normalize([3.0, 4.0])
    mag = math.sqrt(sum(x * x for x in vec))
    assert abs(mag - 1.0) < 1e-6


def test_normalize_zero_vector() -> None:
    vec = normalize([0.0, 0.0, 0.0])
    assert vec == [0.0, 0.0, 0.0]


def test_normalize_handles_nan() -> None:
    vec = normalize([float("nan"), 1.0])
    assert not any(math.isnan(x) for x in vec)
