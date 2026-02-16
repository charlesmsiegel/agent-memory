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
