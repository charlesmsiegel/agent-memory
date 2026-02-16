"""Tests for OpenRouterEmbeddings provider."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

from agent_memory.memory.embeddings import EmbeddingProvider
from agent_memory.memory.embeddings_openrouter import OpenRouterEmbeddings


def _mock_response(embeddings: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "data": [{"embedding": e, "index": i} for i, e in enumerate(embeddings)],
        "model": "test-model",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }
    return resp


def test_satisfies_protocol() -> None:
    provider = OpenRouterEmbeddings(api_key="test", model="m", dimensions=3)
    assert isinstance(provider, EmbeddingProvider)


def test_dimensions() -> None:
    provider = OpenRouterEmbeddings(api_key="k", model="m", dimensions=512)
    assert provider.dimensions == 512


def test_embed_single() -> None:
    provider = OpenRouterEmbeddings(api_key="k", model="m", dimensions=3)
    mock_resp = _mock_response([[3.0, 4.0, 0.0]])
    with patch.object(provider._client, "post", return_value=mock_resp):
        vec = provider.embed("hello")
    assert len(vec) == 3
    mag = math.sqrt(sum(x * x for x in vec))
    assert abs(mag - 1.0) < 1e-6


def test_embed_batch_native() -> None:
    provider = OpenRouterEmbeddings(api_key="k", model="m", dimensions=2)
    mock_resp = _mock_response([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    with patch.object(provider._client, "post", return_value=mock_resp) as mock_post:
        results = provider.embed_batch(["a", "b", "c"])
    assert len(results) == 3
    # Native batch: single HTTP call
    mock_post.assert_called_once()


def test_embed_batch_empty() -> None:
    provider = OpenRouterEmbeddings(api_key="k", model="m", dimensions=2)
    results = provider.embed_batch([])
    assert results == []
