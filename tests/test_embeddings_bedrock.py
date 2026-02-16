"""Tests for BedrockEmbeddings provider."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

from agent_memory.memory.embeddings import EmbeddingProvider
from agent_memory.memory.embeddings_bedrock import BedrockEmbeddings


def _make_bedrock_response(embedding: list[float]) -> dict:
    """Create a mock Bedrock invoke_model response."""
    body = MagicMock()
    body.read.return_value = json.dumps({"embedding": embedding}).encode()
    return {"body": body}


def test_satisfies_protocol() -> None:
    with patch("boto3.client"):
        provider = BedrockEmbeddings()
    assert isinstance(provider, EmbeddingProvider)


def test_dimensions_default() -> None:
    with patch("boto3.client"):
        provider = BedrockEmbeddings()
    assert provider.dimensions == 1024


def test_embed_returns_normalized_vector() -> None:
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = _make_bedrock_response([3.0, 4.0, 0.0])
    with patch("boto3.client", return_value=mock_client):
        provider = BedrockEmbeddings(dimensions=3)
    vec = provider.embed("hello")
    assert len(vec) == 3
    mag = math.sqrt(sum(x * x for x in vec))
    assert abs(mag - 1.0) < 1e-6


def test_embed_batch_calls_embed_per_item() -> None:
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = _make_bedrock_response([1.0, 0.0])
    with patch("boto3.client", return_value=mock_client):
        provider = BedrockEmbeddings(dimensions=2)
    results = provider.embed_batch(["a", "b", "c"])
    assert len(results) == 3
    assert mock_client.invoke_model.call_count == 3
