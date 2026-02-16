"""Tests for LocalGGUFEmbeddings provider."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from agent_memory.memory.embeddings import EmbeddingProvider
from agent_memory.memory.embeddings_gguf import LocalGGUFEmbeddings


def _make_mock_llama(n_embd: int = 384):
    """Create a mock Llama instance."""
    mock = MagicMock()
    mock.n_embd.return_value = n_embd
    mock.embed.return_value = [[1.0] + [0.0] * (n_embd - 1)]
    return mock


def test_satisfies_protocol() -> None:
    with patch("agent_memory.memory.embeddings_gguf.Llama", return_value=_make_mock_llama()):
        with patch("pathlib.Path.exists", return_value=True):
            provider = LocalGGUFEmbeddings(model_path="/fake/model.gguf")
    assert isinstance(provider, EmbeddingProvider)


def test_dimensions_detected_from_model() -> None:
    mock = _make_mock_llama(n_embd=768)
    with patch("agent_memory.memory.embeddings_gguf.Llama", return_value=mock):
        with patch("pathlib.Path.exists", return_value=True):
            provider = LocalGGUFEmbeddings(model_path="/fake/model.gguf")
    assert provider.dimensions == 768


def test_embed_returns_normalized() -> None:
    mock = _make_mock_llama(n_embd=3)
    mock.embed.return_value = [[3.0, 4.0, 0.0]]
    with patch("agent_memory.memory.embeddings_gguf.Llama", return_value=mock):
        with patch("pathlib.Path.exists", return_value=True):
            provider = LocalGGUFEmbeddings(model_path="/fake/model.gguf")
    vec = provider.embed("hello")
    assert len(vec) == 3
    mag = math.sqrt(sum(x * x for x in vec))
    assert abs(mag - 1.0) < 1e-6


def test_embed_batch() -> None:
    mock = _make_mock_llama(n_embd=2)
    mock.embed.side_effect = [[[1.0, 0.0]], [[0.0, 1.0]]]
    with patch("agent_memory.memory.embeddings_gguf.Llama", return_value=mock):
        with patch("pathlib.Path.exists", return_value=True):
            provider = LocalGGUFEmbeddings(model_path="/fake/model.gguf")
    results = provider.embed_batch(["a", "b"])
    assert len(results) == 2


def test_file_not_found() -> None:
    with patch("agent_memory.memory.embeddings_gguf.Llama", MagicMock()):
        with pytest.raises(FileNotFoundError):
            LocalGGUFEmbeddings(model_path="/nonexistent/model.gguf")
