"""Tests for resolve_embeddings factory."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from agent_memory.memory.embeddings import FallbackChain, resolve_embeddings


def test_bedrock_provider() -> None:
    cfg = {
        "provider": "bedrock",
        "model_id": "amazon.titan-embed-text-v2:0",
        "region": "us-east-1",
        "dimensions": 1024,
    }
    with patch("boto3.client"):
        provider = resolve_embeddings(cfg)
    from agent_memory.memory.embeddings_bedrock import BedrockEmbeddings
    assert isinstance(provider, BedrockEmbeddings)


def test_openrouter_provider() -> None:
    cfg = {
        "provider": "openrouter",
        "openrouter": {
            "api_key": "test-key",
            "model": "thenlper/gte-large",
            "dimensions": 1024,
        },
    }
    provider = resolve_embeddings(cfg)
    from agent_memory.memory.embeddings_openrouter import OpenRouterEmbeddings
    assert isinstance(provider, OpenRouterEmbeddings)


def test_gguf_provider() -> None:
    mock_llama = MagicMock()
    mock_llama.n_embd.return_value = 384
    cfg = {
        "provider": "gguf",
        "gguf": {"model_path": "/fake/model.gguf"},
    }
    with patch("agent_memory.memory.embeddings_gguf.Llama", return_value=mock_llama):
        with patch("pathlib.Path.exists", return_value=True):
            provider = resolve_embeddings(cfg)
    from agent_memory.memory.embeddings_gguf import LocalGGUFEmbeddings
    assert isinstance(provider, LocalGGUFEmbeddings)


def test_fallback_chain() -> None:
    cfg = {
        "provider": "bedrock",
        "model_id": "amazon.titan-embed-text-v2:0",
        "region": "us-east-1",
        "dimensions": 1024,
        "fallback": "openrouter",
        "openrouter": {
            "api_key": "test-key",
            "model": "m",
            "dimensions": 1024,
        },
    }
    with patch("boto3.client"):
        provider = resolve_embeddings(cfg)
    assert isinstance(provider, FallbackChain)


def test_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        resolve_embeddings({"provider": "magic"})
