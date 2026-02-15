# OpenClaw Feature Port — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port missing OpenClaw features into agent_memory: multi-provider embeddings (Bedrock, OpenRouter, local GGUF) with fallback chain, citations, prompt modes, bootstrap truncation, agent config (sandbox, group chat, heartbeat).

**Architecture:** Bottom-up layering. Abstract embedding protocol first, then concrete providers, then fallback chain, then config dataclasses, then context features, then application features, then facade wiring. Each layer is independently testable.

**Tech Stack:** Python 3.11+, pydantic, boto3, httpx (new), llama-cpp-python (optional), SQLite, numpy

---

### Task 1: EmbeddingProvider Protocol + _normalize Helper

**Files:**
- Create: `src/agent_memory/memory/embeddings.py` (replace existing)
- Create: `tests/test_embeddings_protocol.py`

**Step 1: Write the failing test**

```python
# tests/test_embeddings_protocol.py
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embeddings_protocol.py -v`
Expected: FAIL — ImportError (module has no `EmbeddingProvider` or `normalize`)

**Step 3: Write the implementation**

Replace `src/agent_memory/memory/embeddings.py` entirely:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_embeddings_protocol.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/agent_memory/memory/embeddings.py tests/test_embeddings_protocol.py
git commit -m "feat: add EmbeddingProvider protocol and normalize helper"
```

---

### Task 2: Extract BedrockEmbeddings to Dedicated Module

**Files:**
- Create: `src/agent_memory/memory/embeddings_bedrock.py`
- Create: `tests/test_embeddings_bedrock.py`
- Modify: `src/agent_memory/memory/manager.py` (import path)
- Modify: `src/agent_memory/memory/facts.py` (import path)
- Modify: `src/agent_memory/facade.py` (import path)
- Modify: `src/agent_memory/integration.py` (import path)

**Step 1: Write the failing test**

```python
# tests/test_embeddings_bedrock.py
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embeddings_bedrock.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write the implementation**

```python
# src/agent_memory/memory/embeddings_bedrock.py
"""Embedding provider using AWS Bedrock Titan."""

from __future__ import annotations

import json

import boto3

from .embeddings import normalize


class BedrockEmbeddings:
    """Generate text embeddings via AWS Bedrock Titan."""

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimensions: int = 1024,
    ) -> None:
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a normalized vector."""
        body = json.dumps({"inputText": text, "dimensions": self._dimensions})
        response = self._client.invoke_model(modelId=self._model_id, body=body)
        result = json.loads(response["body"].read())
        vec = result["embedding"]
        return normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Titan doesn't support native batching, so we loop."""
        return [self.embed(t) for t in texts]
```

**Step 4: Update imports in dependent files**

In `src/agent_memory/memory/manager.py` change:
- `from .embeddings import BedrockEmbeddings` → `from .embeddings import EmbeddingProvider`
- Type hint `embeddings: BedrockEmbeddings` → `embeddings: EmbeddingProvider`

In `src/agent_memory/memory/facts.py` change:
- `from .embeddings import BedrockEmbeddings` → `from .embeddings import EmbeddingProvider`
- Type hint `embeddings: BedrockEmbeddings` → `embeddings: EmbeddingProvider`

In `src/agent_memory/facade.py` change:
- `from .memory.embeddings import BedrockEmbeddings` → `from .memory.embeddings_bedrock import BedrockEmbeddings`

In `src/agent_memory/integration.py` change:
- `from pyclawmem.memory.embeddings import BedrockEmbeddings` → `from pyclawmem.memory.embeddings_bedrock import BedrockEmbeddings`

**Step 5: Run all tests to verify nothing broke**

Run: `python -m pytest tests/ -v`
Expected: All pass

**Step 6: Commit**

```bash
git add src/agent_memory/memory/embeddings_bedrock.py tests/test_embeddings_bedrock.py \
  src/agent_memory/memory/embeddings.py src/agent_memory/memory/manager.py \
  src/agent_memory/memory/facts.py src/agent_memory/facade.py \
  src/agent_memory/integration.py
git commit -m "refactor: extract BedrockEmbeddings to dedicated module, use EmbeddingProvider protocol"
```

---

### Task 3: OpenRouterEmbeddings Provider

**Files:**
- Create: `src/agent_memory/memory/embeddings_openrouter.py`
- Create: `tests/test_embeddings_openrouter.py`
- Modify: `pyproject.toml` (add httpx dep)

**Step 1: Add httpx dependency**

In `pyproject.toml`, add `"httpx>=0.27"` to the `dependencies` list.

**Step 2: Write the failing test**

```python
# tests/test_embeddings_openrouter.py
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
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_embeddings_openrouter.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 4: Write the implementation**

```python
# src/agent_memory/memory/embeddings_openrouter.py
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
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_embeddings_openrouter.py -v`
Expected: 5 passed

**Step 6: Commit**

```bash
git add src/agent_memory/memory/embeddings_openrouter.py \
  tests/test_embeddings_openrouter.py pyproject.toml
git commit -m "feat: add OpenRouterEmbeddings provider with native batching"
```

---

### Task 4: LocalGGUFEmbeddings Provider

**Files:**
- Create: `src/agent_memory/memory/embeddings_gguf.py`
- Create: `tests/test_embeddings_gguf.py`
- Modify: `pyproject.toml` (add optional dep)

**Step 1: Add optional dependency**

In `pyproject.toml`, add under `[project.optional-dependencies]`:
```toml
local = ["llama-cpp-python>=0.3"]
```

**Step 2: Write the failing test**

```python
# tests/test_embeddings_gguf.py
"""Tests for LocalGGUFEmbeddings provider."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

from agent_memory.memory.embeddings import EmbeddingProvider
from agent_memory.memory.embeddings_gguf import LocalGGUFEmbeddings


def _make_mock_llama(n_embd: int = 384):
    """Create a mock Llama instance."""
    mock = MagicMock()
    mock.n_embd.return_value = n_embd
    # embed() returns list of list of floats
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
    import pytest
    with pytest.raises(FileNotFoundError):
        LocalGGUFEmbeddings(model_path="/nonexistent/model.gguf")


def test_import_error_without_llama_cpp() -> None:
    import pytest
    with patch.dict("sys.modules", {"llama_cpp": None}):
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(ImportError, match="llama-cpp-python"):
                # Force reimport to trigger the ImportError
                import importlib
                import agent_memory.memory.embeddings_gguf as mod
                importlib.reload(mod)
                mod.LocalGGUFEmbeddings(model_path="/fake/model.gguf")
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_embeddings_gguf.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 4: Write the implementation**

```python
# src/agent_memory/memory/embeddings_gguf.py
"""Embedding provider using a local GGUF model via llama-cpp-python."""

from __future__ import annotations

from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment,misc]

from .embeddings import normalize


class LocalGGUFEmbeddings:
    """In-process GGUF embedding model. No external server needed.

    Requires llama-cpp-python: ``pip install llama-cpp-python``
    """

    def __init__(self, model_path: str) -> None:
        """Load a GGUF model from disk.

        Args:
            model_path: Absolute path to a .gguf file. Must exist.

        Raises:
            ImportError: If llama-cpp-python is not installed.
            FileNotFoundError: If model_path does not exist.
        """
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is required for local GGUF embeddings. "
                "Install it with: pip install llama-cpp-python"
            )
        if not Path(model_path).exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        self._model = Llama(model_path=model_path, embedding=True, verbose=False)
        self._dimensions = self._model.n_embd()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a normalized vector."""
        result = self._model.embed(text)
        # llama-cpp-python returns list[list[float]] even for single input
        vec = result[0] if isinstance(result[0], list) else result
        return normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts sequentially."""
        return [self.embed(t) for t in texts]
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_embeddings_gguf.py -v`
Expected: All pass (tests mock llama_cpp)

**Step 6: Commit**

```bash
git add src/agent_memory/memory/embeddings_gguf.py tests/test_embeddings_gguf.py pyproject.toml
git commit -m "feat: add LocalGGUFEmbeddings provider (in-process, no server)"
```

---

### Task 5: FallbackChain

**Files:**
- Modify: `src/agent_memory/memory/embeddings.py`
- Create: `tests/test_fallback_chain.py`

**Step 1: Write the failing test**

```python
# tests/test_fallback_chain.py
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


def test_embed_batch_fallback() -> None:
    chain = FallbackChain([BrokenProvider(), WorkingProvider(dims=3)])
    results = chain.embed_batch(["a", "b"])
    assert len(results) == 2


def test_empty_providers_raises() -> None:
    with pytest.raises(ValueError):
        FallbackChain([])
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fallback_chain.py -v`
Expected: FAIL — ImportError (no `FallbackChain`)

**Step 3: Add FallbackChain to embeddings.py**

Append to `src/agent_memory/memory/embeddings.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fallback_chain.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add src/agent_memory/memory/embeddings.py tests/test_fallback_chain.py
git commit -m "feat: add FallbackChain embedding provider wrapper"
```

---

### Task 6: resolve_embeddings Factory

**Files:**
- Modify: `src/agent_memory/memory/embeddings.py`
- Create: `tests/test_resolve_embeddings.py`

**Step 1: Write the failing test**

```python
# tests/test_resolve_embeddings.py
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_resolve_embeddings.py -v`
Expected: FAIL — ImportError (no `resolve_embeddings`)

**Step 3: Add factory to embeddings.py**

Append to `src/agent_memory/memory/embeddings.py`:

```python
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
        api_key = sub.get("api_key", "")
        return OpenRouterEmbeddings(
            api_key=api_key,
            model=sub.get("model", ""),
            dimensions=sub.get("dimensions", 1024),
            base_url=sub.get("base_url", "https://openrouter.ai/api/v1"),
        )
    if name == "gguf":
        from .embeddings_gguf import LocalGGUFEmbeddings
        sub = cfg.get("gguf", {})
        return LocalGGUFEmbeddings(model_path=sub["model_path"])
    raise ValueError(f"Unknown embedding provider: {name!r}")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_resolve_embeddings.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/agent_memory/memory/embeddings.py tests/test_resolve_embeddings.py
git commit -m "feat: add resolve_embeddings factory with provider routing"
```

---

### Task 7: Update MemoryManager + FactStore Type Hints

**Files:**
- Modify: `src/agent_memory/memory/manager.py`
- Modify: `src/agent_memory/memory/facts.py`

**Step 1: Update manager.py**

In `src/agent_memory/memory/manager.py`:
- Change import: `from .embeddings import BedrockEmbeddings` → `from .embeddings import EmbeddingProvider`
- Change `__init__` signature: `embeddings: BedrockEmbeddings` → `embeddings: EmbeddingProvider`

(These changes were already specified in Task 2 Step 4 — verify they're in place.)

**Step 2: Update facts.py**

In `src/agent_memory/memory/facts.py`:
- Change import: `from .embeddings import BedrockEmbeddings` → `from .embeddings import EmbeddingProvider`
- Change `__init__` signature: `embeddings: BedrockEmbeddings` → `embeddings: EmbeddingProvider`

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

**Step 4: Commit (if any changes were needed)**

```bash
git add src/agent_memory/memory/manager.py src/agent_memory/memory/facts.py
git commit -m "refactor: use EmbeddingProvider protocol in MemoryManager and FactStore"
```

---

### Task 8: AgentConfig Dataclasses

**Files:**
- Create: `src/agent_memory/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for AgentConfig and related dataclasses."""

from __future__ import annotations

from agent_memory.config import (
    AgentConfig,
    GroupChatConfig,
    HeartbeatConfig,
    SandboxConfig,
)


def test_sandbox_defaults() -> None:
    s = SandboxConfig()
    assert s.mode == "off"
    assert s.workspace_access == "rw"


def test_group_chat_defaults() -> None:
    g = GroupChatConfig()
    assert g.enabled is False
    assert g.respond_to_mentions is True
    assert g.respond_to_direct is True
    assert g.quiet_unless_mentioned is False


def test_heartbeat_defaults() -> None:
    h = HeartbeatConfig()
    assert h.enabled is False
    assert h.interval_minutes == 15
    assert h.quiet_start is None
    assert h.quiet_end is None
    assert h.tasks == []


def test_agent_config_defaults() -> None:
    a = AgentConfig()
    assert a.id == "default"
    assert a.name is None
    assert a.sandbox.mode == "off"
    assert a.group_chat.enabled is False
    assert a.heartbeat.enabled is False


def test_agent_config_from_dict_full() -> None:
    data = {
        "id": "bot1",
        "name": "TestBot",
        "sandbox": {"mode": "all", "workspace_access": "ro"},
        "group_chat": {"enabled": True, "quiet_unless_mentioned": True},
        "heartbeat": {
            "enabled": True,
            "interval_minutes": 5,
            "quiet_hours": {"start": "22:00", "end": "07:00"},
        },
    }
    a = AgentConfig.from_dict(data)
    assert a.id == "bot1"
    assert a.name == "TestBot"
    assert a.sandbox.mode == "all"
    assert a.sandbox.workspace_access == "ro"
    assert a.group_chat.enabled is True
    assert a.group_chat.quiet_unless_mentioned is True
    assert a.heartbeat.enabled is True
    assert a.heartbeat.interval_minutes == 5
    assert a.heartbeat.quiet_start == "22:00"
    assert a.heartbeat.quiet_end == "07:00"


def test_agent_config_from_dict_empty() -> None:
    a = AgentConfig.from_dict({})
    assert a.id == "default"
    assert a.sandbox.mode == "off"


def test_agent_config_from_dict_partial() -> None:
    a = AgentConfig.from_dict({"id": "x", "sandbox": {"mode": "non-main"}})
    assert a.id == "x"
    assert a.sandbox.mode == "non-main"
    assert a.sandbox.workspace_access == "rw"  # default preserved
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write the implementation**

```python
# src/agent_memory/config.py
"""Agent-level configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SandboxConfig:
    """Application-level sandbox policy."""

    mode: Literal["off", "non-main", "all"] = "off"
    workspace_access: Literal["none", "ro", "rw"] = "rw"


@dataclass
class GroupChatConfig:
    """Group chat response behavior."""

    enabled: bool = False
    respond_to_mentions: bool = True
    respond_to_direct: bool = True
    quiet_unless_mentioned: bool = False


@dataclass
class HeartbeatConfig:
    """Periodic task scheduling."""

    enabled: bool = False
    interval_minutes: int = 15
    quiet_start: str | None = None  # "HH:MM"
    quiet_end: str | None = None  # "HH:MM"
    tasks: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Top-level agent configuration."""

    id: str = "default"
    name: str | None = None
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    group_chat: GroupChatConfig = field(default_factory=GroupChatConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)

    @classmethod
    def from_dict(cls, data: dict) -> AgentConfig:
        """Build from a config dict, filling defaults for missing keys."""
        sb = data.get("sandbox", {})
        gc = data.get("group_chat", {})
        hb = data.get("heartbeat", {})
        quiet = hb.get("quiet_hours", {})
        return cls(
            id=data.get("id", "default"),
            name=data.get("name"),
            sandbox=SandboxConfig(
                mode=sb.get("mode", "off"),
                workspace_access=sb.get("workspace_access", "rw"),
            ),
            group_chat=GroupChatConfig(
                enabled=gc.get("enabled", False),
                respond_to_mentions=gc.get("respond_to_mentions", True),
                respond_to_direct=gc.get("respond_to_direct", True),
                quiet_unless_mentioned=gc.get("quiet_unless_mentioned", False),
            ),
            heartbeat=HeartbeatConfig(
                enabled=hb.get("enabled", False),
                interval_minutes=hb.get("interval_minutes", 15),
                quiet_start=quiet.get("start"),
                quiet_end=quiet.get("end"),
                tasks=hb.get("tasks", []),
            ),
        )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add src/agent_memory/config.py tests/test_config.py
git commit -m "feat: add AgentConfig, SandboxConfig, GroupChatConfig, HeartbeatConfig"
```

---

### Task 9: CitationMode, PromptMode Enums + citation Property

**Files:**
- Modify: `src/agent_memory/memory/types.py`
- Create: `tests/test_types_enums.py`

**Step 1: Write the failing test**

```python
# tests/test_types_enums.py
"""Tests for CitationMode, PromptMode, and citation property."""

from __future__ import annotations

from agent_memory.memory.types import (
    CitationMode,
    MemorySearchResult,
    MemorySource,
    PromptMode,
)


def test_citation_mode_values() -> None:
    assert CitationMode.AUTO == "auto"
    assert CitationMode.ON == "on"
    assert CitationMode.OFF == "off"


def test_prompt_mode_values() -> None:
    assert PromptMode.FULL == "full"
    assert PromptMode.MINIMAL == "minimal"
    assert PromptMode.NONE == "none"


def test_citation_property() -> None:
    r = MemorySearchResult(
        path="memory/2024-01-15.md",
        start_line=10,
        end_line=25,
        score=0.85,
        snippet="some text",
        source=MemorySource.MEMORY,
    )
    assert r.citation == "memory/2024-01-15.md#L10-L25"


def test_citation_single_line() -> None:
    r = MemorySearchResult(
        path="notes.md",
        start_line=1,
        end_line=1,
        score=0.5,
        snippet="x",
        source=MemorySource.SESSIONS,
    )
    assert r.citation == "notes.md#L1-L1"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_types_enums.py -v`
Expected: FAIL — ImportError (no `CitationMode`, `PromptMode`)

**Step 3: Add to types.py**

Add after the `MemorySource` enum:

```python
class CitationMode(str, Enum):
    """How to display source citations in search results."""

    AUTO = "auto"
    ON = "on"
    OFF = "off"


class PromptMode(str, Enum):
    """Controls how much system prompt content is included."""

    FULL = "full"
    MINIMAL = "minimal"
    NONE = "none"
```

Add a `citation` property to `MemorySearchResult`:

```python
class MemorySearchResult(BaseModel):
    """A single search hit from the memory store."""

    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    source: MemorySource

    @property
    def citation(self) -> str:
        """Format as path#Lstart-Lend."""
        return f"{self.path}#L{self.start_line}-L{self.end_line}"
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_types_enums.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/agent_memory/memory/types.py tests/test_types_enums.py
git commit -m "feat: add CitationMode, PromptMode enums and citation property"
```

---

### Task 10: Bootstrap File Truncation

**Files:**
- Modify: `src/agent_memory/personality/workspace.py`
- Create: `tests/test_bootstrap_truncation.py`

**Step 1: Write the failing test**

```python
# tests/test_bootstrap_truncation.py
"""Tests for bootstrap file truncation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent_memory.personality.workspace import load_bootstrap_files


def test_truncation_applied(tmp_path: Path) -> None:
    """Files exceeding max_chars_per_file get truncated."""
    big = tmp_path / "AGENTS.md"
    big.write_text("x" * 50_000, encoding="utf-8")
    files = load_bootstrap_files(tmp_path, max_chars_per_file=1000)
    agents = [f for f in files if f.name == "AGENTS.md"][0]
    assert len(agents.content) <= 1000 + 50  # allow for suffix
    assert "[... truncated at 1000 chars]" in agents.content


def test_no_truncation_when_under_limit(tmp_path: Path) -> None:
    """Small files are not truncated."""
    small = tmp_path / "AGENTS.md"
    small.write_text("hello", encoding="utf-8")
    files = load_bootstrap_files(tmp_path, max_chars_per_file=20000)
    agents = [f for f in files if f.name == "AGENTS.md"][0]
    assert agents.content == "hello"
    assert "truncated" not in agents.content


def test_default_limit_is_20k(tmp_path: Path) -> None:
    """Default limit is 20000 chars."""
    big = tmp_path / "AGENTS.md"
    big.write_text("y" * 25_000, encoding="utf-8")
    files = load_bootstrap_files(tmp_path)
    agents = [f for f in files if f.name == "AGENTS.md"][0]
    assert "[... truncated at 20000 chars]" in agents.content
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bootstrap_truncation.py -v`
Expected: FAIL — TypeError (unexpected keyword `max_chars_per_file`)

**Step 3: Update workspace.py**

Change `load_bootstrap_files` signature to add the parameter:

```python
def load_bootstrap_files(
    workspace_dir: str | Path,
    *,
    is_subagent: bool = False,
    is_main_session: bool = True,
    max_chars_per_file: int = 20_000,
) -> list[BootstrapFile]:
```

Inside the loop where files are read, add truncation after reading content:

```python
        if p.is_file():
            content = p.read_text(encoding="utf-8")
            if len(content) > max_chars_per_file:
                content = content[:max_chars_per_file] + f"\n\n[... truncated at {max_chars_per_file} chars]"
            result.append(BootstrapFile(name=name, path=str(p), content=content))
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bootstrap_truncation.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/agent_memory/personality/workspace.py tests/test_bootstrap_truncation.py
git commit -m "feat: add per-file truncation to bootstrap file loading"
```

---

### Task 11: HeartbeatScheduler

**Files:**
- Create: `src/agent_memory/heartbeat.py`
- Create: `tests/test_heartbeat.py`

**Step 1: Write the failing test**

```python
# tests/test_heartbeat.py
"""Tests for HeartbeatScheduler."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from agent_memory.config import HeartbeatConfig
from agent_memory.heartbeat import HeartbeatScheduler


def test_no_tasks_due_when_disabled() -> None:
    cfg = HeartbeatConfig(enabled=False)
    sched = HeartbeatScheduler(cfg, workspace_dir="/fake")
    assert sched.tasks_due() == []


def test_tasks_due_on_first_call(tmp_path: Path) -> None:
    hb = tmp_path / "HEARTBEAT.md"
    hb.write_text("Check email\nReview calendar\n", encoding="utf-8")
    cfg = HeartbeatConfig(enabled=True, interval_minutes=15)
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    due = sched.tasks_due()
    assert "Check email" in due
    assert "Review calendar" in due


def test_tasks_not_due_again_immediately(tmp_path: Path) -> None:
    hb = tmp_path / "HEARTBEAT.md"
    hb.write_text("Check email\n", encoding="utf-8")
    cfg = HeartbeatConfig(enabled=True, interval_minutes=15)
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    due = sched.tasks_due()
    assert len(due) == 1
    sched.mark_completed("Check email")
    assert sched.tasks_due() == []


def test_tasks_due_after_interval(tmp_path: Path) -> None:
    hb = tmp_path / "HEARTBEAT.md"
    hb.write_text("Check email\n", encoding="utf-8")
    cfg = HeartbeatConfig(enabled=True, interval_minutes=1)
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    sched.tasks_due()
    sched.mark_completed("Check email")
    # Fast-forward the last_run timestamp
    sched._last_run["Check email"] = datetime(2000, 1, 1)
    due = sched.tasks_due()
    assert "Check email" in due


def test_quiet_hours_respected(tmp_path: Path) -> None:
    hb = tmp_path / "HEARTBEAT.md"
    hb.write_text("Check email\n", encoding="utf-8")
    cfg = HeartbeatConfig(
        enabled=True, interval_minutes=1,
        quiet_start="22:00", quiet_end="07:00",
    )
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))

    # Mock current time to 23:00 (in quiet hours)
    mock_now = datetime(2026, 2, 14, 23, 0, 0)
    with patch("agent_memory.heartbeat._now", return_value=mock_now):
        assert sched.is_quiet_hours() is True
        assert sched.tasks_due() == []


def test_outside_quiet_hours(tmp_path: Path) -> None:
    cfg = HeartbeatConfig(
        enabled=True, quiet_start="22:00", quiet_end="07:00",
    )
    sched = HeartbeatScheduler(cfg, workspace_dir="/fake")
    mock_now = datetime(2026, 2, 14, 12, 0, 0)
    with patch("agent_memory.heartbeat._now", return_value=mock_now):
        assert sched.is_quiet_hours() is False


def test_parse_skips_comments_and_blanks(tmp_path: Path) -> None:
    hb = tmp_path / "HEARTBEAT.md"
    hb.write_text(
        "# HEARTBEAT.md\n\n"
        "# This is a comment\n"
        "Check email\n"
        "\n"
        "  \n"
        "Review calendar\n",
        encoding="utf-8",
    )
    cfg = HeartbeatConfig(enabled=True)
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    tasks = sched.parse_heartbeat_file()
    assert tasks == ["Check email", "Review calendar"]


def test_no_heartbeat_file(tmp_path: Path) -> None:
    cfg = HeartbeatConfig(enabled=True)
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    assert sched.parse_heartbeat_file() == []
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_heartbeat.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write the implementation**

```python
# src/agent_memory/heartbeat.py
"""Heartbeat scheduler — periodic task execution from HEARTBEAT.md."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from .config import HeartbeatConfig


def _now() -> datetime:
    """Current UTC time. Patchable for tests."""
    return datetime.utcnow()


class HeartbeatScheduler:
    """Caller-polled scheduler for periodic tasks.

    Does NOT run a background thread. The caller checks ``tasks_due()``
    on their own cadence (e.g., between LLM calls).
    """

    def __init__(
        self,
        config: HeartbeatConfig,
        workspace_dir: str,
        user_timezone: str = "UTC",
    ) -> None:
        self._config = config
        self._workspace_dir = Path(workspace_dir)
        self._user_timezone = user_timezone
        self._last_run: dict[str, datetime] = {}

    def is_quiet_hours(self) -> bool:
        """Check if current time falls within quiet hours window."""
        start = self._config.quiet_start
        end = self._config.quiet_end
        if not start or not end:
            return False

        now = _now()
        now_minutes = now.hour * 60 + now.minute
        start_h, start_m = (int(x) for x in start.split(":"))
        end_h, end_m = (int(x) for x in end.split(":"))
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m

        if start_minutes <= end_minutes:
            # Same-day window (e.g., 09:00-17:00)
            return start_minutes <= now_minutes < end_minutes
        else:
            # Overnight window (e.g., 22:00-07:00)
            return now_minutes >= start_minutes or now_minutes < end_minutes

    def tasks_due(self) -> list[str]:
        """Return tasks that are due for execution."""
        if not self._config.enabled:
            return []
        if self.is_quiet_hours():
            return []

        all_tasks = self.parse_heartbeat_file()
        if not all_tasks:
            return []

        now = _now()
        interval = timedelta(minutes=self._config.interval_minutes)
        due: list[str] = []
        for task in all_tasks:
            last = self._last_run.get(task)
            if last is None or (now - last) >= interval:
                due.append(task)
        return due

    def mark_completed(self, task: str) -> None:
        """Record that a task was just executed."""
        self._last_run[task] = _now()

    def parse_heartbeat_file(self) -> list[str]:
        """Parse HEARTBEAT.md into task strings.

        Skips blank lines, lines starting with ``#``, and whitespace-only lines.
        """
        path = self._workspace_dir / "HEARTBEAT.md"
        if not path.is_file():
            return []
        tasks: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tasks.append(stripped)
        return tasks
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_heartbeat.py -v`
Expected: 9 passed

**Step 5: Commit**

```bash
git add src/agent_memory/heartbeat.py tests/test_heartbeat.py
git commit -m "feat: add HeartbeatScheduler with quiet hours and interval tracking"
```

---

### Task 12: Update CONFIG.yaml + pyproject.toml

**Files:**
- Modify: `src/agent_memory/CONFIG.yaml`
- Modify: `pyproject.toml`

**Step 1: Update CONFIG.yaml**

Replace the full file with the expanded schema. All new fields have backward-compatible defaults matching existing behavior:

```yaml
# pyclawmem configuration

embedding:
  provider: bedrock                          # bedrock | openrouter | gguf
  # fallback: openrouter                     # optional: try this if primary fails

  # Bedrock-specific (used when provider or fallback is "bedrock")
  model_id: amazon.titan-embed-text-v2:0
  dimensions: 1024
  region: us-east-1

  # OpenRouter-specific (used when provider or fallback is "openrouter")
  # openrouter:
  #   api_key: ${OPENROUTER_API_KEY}
  #   model: thenlper/gte-large
  #   dimensions: 1024

  # GGUF-specific (used when provider or fallback is "gguf")
  # gguf:
  #   model_path: /path/to/model.gguf

llm:
  provider: bedrock
  model_id: us.anthropic.claude-haiku-4-5-20251001-v1:0
  region: us-east-1
  max_tokens: 1024

memory:
  db_path: ~/.pyclawmem/memory.sqlite
  facts_db_path: ~/.pyclawmem/facts.sqlite
  memory_dir: ~/.pyclawmem/workspace/memory
  chunk_tokens: 400
  chunk_overlap: 80
  hybrid_vector_weight: 0.7
  hybrid_text_weight: 0.3
  max_results: 6
  min_score: 0.35
  citations: auto                            # auto | on | off

context:
  max_context_tokens: 100000
  compaction_threshold: 0.85
  summary_preserve: "decisions, TODOs, open questions, constraints"
  session_dir: ~/.pyclawmem/sessions
  prompt_mode: full                          # full | minimal | none
  bootstrap_max_chars: 20000                 # per-file truncation limit

workspace:
  dir: ~/.pyclawmem/workspace

agent:
  id: default
  # name: null

  sandbox:
    mode: "off"                              # off | non-main | all
    workspace_access: rw                     # none | ro | rw

  group_chat:
    enabled: false
    respond_to_mentions: true
    respond_to_direct: true
    quiet_unless_mentioned: false

  heartbeat:
    enabled: false
    interval_minutes: 15
    # quiet_hours:
    #   start: "22:00"
    #   end: "07:00"
    tasks: []
```

**Step 2: Update pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "pyclawmem"
version = "0.1.0"
description = "Memory, personality, and context continuity for LLM agents"
requires-python = ">=3.11"
dependencies = [
    "boto3>=1.34",
    "httpx>=0.27",
    "numpy>=1.26",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
local = ["llama-cpp-python>=0.3"]
```

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add src/agent_memory/CONFIG.yaml pyproject.toml
git commit -m "feat: expand config schema with providers, citations, prompt modes, agent config"
```

---

### Task 13: Full Facade Wiring

**Files:**
- Modify: `src/agent_memory/facade.py`
- Create: `tests/test_facade_features.py`

This is the largest task — it wires prompt modes, citations, sandbox, group chat, and heartbeat into the facade.

**Step 1: Write the failing tests**

```python
# tests/test_facade_features.py
"""Tests for new facade features: prompt modes, citations, sandbox, group chat."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_memory.config import AgentConfig, GroupChatConfig, SandboxConfig
from agent_memory.memory.types import CitationMode, MemorySearchResult, MemorySource, PromptMode


# -- Prompt mode tests --


def test_prompt_mode_none_produces_minimal_prompt() -> None:
    from agent_memory.personality.workspace import build_context_prompt
    from agent_memory.memory.types import BootstrapFile

    # PromptMode.NONE should be handled by the facade, not build_context_prompt
    # The facade returns a one-liner instead of calling build_context_prompt
    assert PromptMode.NONE.value == "none"


# -- Citation tests --


def test_citation_on_shows_citation() -> None:
    r = MemorySearchResult(
        path="test.md", start_line=1, end_line=5,
        score=0.9, snippet="hello", source=MemorySource.MEMORY,
    )
    assert r.citation == "test.md#L1-L5"


def test_citation_mode_auto_resolves() -> None:
    assert CitationMode.AUTO == "auto"


# -- Sandbox tests --


def test_sandbox_blocks_writes_in_ro_mode() -> None:
    cfg = SandboxConfig(mode="all", workspace_access="ro")
    assert cfg.workspace_access == "ro"


def test_sandbox_blocks_all_in_none_mode() -> None:
    cfg = SandboxConfig(mode="all", workspace_access="none")
    assert cfg.workspace_access == "none"


# -- Group chat tests --


def test_should_respond_always_in_non_group() -> None:
    from agent_memory.facade import _should_respond
    gc = GroupChatConfig(enabled=False)
    assert _should_respond(gc, was_mentioned=False, is_direct=False) is True


def test_should_respond_quiet_unless_mentioned() -> None:
    from agent_memory.facade import _should_respond
    gc = GroupChatConfig(enabled=True, quiet_unless_mentioned=True)
    assert _should_respond(gc, was_mentioned=False, is_direct=False) is False
    assert _should_respond(gc, was_mentioned=True, is_direct=False) is True


def test_should_respond_direct_message() -> None:
    from agent_memory.facade import _should_respond
    gc = GroupChatConfig(enabled=True, respond_to_direct=True)
    assert _should_respond(gc, was_mentioned=False, is_direct=True) is True


def test_should_respond_mention() -> None:
    from agent_memory.facade import _should_respond
    gc = GroupChatConfig(enabled=True, respond_to_mentions=True)
    assert _should_respond(gc, was_mentioned=True, is_direct=False) is True
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_facade_features.py -v`
Expected: FAIL — ImportError (no `_should_respond`)

**Step 3: Update facade.py**

Major changes to `src/agent_memory/facade.py`:

1. Add imports:

```python
from .config import AgentConfig, GroupChatConfig, SandboxConfig
from .heartbeat import HeartbeatScheduler
from .memory.embeddings import resolve_embeddings
from .memory.types import CitationMode, PromptMode
```

2. Add module-level helper:

```python
def _should_respond(gc: GroupChatConfig, *, was_mentioned: bool, is_direct: bool) -> bool:
    """Check group chat config to decide if the agent should respond."""
    if not gc.enabled:
        return True
    if gc.quiet_unless_mentioned and not was_mentioned:
        return False
    if is_direct and gc.respond_to_direct:
        return True
    if was_mentioned and gc.respond_to_mentions:
        return True
    return not gc.quiet_unless_mentioned
```

3. Add new parameters to `AgentMemory.__init__`:

```python
    def __init__(
        self,
        *,
        memory: MemoryManager,
        facts: FactStore,
        sessions: SessionStore,
        workspace_dir: str,
        memory_dir: str,
        llm_model_id: str,
        llm_region: str,
        llm_max_tokens: int,
        max_context_tokens: int,
        compaction_threshold: float,
        summary_preserve: str,
        session_dir: str,
        # Soul-evil config
        soul_evil_chance: float = 0.0,
        soul_evil_purge_at: str | None = None,
        soul_evil_purge_duration: str | None = None,
        soul_evil_filename: str = "SOUL_EVIL.md",
        user_timezone: str = "UTC",
        # Presence config
        human_delay: HumanDelay | None = None,
        typing_mode: TypingMode | None = None,
        # NEW: agent config
        agent_config: AgentConfig | None = None,
        # NEW: context features
        prompt_mode: PromptMode = PromptMode.FULL,
        citation_mode: CitationMode = CitationMode.AUTO,
        bootstrap_max_chars: int = 20_000,
    ) -> None:
```

Store the new fields:

```python
        self._agent_config = agent_config or AgentConfig()
        self._prompt_mode = prompt_mode
        self._citation_mode = citation_mode
        self._bootstrap_max_chars = bootstrap_max_chars

        # Heartbeat
        if self._agent_config.heartbeat.enabled:
            self._heartbeat = HeartbeatScheduler(
                self._agent_config.heartbeat,
                workspace_dir=workspace_dir,
                user_timezone=user_timezone,
            )
        else:
            self._heartbeat = None
```

4. Update `start_session` for prompt modes and sandbox:

```python
    def start_session(
        self,
        *,
        is_main_session: bool = True,
        is_subagent: bool = False,
    ) -> Session:
        # Sandbox: decide if this session is sandboxed
        sandboxed = self._is_sandboxed(is_main_session)

        if not sandboxed:
            ensure_workspace(self._workspace_dir)

        if self._guard.should_block:
            raise RuntimeError(
                f"Context window {self._guard.tokens} tokens is below hard minimum."
            )

        # Resolve effective prompt mode
        effective_mode = self._prompt_mode
        if is_subagent:
            effective_mode = PromptMode.MINIMAL

        if effective_mode == PromptMode.NONE:
            name = self._assistant.name or "Assistant"
            system_prompt = f"You are {name}."
        else:
            files = load_bootstrap_files(
                self._workspace_dir,
                is_main_session=is_main_session,
                is_subagent=(effective_mode == PromptMode.MINIMAL),
                max_chars_per_file=self._bootstrap_max_chars,
            )

            soul = load_soul(self._workspace_dir)
            decision = decide_soul_evil(
                chance=self._soul_evil_chance,
                purge_at=self._soul_evil_purge_at,
                purge_duration=self._soul_evil_purge_duration,
                evil_filename=self._soul_evil_filename,
                user_timezone=self._user_timezone,
            )
            soul = apply_soul_evil_override(soul, self._workspace_dir, decision=decision)
            system_prompt = build_context_prompt(files)

            if effective_mode == PromptMode.FULL:
                daily_context = _load_recent_daily_logs(self._memory_dir)
                if daily_context:
                    system_prompt += f"\n\n# Recent Daily Logs\n\n{daily_context}"

        self._needs_sync = True
        entry = self._sessions.create_session()
        messages = [
            SessionMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
                metadata={"type": "bootstrap_load"},
            )
        ]
        return Session(
            entry=entry, messages=messages, system_prompt=system_prompt,
            _sandboxed=sandboxed,
        )
```

5. Add `_sandboxed` field to `Session`:

```python
@dataclass
class Session:
    entry: SessionEntry
    messages: list[SessionMessage] = field(default_factory=list)
    system_prompt: str = ""
    compaction_count: int = 0
    flush_at_compaction: int | None = None
    _sandboxed: bool = False
```

6. Update `end_session` for sandbox:

```python
    def end_session(self, session: Session) -> None:
        if session._sandboxed:
            return
        auto_capture(session.messages, self._facts)
        export_session_to_markdown(
            session.messages,
            memory_dir=self._memory_dir,
            session_id=session.entry.session_id,
            llm_model_id=self._llm_model_id,
            llm_region=self._llm_region,
        )
```

7. Update `before_llm_call` for sandbox (skip memory flush):

In the memory flush section, wrap with:

```python
        if not session._sandboxed and should_run_memory_flush(...):
```

8. Update `execute_tool` for sandbox workspace access:

At the top of `execute_tool`:

```python
    def execute_tool(self, name: str, input_data: dict, *, session: Session | None = None) -> str:
        sandbox = self._agent_config.sandbox
        sandboxed = session._sandboxed if session else False

        if name == "workspace_write" and sandboxed and sandbox.workspace_access != "rw":
            return "Blocked: workspace writes are disabled in sandbox mode."
        if name in ("workspace_read", "workspace_write") and sandboxed and sandbox.workspace_access == "none":
            return "Blocked: workspace access is disabled in sandbox mode."
```

9. Update `execute_tool("memory_search")` for citations:

```python
        if name == "memory_search":
            results = self.search_memory(
                input_data["query"],
                max_results=input_data.get("max_results", 6),
            )
            if not results:
                return "No results found."
            show_citations = self._should_show_citations(session)
            lines = []
            for r in results:
                snippet = r.snippet[:300].replace("\n", " ")
                prefix = f"[{r.source.value}:{r.path}]"
                suffix = f" Source: {r.citation}" if show_citations else ""
                lines.append(f"{prefix} (score: {r.score:.2f}) {snippet}{suffix}")
            return "\n".join(lines)
```

10. Add citation helper:

```python
    def _should_show_citations(self, session: Session | None = None) -> bool:
        if self._citation_mode == CitationMode.ON:
            return True
        if self._citation_mode == CitationMode.OFF:
            return False
        # AUTO: show in main sessions, hide in group
        if self._agent_config.group_chat.enabled:
            return False
        return True
```

11. Add sandbox helper:

```python
    def _is_sandboxed(self, is_main_session: bool) -> bool:
        mode = self._agent_config.sandbox.mode
        if mode == "all":
            return True
        if mode == "non-main" and not is_main_session:
            return True
        return False
```

12. Add group chat + heartbeat properties:

```python
    def should_respond(
        self,
        *,
        was_mentioned: bool = False,
        is_direct: bool = False,
    ) -> bool:
        return _should_respond(
            self._agent_config.group_chat,
            was_mentioned=was_mentioned,
            is_direct=is_direct,
        )

    @property
    def heartbeat(self) -> HeartbeatScheduler | None:
        return self._heartbeat

    @property
    def agent_config(self) -> AgentConfig:
        return self._agent_config
```

13. Update `get_tool_definitions` for sandbox filtering:

```python
    def get_tool_definitions(self, *, session: Session | None = None) -> list[dict]:
        tools = [...]  # existing list
        if session and session._sandboxed:
            access = self._agent_config.sandbox.workspace_access
            if access == "none":
                tools = [t for t in tools if t["toolSpec"]["name"] not in ("workspace_read", "workspace_write")]
            elif access == "ro":
                tools = [t for t in tools if t["toolSpec"]["name"] != "workspace_write"]
        return tools
```

14. Update `from_config` to wire everything:

```python
    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "AgentMemory":
        from . import load_config

        cfg = load_config(Path(config_path) if config_path else None)
        emb_cfg = cfg["embedding"]
        mem_cfg = cfg["memory"]
        ctx_cfg = cfg["context"]
        llm_cfg = cfg["llm"]
        ws_dir = cfg["workspace"]["dir"]

        embeddings = resolve_embeddings(emb_cfg)
        store = MemoryStore(mem_cfg["db_path"])
        memory = MemoryManager(
            store, embeddings,
            chunk_tokens=mem_cfg["chunk_tokens"],
            chunk_overlap=mem_cfg["chunk_overlap"],
            vector_weight=mem_cfg["hybrid_vector_weight"],
            text_weight=mem_cfg["hybrid_text_weight"],
            max_results=mem_cfg["max_results"],
            min_score=mem_cfg["min_score"],
        )
        facts = FactStore(
            db_path=mem_cfg["facts_db_path"],
            embeddings=embeddings,
        )
        sessions = SessionStore(ctx_cfg["session_dir"])
        memory_dir = mem_cfg["memory_dir"]
        agent_cfg = AgentConfig.from_dict(cfg.get("agent", {}))

        return cls(
            memory=memory,
            facts=facts,
            sessions=sessions,
            workspace_dir=ws_dir,
            memory_dir=memory_dir,
            llm_model_id=llm_cfg["model_id"],
            llm_region=llm_cfg.get("region", "us-east-1"),
            llm_max_tokens=llm_cfg.get("max_tokens", 1024),
            max_context_tokens=ctx_cfg.get("max_context_tokens", 100_000),
            compaction_threshold=ctx_cfg.get("compaction_threshold", 0.85),
            summary_preserve=ctx_cfg.get("summary_preserve", "decisions, TODOs, open questions"),
            session_dir=ctx_cfg["session_dir"],
            agent_config=agent_cfg,
            prompt_mode=PromptMode(ctx_cfg.get("prompt_mode", "full")),
            citation_mode=CitationMode(mem_cfg.get("citations", "auto")),
            bootstrap_max_chars=ctx_cfg.get("bootstrap_max_chars", 20_000),
        )
```

**Step 4: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/agent_memory/facade.py tests/test_facade_features.py
git commit -m "feat: wire prompt modes, citations, sandbox, group chat, heartbeat into facade"
```

---

### Task 14: Update integration.py + Public API

**Files:**
- Modify: `src/agent_memory/integration.py`
- Modify: `src/agent_memory/__init__.py`

**Step 1: Update integration.py imports**

Change the `BedrockEmbeddings` import to use `resolve_embeddings`:

```python
from pyclawmem.memory.embeddings import resolve_embeddings
```

Update `create_agent_context`:

```python
def create_agent_context(config_path: str | None = None) -> dict:
    cfg = load_config()
    emb_cfg = cfg["embedding"]
    mem_cfg = cfg["memory"]
    ctx_cfg = cfg["context"]
    ws_dir = cfg["workspace"]["dir"]

    embeddings = resolve_embeddings(emb_cfg)
    # ... rest unchanged
```

**Step 2: Update __init__.py exports**

```python
from .config import AgentConfig, GroupChatConfig, HeartbeatConfig, SandboxConfig
from .heartbeat import HeartbeatScheduler
from .memory.embeddings import EmbeddingProvider, FallbackChain, resolve_embeddings
from .memory.types import CitationMode, PromptMode

__all__ = [
    "AgentMemory", "Session", "PreparedCall", "load_config",
    "EmbeddingProvider", "FallbackChain", "resolve_embeddings",
    "AgentConfig", "SandboxConfig", "GroupChatConfig", "HeartbeatConfig",
    "HeartbeatScheduler",
    "CitationMode", "PromptMode",
]
```

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add src/agent_memory/integration.py src/agent_memory/__init__.py
git commit -m "feat: update integration.py and public API exports for new features"
```
