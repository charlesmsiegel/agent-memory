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
        vec = result[0] if isinstance(result[0], list) else result
        return normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts sequentially."""
        return [self.embed(t) for t in texts]
