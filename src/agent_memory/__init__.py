"""pyclawmem — Memory, personality, and context continuity for LLM agents."""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "CONFIG.yaml"


def load_config(path: Path | None = None) -> dict:
    """Load and return the CONFIG.yaml as a dict."""
    p = path or _CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


# Public API
from .config import AgentConfig, GroupChatConfig, HeartbeatConfig, SandboxConfig  # noqa: E402
from .facade import AgentMemory, PreparedCall, Session  # noqa: E402
from .heartbeat import HeartbeatScheduler  # noqa: E402
from .memory.embeddings import EmbeddingProvider, FallbackChain, resolve_embeddings  # noqa: E402
from .memory.types import CitationMode, PromptMode  # noqa: E402

__all__ = [
    "AgentMemory",
    "Session",
    "PreparedCall",
    "load_config",
    "EmbeddingProvider",
    "FallbackChain",
    "resolve_embeddings",
    "AgentConfig",
    "SandboxConfig",
    "GroupChatConfig",
    "HeartbeatConfig",
    "HeartbeatScheduler",
    "CitationMode",
    "PromptMode",
]
