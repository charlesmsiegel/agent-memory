"""Core data types for pyclawmem."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# -- Memory types --


class MemorySource(str, Enum):
    """Where a memory chunk originated."""

    MEMORY = "memory"
    SESSIONS = "sessions"


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


class MemoryChunk(BaseModel):
    """An indexed chunk of text stored in the vector + FTS tables."""

    id: int | None = None
    path: str
    start_line: int
    end_line: int
    text: str
    source: MemorySource
    embedding: list[float] | None = None


# -- Personality types --


class AgentIdentity(BaseModel):
    """Structured fields parsed from IDENTITY.md."""

    name: str | None = None
    emoji: str | None = None
    creature: str | None = None
    vibe: str | None = None
    theme: str | None = None
    avatar: str | None = None

    def has_values(self) -> bool:
        return any(
            getattr(self, f) is not None
            for f in ("name", "emoji", "creature", "vibe", "theme", "avatar")
        )


class BootstrapFile(BaseModel):
    """A workspace bootstrap file (SOUL.md, IDENTITY.md, etc.)."""

    name: str
    path: str
    content: str | None = None
    missing: bool = False


# -- Context / session types --


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class SessionMessage(BaseModel):
    """A single message in a session transcript."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionEntry(BaseModel):
    """Metadata for a session stored in the session index."""

    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    token_estimate: int = 0
    summary: str | None = None
    transcript_path: str | None = None


class CompactionResult(BaseModel):
    """Result of compacting a session's message history."""

    summary: str
    messages_removed: int
    tokens_before: int
    tokens_after: int
    tool_failures: list[str] = Field(default_factory=list)
    file_operations: list[str] = Field(default_factory=list)
