"""Auto-recall — inject relevant memories before LLM calls."""

from __future__ import annotations

from .facts import FactStore
from .manager import MemoryManager
from .types import MessageRole, SessionMessage


def auto_recall(
    prompt: str,
    *,
    fact_store: FactStore | None = None,
    memory_manager: MemoryManager | None = None,
    max_facts: int = 3,
    max_file_results: int = 3,
    min_score: float = 0.3,
) -> SessionMessage | None:
    """Search both fact store and file memory, return an injection message.

    Returns a SessionMessage with role=system and <relevant-memories> content,
    or None if nothing relevant was found.
    """
    if not prompt or len(prompt) < 5:
        return None

    parts: list[str] = []

    # Search conversational facts
    if fact_store:
        results = fact_store.recall(prompt, limit=max_facts, min_score=min_score)
        for r in results:
            parts.append(f"- [{r.fact.category.value}] {r.fact.text}")

    # Search file-based memory
    if memory_manager:
        results = memory_manager.search(prompt, max_results=max_file_results, min_score=min_score)
        for r in results:
            snippet = r.snippet[:200].replace("\n", " ")
            parts.append(f"- [{r.source.value}:{r.path}] {snippet}")

    if not parts:
        return None

    content = (
        "<relevant-memories>\n"
        "The following memories may be relevant to this conversation:\n"
        + "\n".join(parts)
        + "\n</relevant-memories>"
    )
    return SessionMessage(
        role=MessageRole.SYSTEM,
        content=content,
        metadata={"type": "memory_inject"},
    )
