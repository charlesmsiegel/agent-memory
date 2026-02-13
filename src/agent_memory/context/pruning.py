"""Context pruning — remove stale context to free tokens."""

from __future__ import annotations

from ..memory.types import SessionMessage

# Metadata types that are prunable after they've served their purpose
_PRUNABLE_TYPES = {"identity_read", "bootstrap_load", "memory_inject"}


def prune_stale_context(
    messages: list[SessionMessage],
    *,
    keep_recent: int = 4,
) -> list[SessionMessage]:
    """Remove stale context messages (identity reads, old memory injections).

    Keeps the first message (system prompt) and the most recent `keep_recent`
    messages unconditionally. Everything in between is checked for prunability.
    """
    if len(messages) <= keep_recent + 1:
        return messages

    head = messages[:1]
    tail = messages[-keep_recent:]
    middle = messages[1:-keep_recent]

    pruned_middle = [m for m in middle if not _is_prunable(m)]
    return head + pruned_middle + tail


def _is_prunable(message: SessionMessage) -> bool:
    """A message is prunable if its metadata marks it as a one-time context injection."""
    return message.metadata.get("type") in _PRUNABLE_TYPES


def estimate_pruning_savings(messages: list[SessionMessage], keep_recent: int = 4) -> int:
    """Estimate how many tokens pruning would save (rough chars/4)."""
    if len(messages) <= keep_recent + 1:
        return 0
    middle = messages[1:-keep_recent]
    prunable = [m for m in middle if _is_prunable(m)]
    return sum(len(m.content) // 4 + 1 for m in prunable)
