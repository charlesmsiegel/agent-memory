"""Pre-compaction memory flush — prompt the agent to persist context before truncation."""

from __future__ import annotations

_SILENT_TOKEN = "NO_REPLY"

DEFAULT_SOFT_THRESHOLD_TOKENS = 4000
DEFAULT_RESERVE_FLOOR_TOKENS = 20000

DEFAULT_FLUSH_PROMPT = (
    "Pre-compaction memory flush. "
    "Store durable memories now (use memory/YYYY-MM-DD.md; create memory/ if needed). "
    f"If nothing to store, reply with {_SILENT_TOKEN}."
)

DEFAULT_FLUSH_SYSTEM_PROMPT = (
    "Pre-compaction memory flush turn. "
    "The session is near auto-compaction; capture durable memories to disk. "
    f"You may reply, but usually {_SILENT_TOKEN} is correct."
)


def should_run_memory_flush(
    *,
    total_tokens: int,
    context_window_tokens: int,
    reserve_floor: int = DEFAULT_RESERVE_FLOOR_TOKENS,
    soft_threshold: int = DEFAULT_SOFT_THRESHOLD_TOKENS,
    compaction_count: int = 0,
    last_flush_at_compaction: int | None = None,
) -> bool:
    """Check if a memory flush should fire before compaction.

    Triggers when: totalTokens >= contextWindow - reserveFloor - softThreshold
    Only fires once per compaction cycle.
    """
    if total_tokens <= 0:
        return False
    threshold = max(0, context_window_tokens - reserve_floor - soft_threshold)
    if threshold <= 0 or total_tokens < threshold:
        return False
    # One flush per compaction cycle
    if last_flush_at_compaction is not None and last_flush_at_compaction == compaction_count:
        return False
    return True


SILENT_TOKEN = _SILENT_TOKEN
