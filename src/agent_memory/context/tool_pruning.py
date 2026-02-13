"""Tool result pruning — trim large tool outputs to save context tokens."""

from __future__ import annotations

from ..memory.types import SessionMessage

DEFAULT_MAX_CHARS = 4000
DEFAULT_HEAD_CHARS = 1500
DEFAULT_TAIL_CHARS = 1500
DEFAULT_KEEP_LAST_ASSISTANTS = 3
DEFAULT_HARD_CLEAR_PLACEHOLDER = "[Old tool result content cleared]"


def prune_tool_results(
    messages: list[SessionMessage],
    *,
    keep_last_assistants: int = DEFAULT_KEEP_LAST_ASSISTANTS,
    soft_trim_max_chars: int = DEFAULT_MAX_CHARS,
    soft_trim_head: int = DEFAULT_HEAD_CHARS,
    soft_trim_tail: int = DEFAULT_TAIL_CHARS,
    hard_clear: bool = True,
    hard_clear_placeholder: str = DEFAULT_HARD_CLEAR_PLACEHOLDER,
    cache_ttl_ms: int | None = None,
    last_cache_touch_ms: int | None = None,
) -> list[SessionMessage]:
    """Prune tool result content to reduce context size.

    - Soft trim: large tool results → keep head + tail, truncate middle
    - Hard clear: old tool results → replace with placeholder
    - Protects the most recent `keep_last_assistants` assistant turns
    - Cache-TTL mode: only prune if cache has expired
    """
    # Cache-TTL gate: skip pruning if cache is still warm
    if cache_ttl_ms is not None and last_cache_touch_ms is not None:
        import time
        now_ms = int(time.time() * 1000)
        if (now_ms - last_cache_touch_ms) < cache_ttl_ms:
            return messages

    # Find indices of assistant messages (to identify protected zone)
    assistant_indices = [i for i, m in enumerate(messages) if m.role.value == "assistant"]
    protected_from = assistant_indices[-keep_last_assistants] if len(assistant_indices) >= keep_last_assistants else 0

    result: list[SessionMessage] = []
    for i, msg in enumerate(messages):
        if not _is_tool_result(msg) or i >= protected_from:
            result.append(msg)
            continue

        content = msg.content
        if len(content) <= soft_trim_max_chars:
            result.append(msg)
            continue

        # Old tool result (before protected zone) + large → hard clear
        if hard_clear and i < protected_from - keep_last_assistants:
            result.append(msg.model_copy(update={"content": hard_clear_placeholder}))
            continue

        # Soft trim: keep head + tail
        trimmed = _soft_trim(content, soft_trim_head, soft_trim_tail)
        result.append(msg.model_copy(update={"content": trimmed}))

    return result


def _is_tool_result(msg: SessionMessage) -> bool:
    return msg.metadata.get("type") == "tool_result" or msg.role.value == "tool"


def _soft_trim(content: str, head: int, tail: int) -> str:
    if len(content) <= head + tail:
        return content
    omitted = len(content) - head - tail
    return f"{content[:head]}\n\n[...{omitted} chars truncated...]\n\n{content[-tail:]}"
