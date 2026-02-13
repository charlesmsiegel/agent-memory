"""Human delay and typing mode — personality expression through timing."""

from __future__ import annotations

import random
import time
from enum import Enum
from typing import NamedTuple


class TypingMode(str, Enum):
    """When to show typing indicators."""

    INSTANT = "instant"    # Show immediately on receiving message (DMs)
    MESSAGE = "message"    # Show when agent starts generating text (groups)
    THINKING = "thinking"  # Show when reasoning/thinking starts
    NEVER = "never"        # Never show (heartbeats, background work)


class HumanDelay(NamedTuple):
    """Configurable artificial delay to make responses feel human-paced."""

    min_ms: int
    max_ms: int


DEFAULT_HUMAN_DELAY = HumanDelay(min_ms=500, max_ms=2000)


def resolve_typing_mode(
    *,
    configured: TypingMode | None = None,
    is_group_chat: bool = False,
    was_mentioned: bool = False,
    is_heartbeat: bool = False,
) -> TypingMode:
    """Resolve the effective typing mode from context.

    Heartbeats → never (background work shouldn't show typing).
    Configured → use as-is.
    DMs or mentioned → instant.
    Groups → message (avoids typing indicator for messages agent won't reply to).
    """
    if is_heartbeat:
        return TypingMode.NEVER
    if configured:
        return configured
    if not is_group_chat or was_mentioned:
        return TypingMode.INSTANT
    return TypingMode.MESSAGE


def apply_human_delay(delay: HumanDelay | None = None) -> float:
    """Sleep for a random duration within the delay range. Returns ms slept."""
    d = delay or DEFAULT_HUMAN_DELAY
    ms = random.randint(d.min_ms, d.max_ms)
    time.sleep(ms / 1000.0)
    return float(ms)
