"""Auto-capture — extract memorable facts from conversations."""

from __future__ import annotations

import re

from .facts import FactCategory, FactStore
from .types import SessionMessage

_MAX_CAPTURES_PER_CONVERSATION = 3

_MEMORY_TRIGGERS = [
    re.compile(r"remember", re.IGNORECASE),
    re.compile(r"prefer|rather|don't want", re.IGNORECASE),
    re.compile(r"decided|will use|going with", re.IGNORECASE),
    re.compile(r"\+\d{10,}"),
    re.compile(r"[\w.-]+@[\w.-]+\.\w+"),
    re.compile(r"my\s+\w+\s+is|is\s+my", re.IGNORECASE),
    re.compile(r"i (like|prefer|hate|love|want|need)", re.IGNORECASE),
    re.compile(r"always|never|important", re.IGNORECASE),
]


def should_capture(text: str) -> bool:
    """Check if a message text contains triggers worth capturing."""
    if len(text) < 10 or len(text) > 500:
        return False
    if "<relevant-memories>" in text:
        return False
    if text.startswith("<") and "</" in text:
        return False
    if "**" in text and "\n-" in text:
        return False
    emoji_count = len(re.findall(r"[\U0001F300-\U0001F9FF]", text))
    if emoji_count > 3:
        return False
    return any(r.search(text) for r in _MEMORY_TRIGGERS)


def detect_category(text: str) -> FactCategory:
    """Heuristic category detection from text content."""
    lower = text.lower()
    if re.search(r"prefer|rather|like|love|hate|want", lower):
        return FactCategory.PREFERENCE
    if re.search(r"decided|will use|going with", lower):
        return FactCategory.DECISION
    if re.search(r"\+\d{10,}|@[\w.-]+\.\w+|is called|named", lower):
        return FactCategory.ENTITY
    if re.search(r"\bis\b|\bare\b|\bhas\b|\bhave\b", lower):
        return FactCategory.FACT
    return FactCategory.OTHER


def auto_capture(
    messages: list[SessionMessage],
    fact_store: FactStore,
) -> int:
    """Extract and store memorable facts from a conversation.

    Called on session end. Scans user and assistant messages for
    capturable content, deduplicates, and stores up to 3 new facts.
    Returns the number of facts stored.
    """
    texts: list[str] = []
    for m in messages:
        if m.role.value not in ("user", "assistant"):
            continue
        if m.content and should_capture(m.content):
            texts.append(m.content)

    stored = 0
    for text in texts[:_MAX_CAPTURES_PER_CONVERSATION * 2]:  # check more, store up to max
        if stored >= _MAX_CAPTURES_PER_CONVERSATION:
            break
        category = detect_category(text)
        result = fact_store.store(text, category=category, importance=0.7)
        if result is not None:
            stored += 1
    return stored
