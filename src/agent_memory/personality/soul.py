"""SOUL.md loading, system prompt integration, and soul-evil decision logic."""

from __future__ import annotations

import random as _random_mod
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

_SOUL_GUIDANCE = (
    "If SOUL.md is present, embody its persona and tone. "
    "Avoid stiff, generic replies; follow its guidance "
    "unless higher-priority instructions override it."
)


def load_soul(workspace_dir: str | Path) -> str | None:
    """Read SOUL.md from the workspace. Returns content or None."""
    p = Path(workspace_dir) / "SOUL.md"
    if p.is_file():
        return p.read_text(encoding="utf-8").strip() or None
    return None


def build_soul_prompt_section(soul_content: str | None) -> str | None:
    """Build the system prompt section for SOUL.md, or None if absent."""
    if not soul_content:
        return None
    return f"{_SOUL_GUIDANCE}\n\n## SOUL.md\n\n{soul_content}"


# -- Soul Evil --


class SoulEvilDecision(NamedTuple):
    use_evil: bool
    reason: str | None  # "purge" | "chance" | None
    file_name: str


def decide_soul_evil(
    *,
    chance: float = 0.0,
    purge_at: str | None = None,
    purge_duration: str | None = None,
    evil_filename: str = "SOUL_EVIL.md",
    user_timezone: str = "UTC",
    now: datetime | None = None,
    random: float | None = None,
) -> SoulEvilDecision:
    """Decide whether to swap SOUL.md with the evil variant.

    Purge window takes precedence over random chance.
    """
    now = now or datetime.now(timezone.utc)

    # Check purge window first (higher precedence)
    if purge_at and purge_duration:
        if _is_within_purge_window(purge_at, purge_duration, now, user_timezone):
            return SoulEvilDecision(use_evil=True, reason="purge", file_name=evil_filename)

    # Check random chance
    clamped = max(0.0, min(1.0, chance))
    if clamped > 0:
        roll = random if random is not None else _random_mod.random()
        if roll < clamped:
            return SoulEvilDecision(use_evil=True, reason="chance", file_name=evil_filename)

    return SoulEvilDecision(use_evil=False, reason=None, file_name=evil_filename)


def apply_soul_evil_override(
    soul_content: str | None,
    workspace_dir: str | Path,
    *,
    decision: SoulEvilDecision,
) -> str | None:
    """If decision says use evil, replace soul content (in-memory only)."""
    if not decision.use_evil or not soul_content:
        return soul_content
    evil_path = Path(workspace_dir) / decision.file_name
    if evil_path.is_file():
        evil = evil_path.read_text(encoding="utf-8").strip()
        if evil:
            return evil
    return soul_content


# -- Purge window helpers --

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_DURATION_RE = re.compile(r"^(\d+)\s*(s|m|h)$", re.IGNORECASE)


def _parse_time_minutes(raw: str) -> int | None:
    m = _TIME_RE.match(raw.strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def _parse_duration_ms(raw: str) -> int | None:
    m = _DURATION_RE.match(raw.strip())
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "s":
        return value * 1000
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    return None


def _is_within_purge_window(
    at: str, duration: str, now: datetime, tz_name: str
) -> bool:
    start_minutes = _parse_time_minutes(at)
    duration_ms = _parse_duration_ms(duration)
    if start_minutes is None or duration_ms is None or duration_ms <= 0:
        return False

    day_ms = 24 * 60 * 60 * 1000
    if duration_ms >= day_ms:
        return True

    try:
        tz = ZoneInfo(tz_name)
    except (KeyError, Exception):
        tz = timezone.utc
    local = now.astimezone(tz)
    now_ms = (local.hour * 3600 + local.minute * 60 + local.second) * 1000 + local.microsecond // 1000
    start_ms = start_minutes * 60 * 1000
    end_ms = start_ms + duration_ms

    if end_ms < day_ms:
        return start_ms <= now_ms < end_ms
    # Wraps past midnight
    wrapped_end = end_ms % day_ms
    return now_ms >= start_ms or now_ms < wrapped_end
