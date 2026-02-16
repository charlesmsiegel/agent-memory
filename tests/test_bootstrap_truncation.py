"""Tests for bootstrap file truncation."""

from __future__ import annotations

from pathlib import Path

from agent_memory.personality.workspace import load_bootstrap_files


def test_truncation_applied(tmp_path: Path) -> None:
    """Files exceeding max_chars_per_file get truncated."""
    big = tmp_path / "AGENTS.md"
    big.write_text("x" * 50_000, encoding="utf-8")
    files = load_bootstrap_files(tmp_path, max_chars_per_file=1000)
    agents = [f for f in files if f.name == "AGENTS.md"][0]
    assert len(agents.content) <= 1000 + 50  # allow for suffix
    assert "[... truncated at 1000 chars]" in agents.content


def test_no_truncation_when_under_limit(tmp_path: Path) -> None:
    """Small files are not truncated."""
    small = tmp_path / "AGENTS.md"
    small.write_text("hello", encoding="utf-8")
    files = load_bootstrap_files(tmp_path, max_chars_per_file=20000)
    agents = [f for f in files if f.name == "AGENTS.md"][0]
    assert agents.content == "hello"
    assert "truncated" not in agents.content


def test_default_limit_is_20k(tmp_path: Path) -> None:
    """Default limit is 20000 chars."""
    big = tmp_path / "AGENTS.md"
    big.write_text("y" * 25_000, encoding="utf-8")
    files = load_bootstrap_files(tmp_path)
    agents = [f for f in files if f.name == "AGENTS.md"][0]
    assert "[... truncated at 20000 chars]" in agents.content
