"""Tests for CitationMode, PromptMode, and citation property."""

from __future__ import annotations

from agent_memory.memory.types import (
    CitationMode,
    MemorySearchResult,
    MemorySource,
    PromptMode,
)


def test_citation_mode_values() -> None:
    assert CitationMode.AUTO == "auto"
    assert CitationMode.ON == "on"
    assert CitationMode.OFF == "off"


def test_prompt_mode_values() -> None:
    assert PromptMode.FULL == "full"
    assert PromptMode.MINIMAL == "minimal"
    assert PromptMode.NONE == "none"


def test_citation_property() -> None:
    r = MemorySearchResult(
        path="memory/2024-01-15.md",
        start_line=10,
        end_line=25,
        score=0.85,
        snippet="some text",
        source=MemorySource.MEMORY,
    )
    assert r.citation == "memory/2024-01-15.md#L10-L25"


def test_citation_single_line() -> None:
    r = MemorySearchResult(
        path="notes.md",
        start_line=1,
        end_line=1,
        score=0.5,
        snippet="x",
        source=MemorySource.SESSIONS,
    )
    assert r.citation == "notes.md#L1-L1"
