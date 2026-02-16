"""Tests for new facade features: prompt modes, citations, sandbox, group chat."""

from __future__ import annotations

from agent_memory.config import AgentConfig, GroupChatConfig, SandboxConfig
from agent_memory.memory.types import CitationMode, MemorySearchResult, MemorySource, PromptMode


def test_prompt_mode_none_value() -> None:
    assert PromptMode.NONE.value == "none"


def test_citation_on_shows_citation() -> None:
    r = MemorySearchResult(
        path="test.md", start_line=1, end_line=5,
        score=0.9, snippet="hello", source=MemorySource.MEMORY,
    )
    assert r.citation == "test.md#L1-L5"


def test_should_respond_always_in_non_group() -> None:
    from agent_memory.facade import _should_respond
    gc = GroupChatConfig(enabled=False)
    assert _should_respond(gc, was_mentioned=False, is_direct=False) is True


def test_should_respond_quiet_unless_mentioned() -> None:
    from agent_memory.facade import _should_respond
    gc = GroupChatConfig(enabled=True, quiet_unless_mentioned=True)
    assert _should_respond(gc, was_mentioned=False, is_direct=False) is False
    assert _should_respond(gc, was_mentioned=True, is_direct=False) is True


def test_should_respond_direct_message() -> None:
    from agent_memory.facade import _should_respond
    gc = GroupChatConfig(enabled=True, respond_to_direct=True)
    assert _should_respond(gc, was_mentioned=False, is_direct=True) is True


def test_should_respond_mention() -> None:
    from agent_memory.facade import _should_respond
    gc = GroupChatConfig(enabled=True, respond_to_mentions=True)
    assert _should_respond(gc, was_mentioned=True, is_direct=False) is True
