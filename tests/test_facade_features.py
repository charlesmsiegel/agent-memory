"""Tests for new facade features: prompt modes, citations, sandbox, group chat."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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


# -- _should_respond edge cases (suggestion 4.3 coverage) --


def test_should_respond_not_mentioned_not_direct_no_quiet() -> None:
    """Group chat enabled, not quiet, no mention, no direct → respond freely."""
    from agent_memory.facade import _should_respond
    gc = GroupChatConfig(enabled=True, quiet_unless_mentioned=False)
    assert _should_respond(gc, was_mentioned=False, is_direct=False) is True


def test_should_respond_mention_disabled() -> None:
    """mentioned but respond_to_mentions=False, not quiet → falls to free-respond."""
    from agent_memory.facade import _should_respond
    gc = GroupChatConfig(
        enabled=True, respond_to_mentions=False, quiet_unless_mentioned=False,
    )
    assert _should_respond(gc, was_mentioned=True, is_direct=False) is True


# -- _is_sandboxed (suggestion 4.5) --


def _make_agent_memory(tmp_path, *, agent_config=None, citation_mode=CitationMode.AUTO):
    """Create an AgentMemory with minimal mocked dependencies."""
    from agent_memory.facade import AgentMemory

    mem_mock = MagicMock()
    facts_mock = MagicMock()
    sessions_mock = MagicMock()

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(exist_ok=True)
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(exist_ok=True)

    # Minimal files to prevent errors
    for fname in ("IDENTITY.md", "SOUL.md", "SOUL_EVIL.md"):
        (workspace / fname).write_text("", encoding="utf-8")

    with patch("agent_memory.facade.resolve_context_window") as mock_win, \
         patch("agent_memory.facade.evaluate_guard") as mock_guard, \
         patch("agent_memory.facade.resolve_assistant_identity") as mock_ident, \
         patch("agent_memory.facade.resolve_message_prefix", return_value=">"):
        guard = MagicMock()
        guard.should_block = False
        guard.tokens = 100000
        mock_guard.return_value = guard
        mock_win.return_value = 100000
        ident = MagicMock()
        ident.name = "TestBot"
        mock_ident.return_value = ident

        am = AgentMemory(
            memory=mem_mock,
            facts=facts_mock,
            sessions=sessions_mock,
            workspace_dir=str(workspace),
            memory_dir=str(memory_dir),
            llm_model_id="test-model",
            llm_region="us-east-1",
            llm_max_tokens=1024,
            max_context_tokens=100000,
            compaction_threshold=0.85,
            summary_preserve="decisions",
            session_dir=str(session_dir),
            agent_config=agent_config,
            citation_mode=citation_mode,
        )
    return am


def test_is_sandboxed_off(tmp_path: Path) -> None:
    from agent_memory.facade import AgentMemory
    cfg = AgentConfig(sandbox=SandboxConfig(mode="off"))
    am = _make_agent_memory(tmp_path, agent_config=cfg)
    assert am._is_sandboxed(is_main_session=True) is False
    assert am._is_sandboxed(is_main_session=False) is False


def test_is_sandboxed_all(tmp_path: Path) -> None:
    cfg = AgentConfig(sandbox=SandboxConfig(mode="all"))
    am = _make_agent_memory(tmp_path, agent_config=cfg)
    assert am._is_sandboxed(is_main_session=True) is True
    assert am._is_sandboxed(is_main_session=False) is True


def test_is_sandboxed_non_main(tmp_path: Path) -> None:
    cfg = AgentConfig(sandbox=SandboxConfig(mode="non-main"))
    am = _make_agent_memory(tmp_path, agent_config=cfg)
    assert am._is_sandboxed(is_main_session=True) is False
    assert am._is_sandboxed(is_main_session=False) is True


# -- _should_show_citations (suggestion 4.5) --


def test_citations_on(tmp_path: Path) -> None:
    am = _make_agent_memory(tmp_path, citation_mode=CitationMode.ON)
    assert am._should_show_citations() is True


def test_citations_off(tmp_path: Path) -> None:
    am = _make_agent_memory(tmp_path, citation_mode=CitationMode.OFF)
    assert am._should_show_citations() is False


def test_citations_auto_hides_in_group_chat(tmp_path: Path) -> None:
    cfg = AgentConfig(group_chat=GroupChatConfig(enabled=True))
    am = _make_agent_memory(tmp_path, agent_config=cfg, citation_mode=CitationMode.AUTO)
    assert am._should_show_citations() is False


def test_citations_auto_shows_in_non_group(tmp_path: Path) -> None:
    cfg = AgentConfig(group_chat=GroupChatConfig(enabled=False))
    am = _make_agent_memory(tmp_path, agent_config=cfg, citation_mode=CitationMode.AUTO)
    assert am._should_show_citations() is True


# -- Sandbox enforcement in execute_tool (suggestion 4.5) --


def test_sandbox_blocks_workspace_write(tmp_path: Path) -> None:
    from agent_memory.facade import Session
    cfg = AgentConfig(sandbox=SandboxConfig(mode="all", workspace_access="ro"))
    am = _make_agent_memory(tmp_path, agent_config=cfg)
    session = Session(entry=MagicMock(), _sandboxed=True)
    result = am.execute_tool("workspace_write", {"filename": "SOUL.md", "content": "x"}, session=session)
    assert "Blocked" in result


def test_sandbox_blocks_workspace_read_when_none(tmp_path: Path) -> None:
    from agent_memory.facade import Session
    cfg = AgentConfig(sandbox=SandboxConfig(mode="all", workspace_access="none"))
    am = _make_agent_memory(tmp_path, agent_config=cfg)
    session = Session(entry=MagicMock(), _sandboxed=True)
    result = am.execute_tool("workspace_read", {"filename": "SOUL.md"}, session=session)
    assert "Blocked" in result


# -- Path traversal protection (regression) --


def test_workspace_read_path_traversal(tmp_path: Path) -> None:
    am = _make_agent_memory(tmp_path)
    result = am.execute_tool("workspace_read", {"filename": "../../etc/passwd"})
    assert "Access denied" in result


def test_memory_get_path_traversal(tmp_path: Path) -> None:
    am = _make_agent_memory(tmp_path)
    result = am.execute_tool("memory_get", {"path": "../../../etc/shadow"})
    assert "File not found" in result


# -- Tool filtering in get_tool_definitions (suggestion 4.5) --


def test_tool_definitions_filter_sandbox_none(tmp_path: Path) -> None:
    from agent_memory.facade import Session
    cfg = AgentConfig(sandbox=SandboxConfig(mode="all", workspace_access="none"))
    am = _make_agent_memory(tmp_path, agent_config=cfg)
    session = Session(entry=MagicMock(), _sandboxed=True)
    tools = am.get_tool_definitions(session=session)
    tool_names = {t["toolSpec"]["name"] for t in tools}
    assert "workspace_read" not in tool_names
    assert "workspace_write" not in tool_names


def test_tool_definitions_filter_sandbox_ro(tmp_path: Path) -> None:
    from agent_memory.facade import Session
    cfg = AgentConfig(sandbox=SandboxConfig(mode="all", workspace_access="ro"))
    am = _make_agent_memory(tmp_path, agent_config=cfg)
    session = Session(entry=MagicMock(), _sandboxed=True)
    tools = am.get_tool_definitions(session=session)
    tool_names = {t["toolSpec"]["name"] for t in tools}
    assert "workspace_read" in tool_names
    assert "workspace_write" not in tool_names


# -- from_config (suggestion 4.4) --


def test_from_config(tmp_path: Path) -> None:
    """AgentMemory.from_config constructs all subsystems from CONFIG.yaml."""
    from agent_memory.facade import AgentMemory

    config_yaml = tmp_path / "CONFIG.yaml"
    config_yaml.write_text("""\
embedding:
  provider: bedrock
  model_id: amazon.titan-embed-text-v2:0
  dimensions: 1024
  region: us-east-1
llm:
  provider: bedrock
  model_id: test-model
  region: us-east-1
  max_tokens: 1024
memory:
  db_path: {tmp}/memory.sqlite
  facts_db_path: {tmp}/facts.sqlite
  memory_dir: {tmp}/memory
  chunk_tokens: 400
  chunk_overlap: 80
  hybrid_vector_weight: 0.7
  hybrid_text_weight: 0.3
  max_results: 6
  min_score: 0.35
context:
  max_context_tokens: 100000
  compaction_threshold: 0.85
  summary_preserve: "decisions"
  session_dir: {tmp}/sessions
workspace:
  dir: {tmp}/workspace
agent:
  id: test-agent
  sandbox:
    mode: "off"
""".format(tmp=tmp_path), encoding="utf-8")

    # Create required directories and files
    for d in ("memory", "sessions", "workspace"):
        (tmp_path / d).mkdir(exist_ok=True)
    for fname in ("IDENTITY.md", "SOUL.md", "SOUL_EVIL.md"):
        (tmp_path / "workspace" / fname).write_text("", encoding="utf-8")

    with patch("agent_memory.memory.store.MemoryStore") as mock_store, \
         patch("agent_memory.memory.embeddings._build_provider") as mock_build, \
         patch("agent_memory.facade.resolve_context_window", return_value=100000), \
         patch("agent_memory.facade.evaluate_guard") as mock_guard, \
         patch("agent_memory.facade.resolve_assistant_identity") as mock_ident, \
         patch("agent_memory.facade.resolve_message_prefix", return_value=">"):
        mock_emb = MagicMock()
        mock_emb.dimensions = 1024
        mock_build.return_value = mock_emb
        guard = MagicMock()
        guard.should_block = False
        guard.tokens = 100000
        mock_guard.return_value = guard
        ident = MagicMock()
        ident.name = "TestBot"
        mock_ident.return_value = ident

        am = AgentMemory.from_config(str(config_yaml))

    assert am.agent_config.id == "test-agent"
    assert am.agent_config.sandbox.mode == "off"
    assert am._llm_model_id == "test-model"
