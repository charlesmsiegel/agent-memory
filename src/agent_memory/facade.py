"""Facade — single entry point for using pyclawmem as a package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .context.compaction import apply_compaction, compact, needs_compaction
from .context.memory_flush import (
    DEFAULT_FLUSH_PROMPT,
    DEFAULT_FLUSH_SYSTEM_PROMPT,
    DEFAULT_RESERVE_FLOOR_TOKENS,
    DEFAULT_SOFT_THRESHOLD_TOKENS,
    SILENT_TOKEN,
    should_run_memory_flush,
)
from .context.pruning import prune_stale_context
from .context.tool_pruning import prune_tool_results
from .context.session import SessionStore
from .context.session_export import export_session_to_markdown
from .context.window_guard import evaluate_guard, resolve_context_window
from .memory.auto_capture import auto_capture
from .memory.auto_recall import auto_recall
from .config import AgentConfig, GroupChatConfig
from .heartbeat import HeartbeatScheduler
from .memory.embeddings import resolve_embeddings
from .memory.facts import FactStore
from .memory.manager import MemoryManager
from .memory.store import MemoryStore
from .memory.types import CitationMode, MessageRole, PromptMode, SessionEntry, SessionMessage
from .personality.identity import (
    AssistantIdentity,
    resolve_assistant_identity,
    resolve_message_prefix,
)
from .personality.presence import HumanDelay, TypingMode, apply_human_delay, resolve_typing_mode
from .personality.soul import apply_soul_evil_override, decide_soul_evil, load_soul
from .personality.workspace import build_context_prompt, ensure_workspace, load_bootstrap_files


@dataclass
class Session:
    """Active session state. Passed to before/after hooks."""

    entry: SessionEntry
    messages: list[SessionMessage] = field(default_factory=list)
    system_prompt: str = ""
    compaction_count: int = 0
    flush_at_compaction: int | None = None
    _sandboxed: bool = False


@dataclass
class PreparedCall:
    """Result of before_llm_call — ready to send to your LLM."""

    messages: list[SessionMessage]
    system_prompt: str
    compacted: bool = False
    flushed: bool = False


def _is_within(base: Path, target: Path) -> bool:
    """Return True if *target* resolves to a path under *base*."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _should_respond(gc: GroupChatConfig, *, was_mentioned: bool, is_direct: bool) -> bool:
    """Check group chat config to decide if the agent should respond.

    Returns True unconditionally when group chat is disabled (1:1 mode).
    When enabled, the logic is:
      1. quiet_unless_mentioned=True → only respond when explicitly mentioned.
      2. Direct messages are accepted if respond_to_direct is set.
      3. Mentions are accepted if respond_to_mentions is set.
      4. Otherwise, respond freely unless quiet_unless_mentioned is on.
    """
    # 1:1 mode — always respond
    if not gc.enabled:
        return True
    # Quiet mode: ignore everything except explicit mentions
    if gc.quiet_unless_mentioned and not was_mentioned:
        return False
    # Accept direct messages when configured
    if is_direct and gc.respond_to_direct:
        return True
    # Accept mentions when configured
    if was_mentioned and gc.respond_to_mentions:
        return True
    # Fall-through: respond freely unless we're in quiet mode
    return not gc.quiet_unless_mentioned


class AgentMemory:
    """High-level facade wrapping all pyclawmem subsystems.

    Usage::

        agent = AgentMemory.from_config("CONFIG.yaml")
        session = agent.start_session()

        # In your message loop:
        prepared = agent.before_llm_call("user input", session)
        response = your_llm_call(prepared.messages, prepared.system_prompt)
        agent.after_llm_call(response, session)

        # When done:
        agent.end_session(session)
    """

    def __init__(
        self,
        *,
        memory: MemoryManager,
        facts: FactStore,
        sessions: SessionStore,
        workspace_dir: str,
        memory_dir: str,
        llm_model_id: str,
        llm_region: str,
        llm_max_tokens: int,
        max_context_tokens: int,
        compaction_threshold: float,
        summary_preserve: str,
        session_dir: str,
        # Soul-evil config (optional)
        soul_evil_chance: float = 0.0,
        soul_evil_purge_at: str | None = None,
        soul_evil_purge_duration: str | None = None,
        soul_evil_filename: str = "SOUL_EVIL.md",
        user_timezone: str = "UTC",
        # Presence config (optional)
        human_delay: HumanDelay | None = None,
        typing_mode: TypingMode | None = None,
        # NEW: agent config
        agent_config: AgentConfig | None = None,
        # NEW: context features
        prompt_mode: PromptMode = PromptMode.FULL,
        citation_mode: CitationMode = CitationMode.AUTO,
        bootstrap_max_chars: int = 20_000,
    ) -> None:
        self._memory = memory
        self._facts = facts
        self._sessions = sessions
        self._workspace_dir = workspace_dir
        self._memory_dir = memory_dir
        self._llm_model_id = llm_model_id
        self._llm_region = llm_region
        self._llm_max_tokens = llm_max_tokens
        self._max_context_tokens = max_context_tokens
        self._compaction_threshold = compaction_threshold
        self._summary_preserve = summary_preserve
        self._session_dir = session_dir
        self._soul_evil_chance = soul_evil_chance
        self._soul_evil_purge_at = soul_evil_purge_at
        self._soul_evil_purge_duration = soul_evil_purge_duration
        self._soul_evil_filename = soul_evil_filename
        self._user_timezone = user_timezone
        self._human_delay = human_delay
        self._typing_mode = typing_mode

        self._agent_config = agent_config or AgentConfig()
        self._prompt_mode = prompt_mode
        self._citation_mode = citation_mode
        self._bootstrap_max_chars = bootstrap_max_chars

        # Heartbeat
        if self._agent_config.heartbeat.enabled:
            self._heartbeat: HeartbeatScheduler | None = HeartbeatScheduler(
                self._agent_config.heartbeat,
                workspace_dir=workspace_dir,
                user_timezone=user_timezone,
            )
        else:
            self._heartbeat = None

        # Resolve identity once
        self._assistant = resolve_assistant_identity(workspace_dir=workspace_dir)
        self._prefix = resolve_message_prefix(workspace_dir=workspace_dir)

        # Context window guard
        window = resolve_context_window(config_context_tokens=max_context_tokens)
        self._guard = evaluate_guard(window)

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "AgentMemory":
        """Create an AgentMemory from a CONFIG.yaml file."""
        from . import load_config

        cfg = load_config(Path(config_path) if config_path else None)
        emb_cfg = cfg["embedding"]
        mem_cfg = cfg["memory"]
        ctx_cfg = cfg["context"]
        llm_cfg = cfg["llm"]
        ws_dir = cfg["workspace"]["dir"]

        embeddings = resolve_embeddings(emb_cfg)
        store = MemoryStore(mem_cfg["db_path"])
        memory = MemoryManager(
            store, embeddings,
            chunk_tokens=mem_cfg["chunk_tokens"],
            chunk_overlap=mem_cfg["chunk_overlap"],
            vector_weight=mem_cfg["hybrid_vector_weight"],
            text_weight=mem_cfg["hybrid_text_weight"],
            max_results=mem_cfg["max_results"],
            min_score=mem_cfg["min_score"],
        )
        facts = FactStore(
            db_path=mem_cfg["facts_db_path"],
            embeddings=embeddings,
        )
        sessions = SessionStore(ctx_cfg["session_dir"])
        memory_dir = mem_cfg["memory_dir"]

        agent_cfg = AgentConfig.from_dict(cfg.get("agent", {}))

        return cls(
            memory=memory,
            facts=facts,
            sessions=sessions,
            workspace_dir=ws_dir,
            memory_dir=memory_dir,
            llm_model_id=llm_cfg["model_id"],
            llm_region=llm_cfg.get("region", "us-east-1"),
            llm_max_tokens=llm_cfg.get("max_tokens", 1024),
            max_context_tokens=ctx_cfg.get("max_context_tokens", 100_000),
            compaction_threshold=ctx_cfg.get("compaction_threshold", 0.85),
            summary_preserve=ctx_cfg.get("summary_preserve", "decisions, TODOs, open questions"),
            session_dir=ctx_cfg["session_dir"],
            agent_config=agent_cfg,
            prompt_mode=PromptMode(ctx_cfg.get("prompt_mode", "full")),
            citation_mode=CitationMode(mem_cfg.get("citations", "auto")),
            bootstrap_max_chars=ctx_cfg.get("bootstrap_max_chars", 20_000),
        )

    # -- Properties --

    @property
    def identity(self) -> AssistantIdentity:
        """Resolved display identity (name, avatar, emoji)."""
        return self._assistant

    @property
    def message_prefix(self) -> str:
        """Prefix for outbound messages, e.g. '[AgentName]'."""
        return self._prefix

    @property
    def context_tokens(self) -> int:
        """Effective context window size in tokens."""
        return self._guard.tokens

    # -- Lifecycle hooks --

    def start_session(
        self,
        *,
        is_main_session: bool = True,
        is_subagent: bool = False,
    ) -> Session:
        """Start a new session: ensure workspace, build system prompt, sync memory."""
        sandboxed = self._is_sandboxed(is_main_session)

        if not sandboxed:
            ensure_workspace(self._workspace_dir)

        if self._guard.should_block:
            raise RuntimeError(
                f"Context window {self._guard.tokens} tokens is below hard minimum."
            )

        # Resolve effective prompt mode
        effective_mode = self._prompt_mode
        if is_subagent:
            effective_mode = PromptMode.MINIMAL

        if effective_mode == PromptMode.NONE:
            name = self._assistant.name or "Assistant"
            system_prompt = f"You are {name}."
        else:
            files = load_bootstrap_files(
                self._workspace_dir,
                is_main_session=is_main_session,
                is_subagent=(effective_mode == PromptMode.MINIMAL),
                max_chars_per_file=self._bootstrap_max_chars,
            )

            # Soul-evil decision
            soul = load_soul(self._workspace_dir)
            decision = decide_soul_evil(
                chance=self._soul_evil_chance,
                purge_at=self._soul_evil_purge_at,
                purge_duration=self._soul_evil_purge_duration,
                evil_filename=self._soul_evil_filename,
                user_timezone=self._user_timezone,
            )
            soul = apply_soul_evil_override(soul, self._workspace_dir, decision=decision)
            system_prompt = build_context_prompt(files)

            if effective_mode == PromptMode.FULL:
                daily_context = _load_recent_daily_logs(self._memory_dir)
                if daily_context:
                    system_prompt += f"\n\n# Recent Daily Logs\n\n{daily_context}"

        self._needs_sync = True
        entry = self._sessions.create_session()
        messages = [
            SessionMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
                metadata={"type": "bootstrap_load"},
            )
        ]
        return Session(entry=entry, messages=messages, system_prompt=system_prompt, _sandboxed=sandboxed)

    def before_llm_call(self, user_input: str, session: Session) -> PreparedCall:
        """Prepare messages for an LLM call.

        Handles: auto-recall, add user message, prune, memory flush, compact.
        """
        from .context.compaction import estimate_tokens as _est_tokens

        # Auto-recall
        recall_msg = auto_recall(
            user_input,
            fact_store=self._facts,
            memory_manager=self._memory,
        )
        if recall_msg:
            session.messages.append(recall_msg)

        # Add user message
        user_msg = SessionMessage(role=MessageRole.USER, content=user_input)
        session.messages.append(user_msg)
        self._sessions.append_message(session.entry, user_msg)

        # Prune stale context
        session.messages = prune_stale_context(session.messages, keep_recent=4)

        # Prune large tool results
        session.messages = prune_tool_results(session.messages)

        # Pre-compaction memory flush
        flushed = False
        total_tokens = _est_tokens(session.messages)
        if not session._sandboxed and should_run_memory_flush(
            total_tokens=total_tokens,
            context_window_tokens=self._guard.tokens,
            compaction_count=session.compaction_count,
            last_flush_at_compaction=session.flush_at_compaction,
        ):
            flush_msg = SessionMessage(
                role=MessageRole.USER,
                content=DEFAULT_FLUSH_PROMPT,
                metadata={"type": "memory_flush"},
            )
            session.messages.append(flush_msg)
            session.flush_at_compaction = session.compaction_count
            flushed = True

        # Compact if needed
        compacted = False
        if needs_compaction(session.messages, self._guard.tokens, self._compaction_threshold):
            result = compact(
                session.messages,
                model_id=self._llm_model_id,
                region=self._llm_region,
                max_tokens=self._llm_max_tokens,
                preserve_hint=self._summary_preserve,
            )
            session.messages = apply_compaction(session.messages, result)
            session.compaction_count += 1
            compacted = True

        # Human delay
        if self._human_delay:
            apply_human_delay(self._human_delay)

        return PreparedCall(
            messages=session.messages,
            system_prompt=session.system_prompt,
            compacted=compacted,
            flushed=flushed,
        )

    def after_llm_call(self, response_text: str, session: Session) -> None:
        """Record assistant response after LLM call."""
        msg = SessionMessage(role=MessageRole.ASSISTANT, content=response_text)
        session.messages.append(msg)
        self._sessions.append_message(session.entry, msg)

    def end_session(self, session: Session) -> None:
        """End a session: auto-capture facts, export to markdown."""
        if session._sandboxed:
            return
        auto_capture(session.messages, self._facts)
        export_session_to_markdown(
            session.messages,
            memory_dir=self._memory_dir,
            session_id=session.entry.session_id,
            llm_model_id=self._llm_model_id,
            llm_region=self._llm_region,
        )

    def close(self) -> None:
        """Release all resources."""
        self._memory.close()
        self._facts.close()

    # -- Utility --

    def _ensure_synced(self) -> None:
        if getattr(self, "_needs_sync", False):
            self._memory.sync(
                [self._memory_dir],
                session_dirs=[self._session_dir],
            )
            self._needs_sync = False

    def search_memory(self, query: str, *, max_results: int = 6) -> list:
        """Direct memory search (file-based)."""
        self._ensure_synced()
        return self._memory.search(query, max_results=max_results)

    def search_facts(self, query: str, *, limit: int = 5) -> list:
        """Direct fact store search."""
        return self._facts.recall(query, limit=limit, min_score=0.15)

    def store_fact(self, text: str, **kwargs) -> object | None:
        """Manually store a fact."""
        return self._facts.store(text, **kwargs)

    def forget_fact(self, fact_id: str) -> bool:
        """Delete a fact by ID."""
        return self._facts.forget(fact_id)

    def resolve_typing_mode(
        self,
        *,
        is_group_chat: bool = False,
        was_mentioned: bool = False,
        is_heartbeat: bool = False,
    ) -> TypingMode:
        """Resolve typing mode for the current context."""
        return resolve_typing_mode(
            configured=self._typing_mode,
            is_group_chat=is_group_chat,
            was_mentioned=was_mentioned,
            is_heartbeat=is_heartbeat,
        )

    def should_respond(
        self,
        *,
        was_mentioned: bool = False,
        is_direct: bool = False,
    ) -> bool:
        """Check group chat config to decide if the agent should respond."""
        return _should_respond(
            self._agent_config.group_chat,
            was_mentioned=was_mentioned,
            is_direct=is_direct,
        )

    @property
    def heartbeat(self) -> HeartbeatScheduler | None:
        """The heartbeat scheduler, or None if disabled."""
        return self._heartbeat

    @property
    def agent_config(self) -> AgentConfig:
        """The agent configuration."""
        return self._agent_config

    def _should_show_citations(self) -> bool:
        if self._citation_mode == CitationMode.ON:
            return True
        if self._citation_mode == CitationMode.OFF:
            return False
        # AUTO: show in main sessions, hide in group
        if self._agent_config.group_chat.enabled:
            return False
        return True

    def _is_sandboxed(self, is_main_session: bool) -> bool:
        mode = self._agent_config.sandbox.mode
        if mode == "all":
            return True
        if mode == "non-main" and not is_main_session:
            return True
        return False

    # -- Tools for LLM integration --

    def get_tool_definitions(self, *, session: Session | None = None) -> list[dict]:
        """Return tool definitions in Bedrock Converse toolSpec format."""
        tools = [
            {
                "toolSpec": {
                    "name": "memory_search",
                    "description": (
                        "Search through indexed workspace files and session transcripts. "
                        "Use when you need to find information from past conversations, notes, or documents."
                    ),
                    "inputSchema": {"json": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "description": "Max results (default: 6)"},
                        },
                        "required": ["query"],
                    }},
                }
            },
            {
                "toolSpec": {
                    "name": "fact_recall",
                    "description": (
                        "Search stored facts about the user — preferences, decisions, entities, "
                        "and other memorable information from past conversations."
                    ),
                    "inputSchema": {"json": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for"},
                            "limit": {"type": "integer", "description": "Max facts (default: 5)"},
                        },
                        "required": ["query"],
                    }},
                }
            },
            {
                "toolSpec": {
                    "name": "fact_store",
                    "description": (
                        "Store an important fact for long-term memory — preferences, decisions, "
                        "contact info, or anything worth remembering across sessions."
                    ),
                    "inputSchema": {"json": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The fact to remember"},
                            "category": {
                                "type": "string",
                                "enum": ["preference", "decision", "entity", "fact", "other"],
                                "description": "Category of the fact",
                            },
                        },
                        "required": ["text"],
                    }},
                }
            },
            {
                "toolSpec": {
                    "name": "workspace_read",
                    "description": (
                        "Read a workspace file. Valid files: SOUL.md, USER.md, IDENTITY.md, "
                        "TOOLS.md, AGENTS.md, HEARTBEAT.md, BOOTSTRAP.md, MEMORY.md."
                    ),
                    "inputSchema": {"json": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Filename to read"},
                        },
                        "required": ["filename"],
                    }},
                }
            },
            {
                "toolSpec": {
                    "name": "workspace_write",
                    "description": (
                        "Write or update a workspace file. Use to update USER.md with user info, "
                        "IDENTITY.md with agent identity, SOUL.md with personality, MEMORY.md with "
                        "curated memories. Provide the FULL file content."
                    ),
                    "inputSchema": {"json": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Filename to write"},
                            "content": {"type": "string", "description": "Full file content"},
                        },
                        "required": ["filename", "content"],
                    }},
                }
            },
            {
                "toolSpec": {
                    "name": "memory_get",
                    "description": (
                        "Read a specific memory file with optional line range. "
                        "Use after memory_search to get full context around a snippet."
                    ),
                    "inputSchema": {"json": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path to the memory file"},
                            "from_line": {"type": "integer", "description": "Start line (1-based, optional)"},
                            "lines": {"type": "integer", "description": "Number of lines to read (optional)"},
                        },
                        "required": ["path"],
                    }},
                }
            },
        ]
        if session and session._sandboxed:
            access = self._agent_config.sandbox.workspace_access
            if access == "none":
                tools = [t for t in tools if t["toolSpec"]["name"] not in ("workspace_read", "workspace_write")]
            elif access == "ro":
                tools = [t for t in tools if t["toolSpec"]["name"] != "workspace_write"]
        return tools

    _WRITABLE_FILES = {"SOUL.md", "USER.md", "IDENTITY.md", "TOOLS.md",
                       "AGENTS.md", "HEARTBEAT.md", "BOOTSTRAP.md", "MEMORY.md"}

    def execute_tool(self, name: str, input_data: dict, *, session: Session | None = None) -> str:
        """Execute a tool call by name. Returns result text."""
        sandboxed = session._sandboxed if session else False
        ws_access = self._agent_config.sandbox.workspace_access

        if name == "workspace_write" and sandboxed and ws_access != "rw":
            return "Blocked: workspace writes are disabled in sandbox mode."
        if name in ("workspace_read", "workspace_write") and sandboxed and ws_access == "none":
            return "Blocked: workspace access is disabled in sandbox mode."

        if name == "memory_search":
            results = self.search_memory(
                input_data["query"],
                max_results=input_data.get("max_results", 6),
            )
            if not results:
                return "No results found."
            show_citations = self._should_show_citations()
            lines = []
            for r in results:
                snippet = r.snippet[:300].replace("\n", " ")
                prefix = f"[{r.source.value}:{r.path}]"
                suffix = f" Source: {r.citation}" if show_citations else ""
                lines.append(f"{prefix} (score: {r.score:.2f}) {snippet}{suffix}")
            return "\n".join(lines)

        if name == "fact_recall":
            results = self.search_facts(
                input_data["query"],
                limit=input_data.get("limit", 5),
            )
            if not results:
                return "No matching facts found."
            lines = []
            for r in results:
                lines.append(f"[{r.fact.category.value}] (score: {r.score:.2f}) {r.fact.text}")
            return "\n".join(lines)

        if name == "fact_store":
            from .memory.facts import FactCategory
            cat = FactCategory(input_data.get("category", "other"))
            result = self.store_fact(input_data["text"], category=cat)
            if result is None:
                return "Similar fact already exists — skipped."
            return f"Stored: {input_data['text']}"

        if name == "workspace_read":
            filename = input_data["filename"]
            filepath = Path(self._workspace_dir) / filename
            if not _is_within(Path(self._workspace_dir), filepath):
                return f"Access denied: path escapes workspace"
            if not filepath.is_file():
                return f"File not found: {filename}"
            return filepath.read_text(encoding="utf-8")

        if name == "workspace_write":
            filename = input_data["filename"]
            content = input_data["content"]
            if filename not in self._WRITABLE_FILES:
                return f"Not allowed: {filename}. Allowed: {', '.join(sorted(self._WRITABLE_FILES))}"
            filepath = Path(self._workspace_dir) / filename
            if not _is_within(Path(self._workspace_dir), filepath):
                return f"Access denied: path escapes workspace"
            filepath.write_text(content, encoding="utf-8")
            return f"Written: {filename} ({len(content)} chars)"

        if name == "memory_get":
            raw_path = input_data["path"]
            # Resolve relative to memory_dir or workspace_dir
            for base in (Path(self._memory_dir), Path(self._workspace_dir)):
                candidate = base / raw_path
                if not _is_within(base, candidate):
                    continue
                if candidate.is_file():
                    all_lines = candidate.read_text(encoding="utf-8").splitlines()
                    from_line = input_data.get("from_line", 1)
                    num_lines = input_data.get("lines")
                    start = max(0, from_line - 1)
                    end = start + num_lines if num_lines else len(all_lines)
                    selected = all_lines[start:end]
                    return "\n".join(selected)
            return f"File not found: {raw_path}"

        return f"Unknown tool: {name}"


# -- Module-level helpers --


def _load_recent_daily_logs(memory_dir: str) -> str | None:
    """Load today's and yesterday's daily log files from memory_dir.

    Looks for files matching YYYY-MM-DD*.md pattern.
    """
    from datetime import datetime, timedelta, timezone

    d = Path(memory_dir)
    if not d.is_dir():
        return None

    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    prefixes = [today.isoformat(), yesterday.isoformat()]

    parts: list[str] = []
    for f in sorted(d.glob("*.md")):
        if any(f.name.startswith(p) for p in prefixes):
            content = f.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"## {f.name}\n\n{content}")

    return "\n\n".join(parts) if parts else None
