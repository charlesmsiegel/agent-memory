"""Integration — wires all pyclawmem subsystems into an LLM agent lifecycle.

This is NOT a runnable agent. It demonstrates every connection point
where pyclawmem hooks into your agent's lifecycle.
"""

from __future__ import annotations

from pyclawmem import load_config
from pyclawmem.context.compaction import apply_compaction, compact, needs_compaction
from pyclawmem.context.pruning import prune_stale_context
from pyclawmem.context.session import SessionStore
from pyclawmem.context.session_export import export_session_to_markdown
from pyclawmem.context.window_guard import evaluate_guard, resolve_context_window
from pyclawmem.memory.auto_capture import auto_capture
from pyclawmem.memory.auto_recall import auto_recall
try:
    from pyclawmem.memory.embeddings import BedrockEmbeddings
except ImportError:
    BedrockEmbeddings = None  # type: ignore[assignment,misc]
from pyclawmem.memory.facts import FactStore
from pyclawmem.memory.manager import MemoryManager
from pyclawmem.memory.store import MemoryStore
from pyclawmem.memory.types import MessageRole, SessionMessage
from pyclawmem.personality.identity import resolve_assistant_identity, resolve_message_prefix
from pyclawmem.personality.presence import (
    HumanDelay,
    TypingMode,
    apply_human_delay,
    resolve_typing_mode,
)
from pyclawmem.personality.soul import (
    apply_soul_evil_override,
    decide_soul_evil,
    load_soul,
)
from pyclawmem.personality.workspace import (
    build_context_prompt,
    ensure_workspace,
    load_bootstrap_files,
)


def create_agent_context(config_path: str | None = None) -> dict:
    """Initialize all pyclawmem subsystems from CONFIG.yaml."""
    cfg = load_config()
    emb_cfg = cfg["embedding"]
    mem_cfg = cfg["memory"]
    ctx_cfg = cfg["context"]
    ws_dir = cfg["workspace"]["dir"]

    embeddings = BedrockEmbeddings(
        model_id=emb_cfg["model_id"],
        region=emb_cfg["region"],
        dimensions=emb_cfg["dimensions"],
    )
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
        db_path=str(mem_cfg["db_path"]).replace(".sqlite", "-facts.sqlite"),
        embeddings=embeddings,
    )
    sessions = SessionStore(ctx_cfg["session_dir"])

    return {
        "config": cfg,
        "memory": memory,
        "facts": facts,
        "sessions": sessions,
        "embeddings": embeddings,
        "workspace_dir": ws_dir,
    }


def example_agent_loop():
    """Full lifecycle demonstrating every pyclawmem hook."""
    ctx = create_agent_context()
    cfg = ctx["config"]
    memory: MemoryManager = ctx["memory"]
    facts: FactStore = ctx["facts"]
    sessions: SessionStore = ctx["sessions"]
    workspace_dir: str = ctx["workspace_dir"]
    llm_cfg = cfg["llm"]
    ctx_cfg = cfg["context"]

    # ================================================================
    # PHASE 1: Session start
    # ================================================================

    # 1a. Ensure workspace exists with default templates
    ensure_workspace(workspace_dir)

    # 1b. Context window guard
    window = resolve_context_window(
        config_context_tokens=ctx_cfg.get("max_context_tokens"),
    )
    guard = evaluate_guard(window)
    if guard.should_block:
        print(f"BLOCKED: context window {guard.tokens} tokens below hard minimum.")
        return

    # 1c. Load personality (with MEMORY.md security scoping)
    bootstrap_files = load_bootstrap_files(
        workspace_dir,
        is_main_session=True,   # Set False for group chats / shared surfaces
        is_subagent=False,      # Set True for spawned subagents
    )

    # 1d. Soul-evil decision
    soul_content = load_soul(workspace_dir)
    evil_decision = decide_soul_evil(
        chance=0.0,             # Configure from hook settings
        purge_at=None,          # e.g., "21:00"
        purge_duration=None,    # e.g., "15m"
        user_timezone="UTC",
    )
    soul_content = apply_soul_evil_override(
        soul_content, workspace_dir, decision=evil_decision,
    )

    # 1e. Build system prompt
    system_prompt = build_context_prompt(bootstrap_files)

    # 1f. Resolve identity for display
    assistant = resolve_assistant_identity(workspace_dir=workspace_dir)
    prefix = resolve_message_prefix(workspace_dir=workspace_dir)
    print(f"[{assistant.name}] Session started. Prefix: {prefix}")

    # 1g. Sync memory index (files + session transcripts)
    memory.sync(
        [f"{workspace_dir}/memory"],
        session_dirs=[cfg["context"]["session_dir"]],
    )

    # 1h. Create session
    session = sessions.create_session()
    messages: list[SessionMessage] = [
        SessionMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt,
            metadata={"type": "bootstrap_load"},
        ),
    ]

    # ================================================================
    # PHASE 2: Conversation loop
    # ================================================================

    # Resolve typing mode for this context
    typing_mode = resolve_typing_mode(is_group_chat=False)
    delay = HumanDelay(min_ms=500, max_ms=1500)

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input == "/quit":
            break

        if user_input == "/new":
            # Session reset — export and start fresh
            export_session_to_markdown(
                messages,
                memory_dir=f"{workspace_dir}/memory",
                session_id=session.session_id,
                llm_model_id=llm_cfg["model_id"],
                llm_region=llm_cfg["region"],
            )
            auto_capture(messages, facts)
            print("  [session exported + facts captured]")
            session = sessions.create_session()
            messages = [messages[0]]  # Keep system prompt
            continue

        # 2a. Auto-recall: inject relevant memories
        recall_msg = auto_recall(
            user_input,
            fact_store=facts,
            memory_manager=memory,
        )
        if recall_msg:
            messages.append(recall_msg)

        # 2b. Add user message
        user_msg = SessionMessage(role=MessageRole.USER, content=user_input)
        messages.append(user_msg)
        sessions.append_message(session, user_msg)

        # 2c. Prune stale context
        messages = prune_stale_context(messages, keep_recent=4)

        # 2d. Compact if needed
        if needs_compaction(messages, guard.tokens, ctx_cfg.get("compaction_threshold", 0.85)):
            result = compact(
                messages,
                model_id=llm_cfg["model_id"],
                region=llm_cfg["region"],
                max_tokens=llm_cfg["max_tokens"],
                preserve_hint=ctx_cfg.get("summary_preserve", "decisions, TODOs, open questions"),
            )
            messages = apply_compaction(messages, result)
            print(f"  [compacted: {result.messages_removed} messages → {len(result.summary)} char summary]")

        # 2e. Typing indicator
        if typing_mode == TypingMode.INSTANT:
            print(f"  [{assistant.name} is typing...]")

        # 2f. Human delay
        apply_human_delay(delay)

        # 2g. YOUR LLM CALL HERE
        # response = bedrock_converse(model_id, messages)
        response_text = f"[LLM response placeholder for: {user_input}]"

        # 2h. Record assistant response
        assistant_msg = SessionMessage(role=MessageRole.ASSISTANT, content=response_text)
        messages.append(assistant_msg)
        sessions.append_message(session, assistant_msg)

        print(f"{prefix} {response_text}")

    # ================================================================
    # PHASE 3: Session end
    # ================================================================

    # Auto-capture facts from conversation
    captured = auto_capture(messages, facts)
    if captured:
        print(f"  [auto-captured {captured} facts]")

    # Export session to markdown
    export_session_to_markdown(
        messages,
        memory_dir=f"{workspace_dir}/memory",
        session_id=session.session_id,
        llm_model_id=llm_cfg["model_id"],
        llm_region=llm_cfg["region"],
    )

    memory.close()
    facts.close()
    print(f"Session {session.session_id} ended.")


if __name__ == "__main__":
    example_agent_loop()
