"""Context compaction — LLM summarization via Bedrock when context fills."""

from __future__ import annotations

import json
import logging

import boto3

from ..memory.types import CompactionResult, SessionMessage

log = logging.getLogger(__name__)

# Rough chars-per-token for estimation
_CHARS_PER_TOKEN = 4


def estimate_tokens(messages: list[SessionMessage]) -> int:
    """Rough token estimate for a list of messages."""
    return sum(len(m.content) // _CHARS_PER_TOKEN + 1 for m in messages)


def needs_compaction(
    messages: list[SessionMessage],
    max_context_tokens: int,
    threshold: float = 0.85,
) -> bool:
    """Check if messages exceed the compaction threshold."""
    return estimate_tokens(messages) > int(max_context_tokens * threshold)


def compact(
    messages: list[SessionMessage],
    *,
    model_id: str,
    region: str = "us-east-1",
    max_tokens: int = 1024,
    preserve_hint: str = "decisions, TODOs, open questions, constraints",
) -> CompactionResult:
    """Summarize older messages to free context space.

    Keeps the most recent 30% of messages verbatim and summarizes the rest.
    """
    if len(messages) <= 2:
        return CompactionResult(
            summary="",
            messages_removed=0,
            tokens_before=estimate_tokens(messages),
            tokens_after=estimate_tokens(messages),
        )

    split = max(1, int(len(messages) * 0.7))
    old_messages = messages[:split]
    tokens_before = estimate_tokens(messages)

    # Extract tool failures and file ops before summarizing
    tool_failures = _extract_tool_failures(old_messages)
    file_ops = _extract_file_operations(old_messages)

    transcript = "\n".join(
        f"{m.role.value}: {m.content}" for m in old_messages
    )
    prompt = (
        f"Summarize this conversation concisely. Preserve: {preserve_hint}.\n\n"
        f"{transcript}"
    )

    summary = _invoke_bedrock(prompt, model_id=model_id, region=region, max_tokens=max_tokens)
    kept = messages[split:]
    tokens_after = estimate_tokens(kept) + len(summary) // _CHARS_PER_TOKEN

    return CompactionResult(
        summary=summary,
        messages_removed=len(old_messages),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tool_failures=tool_failures,
        file_operations=file_ops,
    )


def apply_compaction(
    messages: list[SessionMessage],
    result: CompactionResult,
) -> list[SessionMessage]:
    """Replace old messages with summary + safeguard context."""
    kept = messages[result.messages_removed:]
    parts = [result.summary]
    if result.tool_failures:
        parts.append("\n## Recent Tool Failures\n" + "\n".join(f"- {f}" for f in result.tool_failures))
    if result.file_operations:
        parts.append("\n## File Operations\n" + "\n".join(f"- {f}" for f in result.file_operations))
    summary_msg = SessionMessage(
        role="system",
        content="\n".join(parts),
        metadata={"type": "compaction_summary"},
    )
    return [summary_msg] + kept


# -- Helpers --


def _invoke_bedrock(prompt: str, *, model_id: str, region: str, max_tokens: int) -> str:
    client = boto3.client("bedrock-runtime", region_name=region)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = client.invoke_model(modelId=model_id, body=body)
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def _extract_tool_failures(messages: list[SessionMessage], max_failures: int = 8) -> list[str]:
    """Extract recent tool failure summaries from messages."""
    failures: list[str] = []
    for m in messages:
        meta = m.metadata
        if meta.get("type") == "tool_result" and meta.get("is_error"):
            name = meta.get("tool_name", "unknown")
            snippet = m.content[:240].replace("\n", " ")
            failures.append(f"[{name}] {snippet}")
    return failures[-max_failures:]


def _extract_file_operations(messages: list[SessionMessage]) -> list[str]:
    """Extract file read/write operations from tool metadata."""
    ops: list[str] = []
    seen: set[str] = set()
    for m in messages:
        meta = m.metadata
        if meta.get("type") != "tool_result":
            continue
        for key in ("files_read", "files_written"):
            for path in meta.get(key, []):
                label = f"{'read' if 'read' in key else 'write'}: {path}"
                if label not in seen:
                    seen.add(label)
                    ops.append(label)
    return ops
