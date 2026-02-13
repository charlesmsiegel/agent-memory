"""Session-to-markdown export — dump conversation on session reset."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import boto3

from ..memory.types import SessionMessage

log = logging.getLogger(__name__)


def export_session_to_markdown(
    messages: list[SessionMessage],
    *,
    memory_dir: str | Path,
    session_id: str,
    session_key: str = "main",
    llm_model_id: str | None = None,
    llm_region: str = "us-east-1",
    message_count: int = 15,
) -> Path | None:
    """Export recent session messages to memory/YYYY-MM-DD-{slug}.md.

    Generates a descriptive slug via Bedrock LLM, falls back to HHMM timestamp.
    Returns the written file path, or None on failure.
    """
    mem_dir = Path(memory_dir).expanduser()
    mem_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Extract recent user/assistant messages
    relevant = [
        m for m in messages
        if m.role.value in ("user", "assistant") and m.content and not m.content.startswith("/")
    ][-message_count:]

    if not relevant:
        return None

    transcript = "\n".join(f"{m.role.value}: {m.content}" for m in relevant)

    # Generate slug
    slug = _generate_slug(transcript, model_id=llm_model_id, region=llm_region)
    if not slug:
        slug = now.strftime("%H%M")

    filename = f"{date_str}-{slug}.md"
    file_path = mem_dir / filename

    # Build markdown
    lines = [
        f"# Session: {date_str} {time_str} UTC",
        "",
        f"- **Session Key**: {session_key}",
        f"- **Session ID**: {session_id}",
        "",
        "## Conversation Summary",
        "",
        transcript,
        "",
    ]
    file_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Session exported to %s", file_path)
    return file_path


def _generate_slug(transcript: str, *, model_id: str | None, region: str) -> str | None:
    """Generate a short descriptive slug via Bedrock LLM."""
    if not model_id:
        return None
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        prompt = (
            "Generate a short 2-4 word slug (lowercase, hyphens, no spaces) "
            "describing this conversation's main topic. Reply with ONLY the slug.\n\n"
            f"{transcript[:2000]}"
        )
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(modelId=model_id, body=body)
        result = json.loads(response["body"].read())
        raw = result["content"][0]["text"].strip().lower()
        # Sanitize: keep only alphanumeric and hyphens
        slug = "".join(c if c.isalnum() or c == "-" else "-" for c in raw)
        slug = slug.strip("-")[:40]
        return slug if slug else None
    except Exception as e:
        log.warning("Slug generation failed: %s", e)
        return None
