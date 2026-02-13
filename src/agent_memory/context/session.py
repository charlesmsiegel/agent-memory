"""JSONL session transcript persistence."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..memory.types import MessageRole, SessionEntry, SessionMessage

log = logging.getLogger(__name__)


class SessionStore:
    """Manages JSONL session transcripts on disk."""

    def __init__(self, session_dir: str | Path) -> None:
        self._dir = Path(session_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> SessionEntry:
        """Create a new session and return its entry."""
        sid = uuid4().hex[:12]
        path = self._dir / f"{sid}.jsonl"
        entry = SessionEntry(session_id=sid, transcript_path=str(path))
        return entry

    def append_message(self, session: SessionEntry, message: SessionMessage) -> None:
        """Append a message to the session's JSONL transcript."""
        if not session.transcript_path:
            raise ValueError("Session has no transcript path")
        line = json.dumps({
            "type": "message",
            "role": message.role.value,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "metadata": message.metadata,
        })
        with open(session.transcript_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        session.message_count += 1
        session.updated_at = datetime.utcnow()

    def read_messages(self, session: SessionEntry) -> list[SessionMessage]:
        """Read all messages from a session transcript."""
        if not session.transcript_path:
            return []
        path = Path(session.transcript_path)
        if not path.is_file():
            return []
        messages: list[SessionMessage] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "message":
                continue
            messages.append(SessionMessage(
                role=MessageRole(entry["role"]),
                content=entry["content"],
                timestamp=datetime.fromisoformat(entry.get("timestamp", "")),
                metadata=entry.get("metadata", {}),
            ))
        return messages

    def repair_transcript(self, session: SessionEntry) -> int:
        """Repair a corrupted JSONL file by dropping bad lines. Returns lines dropped."""
        if not session.transcript_path:
            return 0
        path = Path(session.transcript_path)
        if not path.is_file():
            return 0
        raw = path.read_text(encoding="utf-8")
        good: list[str] = []
        dropped = 0
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
                good.append(line)
            except json.JSONDecodeError:
                dropped += 1
        if dropped > 0:
            backup = path.with_suffix(".jsonl.bak")
            shutil.copy2(path, backup)
            path.write_text("\n".join(good) + "\n", encoding="utf-8")
            log.warning("Repaired %s: dropped %d lines, backup at %s", path, dropped, backup)
        return dropped

    def list_sessions(self) -> list[SessionEntry]:
        """List all sessions by scanning the session directory."""
        entries: list[SessionEntry] = []
        for f in sorted(self._dir.glob("*.jsonl")):
            sid = f.stem
            count = sum(1 for line in f.read_text().splitlines() if line.strip())
            entries.append(SessionEntry(
                session_id=sid,
                transcript_path=str(f),
                message_count=count,
            ))
        return entries
