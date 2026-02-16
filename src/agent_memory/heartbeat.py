"""Heartbeat scheduler — periodic task execution from HEARTBEAT.md."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from .config import HeartbeatConfig


def _now() -> datetime:
    """Current UTC time. Patchable for tests."""
    return datetime.utcnow()


class HeartbeatScheduler:
    """Caller-polled scheduler for periodic tasks.

    Does NOT run a background thread. The caller checks ``tasks_due()``
    on their own cadence (e.g., between LLM calls).
    """

    def __init__(
        self,
        config: HeartbeatConfig,
        workspace_dir: str,
        user_timezone: str = "UTC",
    ) -> None:
        self._config = config
        self._workspace_dir = Path(workspace_dir)
        self._user_timezone = user_timezone
        self._last_run: dict[str, datetime] = {}

    def is_quiet_hours(self) -> bool:
        """Check if current time falls within quiet hours window."""
        start = self._config.quiet_start
        end = self._config.quiet_end
        if not start or not end:
            return False

        now = _now()
        now_minutes = now.hour * 60 + now.minute
        start_h, start_m = (int(x) for x in start.split(":"))
        end_h, end_m = (int(x) for x in end.split(":"))
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m

        if start_minutes <= end_minutes:
            return start_minutes <= now_minutes < end_minutes
        else:
            return now_minutes >= start_minutes or now_minutes < end_minutes

    def tasks_due(self) -> list[str]:
        """Return tasks that are due for execution."""
        if not self._config.enabled:
            return []
        if self.is_quiet_hours():
            return []

        all_tasks = self.parse_heartbeat_file()
        if not all_tasks:
            return []

        now = _now()
        interval = timedelta(minutes=self._config.interval_minutes)
        due: list[str] = []
        for task in all_tasks:
            last = self._last_run.get(task)
            if last is None or (now - last) >= interval:
                due.append(task)
        return due

    def mark_completed(self, task: str) -> None:
        """Record that a task was just executed."""
        self._last_run[task] = _now()

    def parse_heartbeat_file(self) -> list[str]:
        """Parse HEARTBEAT.md into task strings.

        Skips blank lines, lines starting with ``#``, and whitespace-only lines.
        """
        path = self._workspace_dir / "HEARTBEAT.md"
        if not path.is_file():
            return []
        tasks: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tasks.append(stripped)
        return tasks
