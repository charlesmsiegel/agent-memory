"""Tests for HeartbeatScheduler."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from agent_memory.config import HeartbeatConfig
from agent_memory.heartbeat import HeartbeatScheduler


def test_no_tasks_due_when_disabled() -> None:
    cfg = HeartbeatConfig(enabled=False)
    sched = HeartbeatScheduler(cfg, workspace_dir="/fake")
    assert sched.tasks_due() == []


def test_tasks_due_on_first_call(tmp_path: Path) -> None:
    hb = tmp_path / "HEARTBEAT.md"
    hb.write_text("Check email\nReview calendar\n", encoding="utf-8")
    cfg = HeartbeatConfig(enabled=True, interval_minutes=15)
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    due = sched.tasks_due()
    assert "Check email" in due
    assert "Review calendar" in due


def test_tasks_not_due_again_immediately(tmp_path: Path) -> None:
    hb = tmp_path / "HEARTBEAT.md"
    hb.write_text("Check email\n", encoding="utf-8")
    cfg = HeartbeatConfig(enabled=True, interval_minutes=15)
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    due = sched.tasks_due()
    assert len(due) == 1
    sched.mark_completed("Check email")
    assert sched.tasks_due() == []


def test_tasks_due_after_interval(tmp_path: Path) -> None:
    hb = tmp_path / "HEARTBEAT.md"
    hb.write_text("Check email\n", encoding="utf-8")
    cfg = HeartbeatConfig(enabled=True, interval_minutes=1)
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    sched.tasks_due()
    sched.mark_completed("Check email")
    sched._last_run["Check email"] = datetime(2000, 1, 1)
    due = sched.tasks_due()
    assert "Check email" in due


def test_quiet_hours_respected(tmp_path: Path) -> None:
    hb = tmp_path / "HEARTBEAT.md"
    hb.write_text("Check email\n", encoding="utf-8")
    cfg = HeartbeatConfig(
        enabled=True, interval_minutes=1,
        quiet_start="22:00", quiet_end="07:00",
    )
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    mock_now = datetime(2026, 2, 14, 23, 0, 0)
    with patch("agent_memory.heartbeat._now", return_value=mock_now):
        assert sched.is_quiet_hours() is True
        assert sched.tasks_due() == []


def test_outside_quiet_hours(tmp_path: Path) -> None:
    cfg = HeartbeatConfig(
        enabled=True, quiet_start="22:00", quiet_end="07:00",
    )
    sched = HeartbeatScheduler(cfg, workspace_dir="/fake")
    mock_now = datetime(2026, 2, 14, 12, 0, 0)
    with patch("agent_memory.heartbeat._now", return_value=mock_now):
        assert sched.is_quiet_hours() is False


def test_parse_skips_comments_and_blanks(tmp_path: Path) -> None:
    hb = tmp_path / "HEARTBEAT.md"
    hb.write_text(
        "# HEARTBEAT.md\n\n"
        "# This is a comment\n"
        "Check email\n"
        "\n"
        "  \n"
        "Review calendar\n",
        encoding="utf-8",
    )
    cfg = HeartbeatConfig(enabled=True)
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    tasks = sched.parse_heartbeat_file()
    assert tasks == ["Check email", "Review calendar"]


def test_no_heartbeat_file(tmp_path: Path) -> None:
    cfg = HeartbeatConfig(enabled=True)
    sched = HeartbeatScheduler(cfg, workspace_dir=str(tmp_path))
    assert sched.parse_heartbeat_file() == []
