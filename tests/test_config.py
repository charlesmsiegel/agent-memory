"""Tests for AgentConfig and related dataclasses."""

from __future__ import annotations

from agent_memory.config import (
    AgentConfig,
    GroupChatConfig,
    HeartbeatConfig,
    SandboxConfig,
)


def test_sandbox_defaults() -> None:
    s = SandboxConfig()
    assert s.mode == "off"
    assert s.workspace_access == "rw"


def test_group_chat_defaults() -> None:
    g = GroupChatConfig()
    assert g.enabled is False
    assert g.respond_to_mentions is True
    assert g.respond_to_direct is True
    assert g.quiet_unless_mentioned is False


def test_heartbeat_defaults() -> None:
    h = HeartbeatConfig()
    assert h.enabled is False
    assert h.interval_minutes == 15
    assert h.quiet_start is None
    assert h.quiet_end is None
    assert h.tasks == []


def test_agent_config_defaults() -> None:
    a = AgentConfig()
    assert a.id == "default"
    assert a.name is None
    assert a.sandbox.mode == "off"
    assert a.group_chat.enabled is False
    assert a.heartbeat.enabled is False


def test_agent_config_from_dict_full() -> None:
    data = {
        "id": "bot1",
        "name": "TestBot",
        "sandbox": {"mode": "all", "workspace_access": "ro"},
        "group_chat": {"enabled": True, "quiet_unless_mentioned": True},
        "heartbeat": {
            "enabled": True,
            "interval_minutes": 5,
            "quiet_hours": {"start": "22:00", "end": "07:00"},
        },
    }
    a = AgentConfig.from_dict(data)
    assert a.id == "bot1"
    assert a.name == "TestBot"
    assert a.sandbox.mode == "all"
    assert a.sandbox.workspace_access == "ro"
    assert a.group_chat.enabled is True
    assert a.group_chat.quiet_unless_mentioned is True
    assert a.heartbeat.enabled is True
    assert a.heartbeat.interval_minutes == 5
    assert a.heartbeat.quiet_start == "22:00"
    assert a.heartbeat.quiet_end == "07:00"


def test_agent_config_from_dict_empty() -> None:
    a = AgentConfig.from_dict({})
    assert a.id == "default"
    assert a.sandbox.mode == "off"


def test_agent_config_from_dict_partial() -> None:
    a = AgentConfig.from_dict({"id": "x", "sandbox": {"mode": "non-main"}})
    assert a.id == "x"
    assert a.sandbox.mode == "non-main"
    assert a.sandbox.workspace_access == "rw"
