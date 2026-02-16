"""Agent-level configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SandboxConfig:
    """Application-level sandbox policy."""

    mode: Literal["off", "non-main", "all"] = "off"
    workspace_access: Literal["none", "ro", "rw"] = "rw"


@dataclass
class GroupChatConfig:
    """Group chat response behavior."""

    enabled: bool = False
    respond_to_mentions: bool = True
    respond_to_direct: bool = True
    quiet_unless_mentioned: bool = False


@dataclass
class HeartbeatConfig:
    """Periodic task scheduling."""

    enabled: bool = False
    interval_minutes: int = 15
    quiet_start: str | None = None
    quiet_end: str | None = None
    tasks: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Top-level agent configuration."""

    id: str = "default"
    name: str | None = None
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    group_chat: GroupChatConfig = field(default_factory=GroupChatConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)

    @classmethod
    def from_dict(cls, data: dict) -> AgentConfig:
        """Build from a config dict, filling defaults for missing keys."""
        sb = data.get("sandbox", {})
        gc = data.get("group_chat", {})
        hb = data.get("heartbeat", {})
        quiet = hb.get("quiet_hours", {})
        return cls(
            id=data.get("id", "default"),
            name=data.get("name"),
            sandbox=SandboxConfig(
                mode=sb.get("mode", "off"),
                workspace_access=sb.get("workspace_access", "rw"),
            ),
            group_chat=GroupChatConfig(
                enabled=gc.get("enabled", False),
                respond_to_mentions=gc.get("respond_to_mentions", True),
                respond_to_direct=gc.get("respond_to_direct", True),
                quiet_unless_mentioned=gc.get("quiet_unless_mentioned", False),
            ),
            heartbeat=HeartbeatConfig(
                enabled=hb.get("enabled", False),
                interval_minutes=hb.get("interval_minutes", 15),
                quiet_start=quiet.get("start"),
                quiet_end=quiet.get("end"),
                tasks=hb.get("tasks", []),
            ),
        )
