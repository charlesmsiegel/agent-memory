"""IDENTITY.md parsing + layered identity resolution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from ..memory.types import AgentIdentity

_PLACEHOLDER_VALUES = {
    "pick something you like",
    "ai? robot? familiar? ghost in the machine? something weirder?",
    "how do you come across? sharp? warm? chaotic? calm?",
    "your signature - pick one that feels right",
    "workspace-relative path, http(s) url, or data uri",
}

_FIELD_MAP = {"name", "emoji", "creature", "vibe", "theme", "avatar"}

_DEFAULT_ACK_REACTION = "👀"


def parse_identity_markdown(content: str) -> AgentIdentity:
    """Parse IDENTITY.md content into structured fields."""
    fields: dict[str, str] = {}
    for line in content.splitlines():
        cleaned = line.strip().lstrip("- ").strip()
        colon = cleaned.find(":")
        if colon == -1:
            continue
        label = re.sub(r"[*_]", "", cleaned[:colon]).strip().lower()
        value = re.sub(r"^[*_]+|[*_]+$", "", cleaned[colon + 1 :]).strip()
        if not value or label not in _FIELD_MAP:
            continue
        if _is_placeholder(value):
            continue
        fields[label] = value
    return AgentIdentity(**fields)


def load_identity(workspace_dir: str | Path) -> AgentIdentity | None:
    """Load and parse IDENTITY.md from workspace. Returns None if missing or empty."""
    p = Path(workspace_dir) / "IDENTITY.md"
    if not p.is_file():
        return None
    identity = parse_identity_markdown(p.read_text(encoding="utf-8"))
    return identity if identity.has_values() else None


# -- Layered resolution --


class AssistantIdentity(NamedTuple):
    """Resolved display identity for UI / gateway clients."""

    name: str
    avatar: str
    emoji: str | None


def resolve_identity(
    *,
    config_identity: dict | None = None,
    workspace_dir: str | Path | None = None,
) -> AgentIdentity:
    """Resolve agent identity with layered precedence: config → file → defaults."""
    file_identity = load_identity(workspace_dir) if workspace_dir else None
    cfg = config_identity or {}

    return AgentIdentity(
        name=cfg.get("name") or (file_identity.name if file_identity else None),
        emoji=cfg.get("emoji") or (file_identity.emoji if file_identity else None),
        creature=cfg.get("creature") or (file_identity.creature if file_identity else None),
        vibe=cfg.get("vibe") or (file_identity.vibe if file_identity else None),
        theme=cfg.get("theme") or (file_identity.theme if file_identity else None),
        avatar=cfg.get("avatar") or (file_identity.avatar if file_identity else None),
    )


def resolve_assistant_identity(
    *,
    config_identity: dict | None = None,
    config_ui_assistant: dict | None = None,
    workspace_dir: str | Path | None = None,
) -> AssistantIdentity:
    """Resolve display identity for UI: ui.assistant config → agent config → file → defaults."""
    ui = config_ui_assistant or {}
    agent = resolve_identity(config_identity=config_identity, workspace_dir=workspace_dir)

    name = (
        _coerce(ui.get("name"), 50)
        or _coerce(agent.name, 50)
        or "Assistant"
    )
    avatar = (
        _coerce(ui.get("avatar"), 200)
        or _coerce(agent.avatar, 200)
        or _coerce(agent.emoji, 200)
        or "A"
    )
    emoji = _coerce(agent.emoji, 16)

    return AssistantIdentity(name=name, avatar=avatar, emoji=emoji)


def resolve_ack_reaction(
    *,
    config_ack: str | None = None,
    config_identity: dict | None = None,
    workspace_dir: str | Path | None = None,
) -> str:
    """Resolve the emoji used to acknowledge message receipt."""
    if config_ack is not None:
        return config_ack.strip() or _DEFAULT_ACK_REACTION
    identity = resolve_identity(config_identity=config_identity, workspace_dir=workspace_dir)
    return identity.emoji or _DEFAULT_ACK_REACTION


def resolve_message_prefix(
    *,
    config_prefix: str | None = None,
    config_identity: dict | None = None,
    workspace_dir: str | Path | None = None,
    fallback: str = "[assistant]",
) -> str:
    """Resolve the prefix prepended to outbound messages (e.g., '[AgentName]')."""
    if config_prefix is not None:
        return config_prefix
    identity = resolve_identity(config_identity=config_identity, workspace_dir=workspace_dir)
    if identity.name:
        return f"[{identity.name}]"
    return fallback


# -- Helpers --


def _is_placeholder(value: str) -> bool:
    normalized = re.sub(r"[*_]", "", value).strip().lower()
    normalized = re.sub(r"^\(|\)$", "", normalized).strip()
    normalized = re.sub(r"[\u2013\u2014]", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized in _PLACEHOLDER_VALUES


def _coerce(value: str | None, max_len: int) -> str | None:
    if not value:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed[:max_len] if len(trimmed) > max_len else trimmed
