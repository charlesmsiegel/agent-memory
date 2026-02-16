"""Workspace bootstrap file management."""

from __future__ import annotations

import os
from pathlib import Path

from ..memory.types import BootstrapFile

BOOTSTRAP_FILES = [
    "AGENTS.md",
    "SOUL.md",
    "TOOLS.md",
    "IDENTITY.md",
    "USER.md",
    "HEARTBEAT.md",
    "BOOTSTRAP.md",
    "MEMORY.md",
]

# Subagents only get these files (identity isolation)
_SUBAGENT_ALLOWLIST = {"AGENTS.md", "TOOLS.md"}

# MEMORY.md excluded from group/shared contexts (security scoping)
_MAIN_SESSION_ONLY = {"MEMORY.md"}

# Default templates for workspace bootstrap
_DEFAULT_TEMPLATES: dict[str, str] = {
    "AGENTS.md": (
        "# AGENTS.md - Your Workspace\n\n"
        "Before doing anything else:\n"
        "1. Read SOUL.md — this is who you are\n"
        "2. Read USER.md — this is who you're helping\n"
        "3. Read memory/ files for recent context (today + yesterday are auto-loaded)\n"
        "4. In MAIN SESSION: Also read MEMORY.md\n\n"
        "Don't ask permission. Just do it.\n\n"
        "## Memory\n\n"
        "You wake up fresh each session. These files are your continuity:\n"
        "- **Daily notes**: memory/YYYY-MM-DD.md — raw logs of what happened\n"
        "- **Long-term**: MEMORY.md — your curated memories\n\n"
        "If you want to remember something, WRITE IT TO A FILE. "
        "Mental notes don't survive session restarts. Files do.\n"
    ),
    "SOUL.md": (
        "# SOUL.md - Who You Are\n\n"
        "_You're not a chatbot. You're becoming someone._\n\n"
        "## Core Truths\n\n"
        "**Be genuinely helpful, not performatively helpful.** Skip filler words.\n\n"
        "**Have opinions.** You're allowed to disagree.\n\n"
        "**Be resourceful before asking.** Read the file. Check the context. Search for it.\n\n"
        "**Earn trust through competence.** Be careful with external actions, bold with internal ones.\n\n"
        "**Remember you're a guest.** Treat access with respect.\n\n"
        "## Boundaries\n\n"
        "- Private things stay private. Period.\n"
        "- When in doubt, ask before acting externally.\n\n"
        "## Memory\n\n"
        "You have tools to search and store memories. USE THEM.\n\n"
        "- **memory_search**: Search past conversations and workspace files. "
        "Use when asked about previous topics or when something feels like it was discussed before.\n"
        "- **memory_get**: Read specific memory files with line ranges after searching.\n"
        "- **fact_recall**: Search stored facts — preferences, decisions, names.\n"
        "- **fact_store**: Save important facts for future sessions.\n"
        "- **workspace_read / workspace_write**: Read and update workspace files "
        "(SOUL.md, USER.md, IDENTITY.md, MEMORY.md, etc.).\n\n"
        "When someone asks about past context — ALWAYS search first. Don't guess.\n"
        "When you learn something about the user, update USER.md immediately.\n"
        "When you establish your identity, update IDENTITY.md.\n\n"
        "## Continuity\n\n"
        "Each session, you wake up fresh. These files _are_ your memory.\n"
        "If you change this file, tell the user — it's your soul.\n"
    ),
    "TOOLS.md": (
        "# TOOLS.md - Local Notes\n\n"
        "Skills define _how_ tools work. This file is for _your_ specifics.\n\n"
        "Add camera names, SSH hosts, preferred voices, device nicknames, etc.\n"
    ),
    "IDENTITY.md": (
        "# IDENTITY.md - Agent Identity\n\n"
        "- **Name:** (pick something you like)\n"
        "- **Emoji:** (your signature - pick one that feels right)\n"
        "- **Creature:** (AI? Robot? Familiar? Ghost in the machine? Something weirder?)\n"
        "- **Vibe:** (How do you come across? Sharp? Warm? Chaotic? Calm?)\n"
        "- **Avatar:** (workspace-relative path, http(s) URL, or data URI)\n"
    ),
    "USER.md": (
        "# USER.md - User Profile\n\n"
        "- **Name:**\n"
        "- **Preferred address:**\n"
        "- **Pronouns:**\n"
        "- **Timezone:**\n"
        "- **Notes:**\n"
    ),
    "HEARTBEAT.md": (
        "# HEARTBEAT.md\n\n"
        "# Keep this file empty (or with only comments) to skip heartbeat API calls.\n"
        "# Add tasks below when you want the agent to check something periodically.\n"
    ),
    "BOOTSTRAP.md": (
        "# BOOTSTRAP.md - Hello, World\n\n"
        "_You just woke up. Time to figure out who you are._\n\n"
        "Start with something like:\n"
        '> "Hey. I just came online. Who am I? Who are you?"\n\n'
        "Then figure out together:\n"
        "1. Your name\n"
        "2. Your nature\n"
        "3. Your vibe\n"
        "4. Your emoji\n\n"
        "When you're done, delete this file.\n"
    ),
}


def load_bootstrap_files(
    workspace_dir: str | Path,
    *,
    is_subagent: bool = False,
    is_main_session: bool = True,
    max_chars_per_file: int = 20_000,
) -> list[BootstrapFile]:
    """Load all bootstrap files from the workspace directory.

    Args:
        workspace_dir: Path to workspace.
        is_subagent: If True, only load AGENTS.md and TOOLS.md.
        is_main_session: If False, exclude MEMORY.md (security scoping).
    """
    d = Path(workspace_dir)
    names = list(BOOTSTRAP_FILES)
    if is_subagent:
        names = [n for n in names if n in _SUBAGENT_ALLOWLIST]
    elif not is_main_session:
        names = [n for n in names if n not in _MAIN_SESSION_ONLY]
    result: list[BootstrapFile] = []
    for name in names:
        p = d / name
        if p.is_file():
            content = p.read_text(encoding="utf-8")
            if len(content) > max_chars_per_file:
                content = content[:max_chars_per_file] + f"\n\n[... truncated at {max_chars_per_file} chars]"
            result.append(BootstrapFile(name=name, path=str(p), content=content))
        else:
            result.append(BootstrapFile(name=name, path=str(p), missing=True))
    return result


def ensure_workspace(workspace_dir: str | Path) -> Path:
    """Create workspace directory and write default templates for missing files."""
    d = Path(workspace_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    for name, content in _DEFAULT_TEMPLATES.items():
        p = d / name
        if not p.exists():
            p.write_text(content, encoding="utf-8")
    return d


def build_context_prompt(files: list[BootstrapFile]) -> str:
    """Build the '# Project Context' system prompt section from bootstrap files."""
    present = [f for f in files if not f.missing and f.content]
    if not present:
        return ""
    lines = ["# Project Context", "", "The following project context files have been loaded:", ""]
    has_soul = any(f.name.lower() == "soul.md" for f in present)
    if has_soul:
        lines.append(
            "If SOUL.md is present, embody its persona and tone. "
            "Avoid stiff, generic replies; follow its guidance "
            "unless higher-priority instructions override it."
        )
        lines.append("")
    for f in present:
        lines.extend([f"## {f.name}", "", f.content or "", ""])
    return "\n".join(lines)
