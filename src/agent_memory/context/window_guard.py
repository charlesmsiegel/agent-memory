"""Context window guard — resolve effective context window and enforce limits."""

from __future__ import annotations

import logging
from enum import Enum
from typing import NamedTuple

log = logging.getLogger(__name__)

HARD_MIN_TOKENS = 16_000
WARN_BELOW_TOKENS = 32_000


class ContextWindowSource(str, Enum):
    MODEL = "model"
    CONFIG = "config"
    DEFAULT = "default"


class ContextWindowInfo(NamedTuple):
    tokens: int
    source: ContextWindowSource


class ContextWindowGuardResult(NamedTuple):
    tokens: int
    source: ContextWindowSource
    should_warn: bool
    should_block: bool


def resolve_context_window(
    *,
    model_context_window: int | None = None,
    config_context_tokens: int | None = None,
    default_tokens: int = 100_000,
) -> ContextWindowInfo:
    """Resolve effective context window from model metadata / config / defaults.

    Config cap takes precedence if it's smaller than the model's window.
    """
    if model_context_window and model_context_window > 0:
        base = ContextWindowInfo(tokens=model_context_window, source=ContextWindowSource.MODEL)
    else:
        base = ContextWindowInfo(tokens=default_tokens, source=ContextWindowSource.DEFAULT)

    if config_context_tokens and 0 < config_context_tokens < base.tokens:
        return ContextWindowInfo(tokens=config_context_tokens, source=ContextWindowSource.CONFIG)

    return base


def evaluate_guard(
    info: ContextWindowInfo,
    *,
    hard_min: int = HARD_MIN_TOKENS,
    warn_below: int = WARN_BELOW_TOKENS,
) -> ContextWindowGuardResult:
    """Evaluate whether the context window is safe to use."""
    should_block = info.tokens < hard_min
    should_warn = info.tokens < warn_below and not should_block

    if should_block:
        log.error(
            "Context window %d tokens is below hard minimum %d. Blocking.",
            info.tokens,
            hard_min,
        )
    elif should_warn:
        log.warning(
            "Context window %d tokens is below recommended %d.",
            info.tokens,
            warn_below,
        )

    return ContextWindowGuardResult(
        tokens=info.tokens,
        source=info.source,
        should_warn=should_warn,
        should_block=should_block,
    )
