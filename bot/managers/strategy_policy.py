"""Interfaces for low-frequency macro strategy decisions."""
from __future__ import annotations

from typing import Any, Protocol

from rl.strategy_actions import StrategyAction


def write_strategy_decision_metadata(
    bot: Any,
    *,
    source: str,
    reason: str,
) -> None:
    """Attach lightweight explanation metadata to the active bot."""
    try:
        setattr(bot, "last_strategy_decision_source", source)
        setattr(bot, "last_strategy_decision_reason", reason)
    except (AttributeError, TypeError):
        return


class StrategyPolicy(Protocol):
    """Policy interface for macro strategy intents.

    StrategyPolicy deliberately does not replace ArmyPolicy. It runs at a lower
    frequency and returns macro intents that a rule executor can translate into
    safe build, tech, production, and defense steps.
    """

    def decide_strategy(self, bot: Any) -> StrategyAction:
        """Choose one high-level macro strategy action for the current state."""
