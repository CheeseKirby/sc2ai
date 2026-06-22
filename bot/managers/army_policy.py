"""Interfaces and action definitions for army-level decisions."""
from __future__ import annotations

from enum import IntEnum
from typing import Any, Protocol


class ArmyAction(IntEnum):
    """Discrete high-level army actions used by rule and future RL policies."""

    RALLY = 0
    ATTACK_MAIN = 1
    RETREAT_HOME = 2
    DEFEND_BASE = 3
    HOLD = 4


class ArmyPolicy(Protocol):
    """Policy interface for replacing army control without changing the bot."""

    def manage_army(self, bot: Any) -> ArmyAction:
        """Choose and execute one high-level army action for the current step."""

