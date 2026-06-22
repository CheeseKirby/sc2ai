"""Teacher policy that deliberately emits broader labels for data collection."""
from __future__ import annotations

from typing import Any

from bot.managers.army_policy import ArmyAction
from bot.managers.rule_army_policy import (
    COMBAT_UNIT_TYPES,
    RuleArmyPolicy,
    base_under_threat,
)


DEFAULT_RETREAT_PEAK_LOSS_RATIO = 0.25
DEFAULT_RETREAT_MIN_PEAK_ARMY = 8
DEFAULT_RETREAT_MIN_LOST_FROM_PEAK = 3


class CoverageArmyPolicy(RuleArmyPolicy):
    """Heuristic teacher for pressure-data collection, not the default bot."""

    def manage_army(self, bot: Any) -> ArmyAction:
        """Choose a broader-coverage action, execute it, and return it."""
        army = bot.units.of_type(COMBAT_UNIT_TYPES)
        action = self.decide_for_coverage(bot, army)
        self.execute(bot, action, army)
        return action

    def decide_for_coverage(self, bot: Any, army: Any) -> ArmyAction:
        """Prefer explicit defense/hold/retreat labels when conditions appear."""
        army_count = int(getattr(army, "amount", 0))
        if army_count <= 0:
            bot.is_attacking = False
            return ArmyAction.HOLD

        if base_under_threat(bot):
            bot.is_attacking = False
            return ArmyAction.DEFEND_BASE

        if bot.is_attacking and _peak_loss_retreat(bot, army_count):
            bot.is_attacking = False
            return ArmyAction.RETREAT_HOME

        if bot.is_attacking and army_count <= bot.ARMY_RETREAT_THRESHOLD:
            bot.is_attacking = False
            return ArmyAction.RETREAT_HOME

        if bot.is_attacking and _idle_count(army) == 0:
            return ArmyAction.HOLD

        if not bot.is_attacking and army_count >= bot.ARMY_ATTACK_THRESHOLD:
            bot.is_attacking = True
            return ArmyAction.ATTACK_MAIN

        if bot.is_attacking:
            return ArmyAction.ATTACK_MAIN
        return ArmyAction.RALLY


def _idle_count(army: Any) -> int:
    idle = getattr(army, "idle", [])
    amount = getattr(idle, "amount", None)
    if amount is not None:
        return int(amount)
    try:
        return len(idle)
    except TypeError:
        return 0


def _peak_loss_retreat(bot: Any, army_count: int) -> bool:
    memory = getattr(bot, "army_memory", None)
    peak = int(getattr(memory, "attack_army_peak", 0))
    lost = int(getattr(memory, "army_lost_from_peak", max(peak - army_count, 0)))
    ratio = float(
        getattr(
            memory,
            "army_lost_from_peak_ratio",
            (lost / peak) if peak else 0.0,
        )
    )
    min_peak = int(getattr(bot, "RETREAT_MIN_PEAK_ARMY", DEFAULT_RETREAT_MIN_PEAK_ARMY))
    min_lost = int(
        getattr(bot, "RETREAT_MIN_LOST_FROM_PEAK", DEFAULT_RETREAT_MIN_LOST_FROM_PEAK)
    )
    min_ratio = float(
        getattr(bot, "RETREAT_PEAK_LOSS_RATIO", DEFAULT_RETREAT_PEAK_LOSS_RATIO)
    )
    return peak >= min_peak and lost >= min_lost and ratio >= min_ratio
