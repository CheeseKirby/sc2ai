"""Rule-based army policy matching the original ProtossRuleBot behavior."""
from __future__ import annotations

from typing import Any

from sc2.ids.unit_typeid import UnitTypeId

from bot.managers.army_policy import ArmyAction


COMBAT_UNIT_TYPES = frozenset({UnitTypeId.ZEALOT, UnitTypeId.STALKER})
DEFAULT_BASE_THREAT_RADIUS = 25.0


class RuleArmyPolicy:
    """Current scripted army logic, isolated behind the ArmyPolicy interface."""

    def manage_army(self, bot: Any) -> ArmyAction:
        """Choose an action, execute it, and return it for logging/training."""
        army = bot.units.of_type(COMBAT_UNIT_TYPES)
        action = self.decide(bot, army.amount)
        self.execute(bot, action, army)
        return action

    def decide(self, bot: Any, army_count: int) -> ArmyAction:
        """Replicate the original attack/retreat threshold state machine."""
        if bot.is_attacking and army_count <= bot.ARMY_RETREAT_THRESHOLD:
            bot.is_attacking = False
            return ArmyAction.RETREAT_HOME

        if not bot.is_attacking and army_count >= bot.ARMY_ATTACK_THRESHOLD:
            bot.is_attacking = True

        if bot.is_attacking:
            return ArmyAction.ATTACK_MAIN
        return ArmyAction.RALLY

    def execute(self, bot: Any, action: ArmyAction, army: Any) -> None:
        """Issue unit commands for the selected high-level action."""
        if action is ArmyAction.ATTACK_MAIN:
            target = bot.enemy_start_locations[0]
            for unit in army.idle:
                unit.attack(target)
            return

        if action is ArmyAction.DEFEND_BASE:
            threat = home_threat_target(bot)
            if threat is not None:
                for unit in army.idle:
                    unit.attack(threat)
                return
            self._move_to_rally(bot, army)
            return

        if action in {ArmyAction.RALLY, ArmyAction.RETREAT_HOME}:
            self._move_to_rally(bot, army)
            return

        if action is ArmyAction.HOLD:
            return

        raise ValueError(f"Unsupported army action: {action!r}")

    def _move_to_rally(self, bot: Any, army: Any) -> None:
        """Move idle army units toward the default home rally point."""
        if not bot.townhalls.exists:
            return
        rally = bot.townhalls.first.position.towards(
            bot.game_info.map_center, 8
        )
        for unit in army.idle:
            unit.move(rally)


def base_under_threat(bot: Any) -> bool:
    """Return whether known enemy units are close to any townhall."""
    return home_threat_target(bot) is not None


def home_threat_target(bot: Any) -> Any | None:
    """Return a known enemy unit near home, if one is visible."""
    if not bot.townhalls.exists or not getattr(bot.enemy_units, "exists", False):
        return None
    if not hasattr(bot.enemy_units, "closer_than"):
        return None
    radius = float(getattr(bot, "BASE_THREAT_RADIUS", DEFAULT_BASE_THREAT_RADIUS))
    for townhall in bot.townhalls:
        threats = bot.enemy_units.closer_than(radius, townhall)
        if not getattr(threats, "exists", False):
            continue
        if hasattr(threats, "closest_to"):
            return threats.closest_to(townhall)
        if hasattr(threats, "first"):
            return threats.first
    return None
