"""Coverage-oriented macro strategy teacher for data collection."""
from __future__ import annotations

from typing import Any

from bot.managers.strategy_policy import write_strategy_decision_metadata
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import (
    build_strategy_observation,
    validate_strategy_observation_dict,
)

STRATEGY_TEACHER_PROFILES = ("standard", "pre-collapse-recovery")


class CoverageStrategyPolicy:
    """Rule teacher that emits diverse, explainable macro strategy labels.

    This policy is intended for opt-in data collection. The default strategy
    policy remains RuleStrategyPolicy, which returns STAY_COURSE.
    """

    target_bases: int = 2
    gateways_per_base: int = 4
    static_defense_per_base: int = 2
    worker_saturation_floor: float = 0.75
    expand_saturation_floor: float = 0.85
    expand_minerals: float = 400.0
    gateway_minerals: float = 150.0
    robo_minerals: float = 150.0
    robo_vespene: float = 100.0
    forge_minerals: float = 150.0
    forge_min_gateways: int = 4
    static_defense_minerals: float = 100.0
    photon_cannon_minerals: float = 150.0
    midgame_time: float = 360.0
    pre_collapse_robo_time: float = 540.0
    pre_collapse_robo_vespene_bank: float = 500.0
    pre_collapse_static_defense_time: float = 420.0
    produce_minerals: float = 100.0
    desired_min_army: float = 10.0
    last_decision_source: str = "coverage-teacher"
    last_decision_reason: str = "uninitialized"

    def __init__(self, *, teacher_profile: str = "standard") -> None:
        if teacher_profile not in STRATEGY_TEACHER_PROFILES:
            choices = ", ".join(STRATEGY_TEACHER_PROFILES)
            raise ValueError(
                f"Unsupported strategy teacher profile: {teacher_profile!r}. "
                f"Expected one of: {choices}"
            )
        self.teacher_profile = teacher_profile
        self.last_decision_source = self._decision_source()
        self.last_decision_reason = "uninitialized"

    def decide_strategy(self, bot: Any) -> StrategyAction:
        """Choose one macro strategy label from the current strategy observation."""
        observation = build_strategy_observation(bot).to_dict()
        action = self.decide_from_observation(observation)
        write_strategy_decision_metadata(
            bot,
            source=self.last_decision_source,
            reason=self.last_decision_reason,
        )
        return action

    def decide_from_observation(
        self,
        observation: dict[str, float],
    ) -> StrategyAction:
        """Choose one macro strategy label from a strategy observation dict."""
        validate_strategy_observation_dict(observation)
        bases = max(_value(observation, "own_bases"), 1.0)
        effective_bases = bases + _value(observation, "pending_bases")
        gateway_count = (
            _value(observation, "ready_gateways")
            + _value(observation, "pending_gateways")
        )
        robo_count = (
            _value(observation, "ready_robo")
            + _value(observation, "pending_robo")
        )
        forge_count = (
            _value(observation, "ready_forge")
            + _value(observation, "pending_forge")
        )
        static_defense_count = (
            _value(observation, "ready_static_defense")
            + _value(observation, "pending_static_defense")
        )
        target_gateways = bases * float(self.gateways_per_base)
        target_static_defense = bases * float(self.static_defense_per_base)

        if (
            _value(observation, "base_under_threat") > 0.0
            and static_defense_count < target_static_defense
            and _value(observation, "minerals") >= self.static_defense_minerals
        ):
            return self._decision(
                StrategyAction.BUILD_STATIC_DEFENSE,
                "base_threat_static_defense_gap",
            )

        if (
            _value(observation, "worker_saturation_ratio")
            < self.worker_saturation_floor
            and _value(observation, "supply_left") > 0.0
            and _value(observation, "minerals") >= 50.0
            and _value(observation, "base_under_threat") <= 0.0
        ):
            return self._decision(
                StrategyAction.BOOST_WORKERS,
                "worker_saturation_below_floor",
            )

        if (
            effective_bases < float(self.target_bases)
            and _value(observation, "minerals") >= self.expand_minerals
            and _value(observation, "worker_saturation_ratio")
            >= self.expand_saturation_floor
            and _value(observation, "base_under_threat") <= 0.0
        ):
            return self._decision(
                StrategyAction.EXPAND,
                "safe_saturated_expand",
            )

        urgent_robo_signal = _value(observation, "enemy_cloaked_units_seen") > 0.0
        robo_needed = (
            _value(observation, "has_cybernetics_core") > 0.0
            and robo_count <= 0.0
            and _value(observation, "minerals") >= self.robo_minerals
            and _value(observation, "vespene") >= self.robo_vespene
            and (
                _value(observation, "enemy_armored_units_known") > 0.0
                or _value(observation, "enemy_cloaked_units_seen") > 0.0
                or _value(observation, "game_time") >= self.midgame_time
                or _value(observation, "army_count") >= self.desired_min_army
            )
        )
        if urgent_robo_signal and robo_needed:
            return self._decision(
                StrategyAction.TECH_ROBO,
                "urgent_cloak_detection_robo",
            )

        pre_collapse_action = self._pre_collapse_recovery_decision(
            observation,
            bases=bases,
            robo_count=robo_count,
            static_defense_count=static_defense_count,
        )
        if pre_collapse_action is not None:
            return pre_collapse_action

        forge_upgrade_incomplete = (
            _value(observation, "ground_weapon_level") < 1.0
            or _value(observation, "ground_armor_level") < 1.0
        )
        forge_upgrade_pending = (
            _value(observation, "ground_weapon_upgrade_pending") > 0.0
            or _value(observation, "ground_armor_upgrade_pending") > 0.0
        )
        if (
            _value(observation, "minerals") >= self.forge_minerals
            and _value(observation, "game_time") >= self.midgame_time
            and gateway_count >= float(self.forge_min_gateways)
            and (
                forge_count <= 0.0
                or (
                    _value(observation, "ready_forge") > 0.0
                    and forge_upgrade_incomplete
                    and not forge_upgrade_pending
                )
            )
        ):
            return self._decision(
                StrategyAction.FORGE_UPGRADES,
                "midgame_forge_upgrade_gap",
            )

        if (
            _value(observation, "game_time") >= self.midgame_time
            and _value(observation, "ready_forge") > 0.0
            and static_defense_count < bases
            and _value(observation, "minerals") >= self.photon_cannon_minerals
        ):
            return self._decision(
                StrategyAction.BUILD_STATIC_DEFENSE,
                "midgame_static_defense_floor",
            )

        if (
            gateway_count < target_gateways
            and _value(observation, "minerals") >= self.gateway_minerals
        ):
            return self._decision(
                StrategyAction.ADD_GATEWAYS,
                "gateway_count_below_target",
            )

        if robo_needed:
            return self._decision(
                StrategyAction.TECH_ROBO,
                "robo_needed_for_enemy_or_midgame_signal",
            )

        idle_production = (
            _value(observation, "gateway_idle_count")
            + _value(observation, "robo_idle_count")
        )
        if (
            _value(observation, "supply_left") > 0.0
            and _value(observation, "minerals") >= self.produce_minerals
            and (
                idle_production > 0.0
                or _value(observation, "army_count") < self.desired_min_army
            )
        ):
            return self._decision(
                StrategyAction.PRODUCE_ARMY,
                "idle_or_underbuilt_army_production",
            )

        return self._decision(
            StrategyAction.STAY_COURSE,
            "no_strategy_rule_triggered",
        )

    def _decision(
        self,
        action: StrategyAction,
        reason: str,
    ) -> StrategyAction:
        self.last_decision_source = self._decision_source()
        self.last_decision_reason = reason
        return action

    def _decision_source(self) -> str:
        if self.teacher_profile == "standard":
            return "coverage-teacher"
        return f"coverage-teacher:{self.teacher_profile}"

    def _pre_collapse_recovery_decision(
        self,
        observation: dict[str, float],
        *,
        bases: float,
        robo_count: float,
        static_defense_count: float,
    ) -> StrategyAction | None:
        if self.teacher_profile != "pre-collapse-recovery":
            return None

        if (
            _value(observation, "game_time") >= self.pre_collapse_robo_time
            and _value(observation, "base_under_threat") <= 0.0
            and _value(observation, "has_cybernetics_core") > 0.0
            and robo_count <= 0.0
            and _value(observation, "minerals") >= self.robo_minerals
            and _value(observation, "vespene")
            >= self.pre_collapse_robo_vespene_bank
        ):
            return self._decision(
                StrategyAction.TECH_ROBO,
                "pre_collapse_high_vespene_no_robo",
            )

        can_build_cannon = (
            _value(observation, "ready_forge") > 0.0
            and _value(observation, "minerals") >= self.photon_cannon_minerals
        )
        can_build_battery = (
            _value(observation, "has_cybernetics_core") > 0.0
            and _value(observation, "minerals") >= self.static_defense_minerals
        )
        if (
            _value(observation, "game_time")
            >= self.pre_collapse_static_defense_time
            and _value(observation, "base_under_threat") <= 0.0
            and static_defense_count < bases
            and (can_build_cannon or can_build_battery)
        ):
            return self._decision(
                StrategyAction.BUILD_STATIC_DEFENSE,
                "pre_collapse_static_defense_floor",
            )

        return None


def _value(observation: dict[str, float], field: str) -> float:
    return float(observation.get(field, 0.0))
