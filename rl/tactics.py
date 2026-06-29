"""Tactic specifications for grouping macro strategy actions.

The tactic layer is metadata and policy scaffolding only. It does not change the
default rule/no-op strategy path unless a future strategy policy explicitly opts
into using these specs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rl.strategy_actions import StrategyAction


EARLY_GATEWAY_CUTOFF_SECONDS = 240.0
SAFE_MACRO_EARLY_GATEWAY_PENDING_CAP = 2
SAFE_MACRO_PRE_ROBO_GATEWAY_CUTOFF_SECONDS = 120.0
SAFE_MACRO_PRE_ROBO_GATEWAY_PENDING_CAP = 3
SAFE_MACRO_PRE_ROBO_GATEWAY_MINERALS = 250.0
GATEWAYS_PER_BASE_TARGET = 4
TECH_POWER_GATEWAY_SCALING_PENDING_CAP = 2
STATIC_DEFENSE_MINERALS = 100.0
INITIAL_ROBO_MINERALS = 150.0
INITIAL_ROBO_VESPENE = 100.0
FIRST_IMMORTAL_MINERALS = 275.0
FIRST_IMMORTAL_VESPENE = 100.0
FIRST_IMMORTAL_SUPPLY = 4.0
FIRST_IMMORTAL_BANK_MINERALS = 100.0


class TacticId(str, Enum):
    """Stable tactic identifiers for tactic-aware strategy work."""

    SAFE_MACRO = "SAFE_MACRO"
    ANTI_RUSH_DEFENSE = "ANTI_RUSH_DEFENSE"
    GATEWAY_PRESSURE = "GATEWAY_PRESSURE"
    ROBO_TIMING = "ROBO_TIMING"
    TECH_POWER = "TECH_POWER"
    ANTI_AIR_RESPONSE = "ANTI_AIR_RESPONSE"
    RECOVERY = "RECOVERY"


class TacticPhase(str, Enum):
    """Coarse phase labels for tactic state metadata."""

    OPENING = "OPENING"
    STABILIZE = "STABILIZE"
    POWER_SPIKE = "POWER_SPIKE"
    ATTACK_WINDOW = "ATTACK_WINDOW"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class TacticSpec:
    """Rule-level constraints for one tactic."""

    tactic_id: TacticId
    name: str
    allowed_strategy_actions: tuple[StrategyAction, ...]
    preferred_strategy_actions: tuple[StrategyAction, ...]
    avoid_strategy_actions: tuple[StrategyAction, ...] = ()
    opponent_ai_build_hints: tuple[str, ...] = ()
    min_game_time: float = 0.0
    max_game_time: float | None = None
    mineral_reserve: int = 0
    vespene_reserve: int = 0
    max_pending_gateways: int = 1
    max_pending_robo: int = 1
    max_pending_static_defense: int = 1
    attack_army_threshold_bias: int = 0
    expand_allowed_under_threat: bool = False
    transition_triggers: tuple[str, ...] = ()
    abort_triggers: tuple[str, ...] = ()


@dataclass(frozen=True)
class TacticState:
    """Current tactic state stored as metadata by future tactic-aware policies."""

    current_tactic: TacticId
    phase: TacticPhase
    started_game_time: float
    last_switch_game_time: float
    last_switch_reason: str
    previous_tactic: TacticId | None = None


DEFAULT_TACTIC_POOL: dict[TacticId, TacticSpec] = {
    TacticId.SAFE_MACRO: TacticSpec(
        tactic_id=TacticId.SAFE_MACRO,
        name="Safe macro",
        allowed_strategy_actions=(
            StrategyAction.STAY_COURSE,
            StrategyAction.BOOST_WORKERS,
            StrategyAction.EXPAND,
            StrategyAction.ADD_GATEWAYS,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.TECH_ROBO,
            StrategyAction.FORGE_UPGRADES,
        ),
        preferred_strategy_actions=(
            StrategyAction.BOOST_WORKERS,
            StrategyAction.ADD_GATEWAYS,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.EXPAND,
        ),
        avoid_strategy_actions=(StrategyAction.BUILD_STATIC_DEFENSE,),
        opponent_ai_build_hints=("RandomBuild", "Macro"),
        mineral_reserve=100,
        vespene_reserve=0,
        transition_triggers=("enemy_threat", "tech_signal", "air_signal"),
    ),
    TacticId.ANTI_RUSH_DEFENSE: TacticSpec(
        tactic_id=TacticId.ANTI_RUSH_DEFENSE,
        name="Anti-rush defense",
        allowed_strategy_actions=(
            StrategyAction.STAY_COURSE,
            StrategyAction.EXPAND,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.BUILD_STATIC_DEFENSE,
            StrategyAction.ADD_GATEWAYS,
            StrategyAction.BOOST_WORKERS,
            StrategyAction.TECH_ROBO,
        ),
        preferred_strategy_actions=(
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.BUILD_STATIC_DEFENSE,
            StrategyAction.ADD_GATEWAYS,
        ),
        avoid_strategy_actions=(
            StrategyAction.FORGE_UPGRADES,
        ),
        opponent_ai_build_hints=("Rush",),
        mineral_reserve=100,
        max_pending_static_defense=1,
        expand_allowed_under_threat=False,
        transition_triggers=("rush_build", "early_base_threat"),
        abort_triggers=("stabilized_economy",),
    ),
    TacticId.GATEWAY_PRESSURE: TacticSpec(
        tactic_id=TacticId.GATEWAY_PRESSURE,
        name="Gateway pressure",
        allowed_strategy_actions=(
            StrategyAction.STAY_COURSE,
            StrategyAction.EXPAND,
            StrategyAction.ADD_GATEWAYS,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.BOOST_WORKERS,
            StrategyAction.TECH_ROBO,
        ),
        preferred_strategy_actions=(
            StrategyAction.ADD_GATEWAYS,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.BOOST_WORKERS,
        ),
        avoid_strategy_actions=(
            StrategyAction.FORGE_UPGRADES,
            StrategyAction.BUILD_STATIC_DEFENSE,
        ),
        opponent_ai_build_hints=("Macro",),
        mineral_reserve=100,
        max_pending_gateways=2,
        transition_triggers=("opponent_macro", "army_ready"),
        abort_triggers=("base_under_threat", "tech_signal"),
    ),
    TacticId.ROBO_TIMING: TacticSpec(
        tactic_id=TacticId.ROBO_TIMING,
        name="Robo timing",
        allowed_strategy_actions=(
            StrategyAction.STAY_COURSE,
            StrategyAction.EXPAND,
            StrategyAction.TECH_ROBO,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.ADD_GATEWAYS,
            StrategyAction.BOOST_WORKERS,
            StrategyAction.BUILD_STATIC_DEFENSE,
        ),
        preferred_strategy_actions=(
            StrategyAction.TECH_ROBO,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.ADD_GATEWAYS,
        ),
        avoid_strategy_actions=(
            StrategyAction.FORGE_UPGRADES,
        ),
        opponent_ai_build_hints=("Timing", "Power", "Air"),
        mineral_reserve=100,
        vespene_reserve=100,
        max_pending_robo=1,
        transition_triggers=("armored_signal", "cloaked_signal", "midgame_timing"),
        abort_triggers=("robo_established",),
    ),
    TacticId.TECH_POWER: TacticSpec(
        tactic_id=TacticId.TECH_POWER,
        name="Tech power",
        allowed_strategy_actions=(
            StrategyAction.STAY_COURSE,
            StrategyAction.EXPAND,
            StrategyAction.FORGE_UPGRADES,
            StrategyAction.TECH_ROBO,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.ADD_GATEWAYS,
            StrategyAction.BOOST_WORKERS,
        ),
        preferred_strategy_actions=(
            StrategyAction.TECH_ROBO,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.ADD_GATEWAYS,
            StrategyAction.FORGE_UPGRADES,
        ),
        avoid_strategy_actions=(
            StrategyAction.BUILD_STATIC_DEFENSE,
        ),
        opponent_ai_build_hints=("Power",),
        min_game_time=240.0,
        mineral_reserve=150,
        vespene_reserve=100,
        max_pending_robo=1,
        transition_triggers=("power_build", "upgrade_window"),
        abort_triggers=("base_under_threat", "economy_damaged"),
    ),
    TacticId.ANTI_AIR_RESPONSE: TacticSpec(
        tactic_id=TacticId.ANTI_AIR_RESPONSE,
        name="Anti-air response",
        allowed_strategy_actions=(
            StrategyAction.STAY_COURSE,
            StrategyAction.EXPAND,
            StrategyAction.TECH_ROBO,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.BUILD_STATIC_DEFENSE,
            StrategyAction.ADD_GATEWAYS,
            StrategyAction.BOOST_WORKERS,
        ),
        preferred_strategy_actions=(
            StrategyAction.TECH_ROBO,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.BUILD_STATIC_DEFENSE,
        ),
        avoid_strategy_actions=(
            StrategyAction.FORGE_UPGRADES,
        ),
        opponent_ai_build_hints=("Air",),
        mineral_reserve=100,
        vespene_reserve=100,
        max_pending_robo=1,
        max_pending_static_defense=1,
        transition_triggers=("air_build", "air_signal"),
        abort_triggers=("air_threat_cleared",),
    ),
    TacticId.RECOVERY: TacticSpec(
        tactic_id=TacticId.RECOVERY,
        name="Recovery",
        allowed_strategy_actions=(
            StrategyAction.STAY_COURSE,
            StrategyAction.BOOST_WORKERS,
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.BUILD_STATIC_DEFENSE,
            StrategyAction.ADD_GATEWAYS,
        ),
        preferred_strategy_actions=(
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.BUILD_STATIC_DEFENSE,
            StrategyAction.BOOST_WORKERS,
        ),
        avoid_strategy_actions=(
            StrategyAction.EXPAND,
            StrategyAction.FORGE_UPGRADES,
            StrategyAction.TECH_ROBO,
        ),
        opponent_ai_build_hints=(),
        mineral_reserve=50,
        max_pending_static_defense=1,
        transition_triggers=("economy_damaged", "army_collapsed"),
        abort_triggers=("stabilized_economy",),
    ),
}


def get_tactic_spec(tactic_id: TacticId) -> TacticSpec:
    """Return the configured spec for ``tactic_id``."""
    return DEFAULT_TACTIC_POOL[tactic_id]


def filter_strategy_action(
    spec: TacticSpec,
    proposed_action: StrategyAction,
    observation: dict[str, float],
) -> StrategyAction:
    """Conservatively filter a proposed StrategyAction through a tactic spec."""
    static_defense_retention = _static_defense_retention(
        spec,
        proposed_action,
        observation,
    )
    if static_defense_retention is not None:
        return static_defense_retention
    initial_robo_precedence = _initial_robo_precedence(
        spec,
        proposed_action,
        observation,
    )
    if initial_robo_precedence is not None:
        return initial_robo_precedence
    first_immortal_bias = _first_immortal_bias(spec, proposed_action, observation)
    if first_immortal_bias is not None:
        return first_immortal_bias
    if _action_passes(spec, proposed_action, observation):
        return proposed_action
    if proposed_action is StrategyAction.TECH_ROBO and _should_save_for_initial_robo(
        spec,
        observation,
    ):
        return StrategyAction.STAY_COURSE
    for fallback in _fallback_candidates(spec, proposed_action, observation):
        if _action_passes(spec, fallback, observation):
            return fallback
    return StrategyAction.STAY_COURSE


def _fallback_candidates(
    spec: TacticSpec,
    proposed_action: StrategyAction,
    observation: dict[str, float],
) -> tuple[StrategyAction, ...]:
    if (
        spec.tactic_id is TacticId.TECH_POWER
        and proposed_action is StrategyAction.BUILD_STATIC_DEFENSE
        and _value(observation, "base_under_threat") > 0.0
    ):
        return (
            StrategyAction.PRODUCE_ARMY,
            StrategyAction.TECH_ROBO,
            StrategyAction.ADD_GATEWAYS,
            StrategyAction.FORGE_UPGRADES,
            StrategyAction.BOOST_WORKERS,
        )
    return spec.preferred_strategy_actions


def _static_defense_retention(
    spec: TacticSpec,
    proposed_action: StrategyAction,
    observation: dict[str, float],
) -> StrategyAction | None:
    if (
        not _can_retain_static_defense(spec)
        or proposed_action is not StrategyAction.BUILD_STATIC_DEFENSE
        or _value(observation, "base_under_threat") <= 0.0
    ):
        return None
    if (
        _value(observation, "pending_static_defense")
        >= float(spec.max_pending_static_defense)
    ):
        if _value(observation, "ready_static_defense") <= 0.0:
            return StrategyAction.STAY_COURSE
        return None
    if _value(observation, "minerals") >= STATIC_DEFENSE_MINERALS:
        return StrategyAction.BUILD_STATIC_DEFENSE
    if _anti_air_should_bank_with_ready_static(spec, observation):
        return StrategyAction.STAY_COURSE
    if (
        spec.tactic_id is not TacticId.TECH_POWER
        and _value(observation, "ready_static_defense") > 0.0
    ):
        return None
    return StrategyAction.STAY_COURSE


def _anti_air_should_bank_with_ready_static(
    spec: TacticSpec,
    observation: dict[str, float],
) -> bool:
    return (
        spec.tactic_id is TacticId.ANTI_AIR_RESPONSE
        and _value(observation, "base_under_air_threat") > 0.0
        and _value(observation, "ready_static_defense") > 0.0
        and _value(observation, "pending_static_defense") <= 0.0
        and _value(observation, "minerals") < STATIC_DEFENSE_MINERALS
    )


def _can_retain_static_defense(spec: TacticSpec) -> bool:
    return (
        spec.tactic_id is TacticId.TECH_POWER
        or (
            StrategyAction.BUILD_STATIC_DEFENSE in spec.allowed_strategy_actions
            and StrategyAction.BUILD_STATIC_DEFENSE not in spec.avoid_strategy_actions
        )
    )


def _initial_robo_precedence(
    spec: TacticSpec,
    proposed_action: StrategyAction,
    observation: dict[str, float],
) -> StrategyAction | None:
    if spec.tactic_id is not TacticId.TECH_POWER or not _needs_initial_robo(
        spec,
        observation,
    ):
        return None
    if not _action_can_delay_initial_robo(proposed_action):
        return None
    if _can_start_initial_robo(spec, observation):
        return StrategyAction.TECH_ROBO
    if _should_save_for_initial_robo(spec, observation):
        return StrategyAction.STAY_COURSE
    return None


def _needs_initial_robo(
    spec: TacticSpec,
    observation: dict[str, float],
) -> bool:
    game_time = _value(observation, "game_time")
    if spec.max_game_time is not None and game_time > spec.max_game_time:
        return False
    return (
        spec.tactic_id
        in {TacticId.ROBO_TIMING, TacticId.TECH_POWER, TacticId.ANTI_AIR_RESPONSE}
        and StrategyAction.TECH_ROBO in spec.allowed_strategy_actions
        and StrategyAction.TECH_ROBO not in spec.avoid_strategy_actions
        and game_time >= spec.min_game_time
        and _value(observation, "has_cybernetics_core") > 0.0
        and _value(observation, "ready_robo") <= 0.0
        and _value(observation, "pending_robo") <= 0.0
        and _value(observation, "base_under_threat") <= 0.0
    )


def _can_start_initial_robo(
    spec: TacticSpec,
    observation: dict[str, float],
) -> bool:
    return (
        _action_passes(spec, StrategyAction.TECH_ROBO, observation)
        and _value(observation, "minerals") >= INITIAL_ROBO_MINERALS
        and _value(observation, "vespene") >= INITIAL_ROBO_VESPENE
    )


def _action_passes(
    spec: TacticSpec,
    action: StrategyAction,
    observation: dict[str, float],
) -> bool:
    if action not in spec.allowed_strategy_actions:
        return False
    if action in spec.avoid_strategy_actions:
        return False
    game_time = _value(observation, "game_time")
    if game_time < spec.min_game_time:
        return False
    if spec.max_game_time is not None and game_time > spec.max_game_time:
        return False
    if (
        action is StrategyAction.EXPAND
        and _value(observation, "base_under_threat") > 0.0
        and not spec.expand_allowed_under_threat
    ):
        return False
    if (
        action is StrategyAction.ADD_GATEWAYS
        and _value(observation, "pending_gateways")
        >= float(_max_pending_gateways(spec, observation))
    ):
        return False
    if (
        action is StrategyAction.TECH_ROBO
        and _value(observation, "pending_robo") >= float(spec.max_pending_robo)
    ):
        return False
    if (
        spec.tactic_id is TacticId.TECH_POWER
        and action is StrategyAction.TECH_ROBO
        and _value(observation, "ready_robo") > 0.0
    ):
        return False
    if (
        action is StrategyAction.BUILD_STATIC_DEFENSE
        and _value(observation, "pending_static_defense")
        >= float(spec.max_pending_static_defense)
    ):
        return False
    if _spending_action(action):
        if _value(observation, "minerals") < float(spec.mineral_reserve):
            return False
        if _value(observation, "vespene") < float(spec.vespene_reserve):
            return False
    return True


def _spending_action(action: StrategyAction) -> bool:
    return action in {
        StrategyAction.EXPAND,
        StrategyAction.ADD_GATEWAYS,
        StrategyAction.TECH_ROBO,
        StrategyAction.FORGE_UPGRADES,
        StrategyAction.BUILD_STATIC_DEFENSE,
    }


def _max_pending_gateways(
    spec: TacticSpec,
    observation: dict[str, float],
) -> int:
    if (
        spec.tactic_id is TacticId.SAFE_MACRO
        and _value(observation, "game_time") < EARLY_GATEWAY_CUTOFF_SECONDS
    ):
        if _safe_macro_pre_robo_gateway_preservation(observation):
            return max(
                spec.max_pending_gateways,
                SAFE_MACRO_PRE_ROBO_GATEWAY_PENDING_CAP,
            )
        return max(spec.max_pending_gateways, SAFE_MACRO_EARLY_GATEWAY_PENDING_CAP)
    if (
        spec.tactic_id is TacticId.TECH_POWER
        and _value(observation, "base_under_threat") <= 0.0
        and (
            _value(observation, "ready_robo") > 0.0
            or _value(observation, "pending_robo") > 0.0
        )
        and _gateway_scaling_needed(observation)
    ):
        return max(spec.max_pending_gateways, TECH_POWER_GATEWAY_SCALING_PENDING_CAP)
    return spec.max_pending_gateways


def _safe_macro_pre_robo_gateway_preservation(
    observation: dict[str, float],
) -> bool:
    return (
        _value(observation, "game_time") < SAFE_MACRO_PRE_ROBO_GATEWAY_CUTOFF_SECONDS
        and _value(observation, "ready_gateways") <= 0.0
        and _value(observation, "pending_gateways")
        >= float(SAFE_MACRO_EARLY_GATEWAY_PENDING_CAP)
        and _value(observation, "minerals") >= SAFE_MACRO_PRE_ROBO_GATEWAY_MINERALS
        and _value(observation, "vespene") < INITIAL_ROBO_VESPENE
        and _value(observation, "base_under_threat") <= 0.0
    )


def _should_save_for_initial_robo(
    spec: TacticSpec,
    observation: dict[str, float],
) -> bool:
    return (
        _needs_initial_robo(spec, observation)
        and _value(observation, "minerals") < INITIAL_ROBO_MINERALS
        and _value(observation, "vespene") >= INITIAL_ROBO_VESPENE
    )


def _action_can_delay_initial_robo(action: StrategyAction) -> bool:
    return action in {
        StrategyAction.STAY_COURSE,
        StrategyAction.ADD_GATEWAYS,
        StrategyAction.TECH_ROBO,
        StrategyAction.FORGE_UPGRADES,
        StrategyAction.BUILD_STATIC_DEFENSE,
        StrategyAction.PRODUCE_ARMY,
        StrategyAction.BOOST_WORKERS,
    }


def _first_immortal_bias(
    spec: TacticSpec,
    proposed_action: StrategyAction,
    observation: dict[str, float],
) -> StrategyAction | None:
    if spec.tactic_id not in {
        TacticId.ROBO_TIMING,
        TacticId.TECH_POWER,
        TacticId.ANTI_AIR_RESPONSE,
    }:
        return None
    if (
        _value(observation, "ready_robo") <= 0.0
        or _value(observation, "observers") <= 0.0
        or _value(observation, "immortals") > 0.0
        or _value(observation, "base_under_threat") > 0.0
        or _value(observation, "vespene") < FIRST_IMMORTAL_VESPENE
    ):
        return None
    if (
        proposed_action is StrategyAction.ADD_GATEWAYS
        and _gateway_scaling_needed(observation)
        and _value(observation, "pending_gateways")
        < float(_max_pending_gateways(spec, observation))
    ):
        return None

    minerals = _value(observation, "minerals")
    supply_left = _value(observation, "supply_left")
    if minerals >= FIRST_IMMORTAL_MINERALS:
        if supply_left >= FIRST_IMMORTAL_SUPPLY:
            return StrategyAction.PRODUCE_ARMY
        if _action_can_delay_first_immortal(proposed_action):
            return StrategyAction.STAY_COURSE
        return None

    if (
        minerals >= FIRST_IMMORTAL_BANK_MINERALS
        and supply_left >= FIRST_IMMORTAL_SUPPLY
        and _action_can_delay_first_immortal(proposed_action)
    ):
        return StrategyAction.STAY_COURSE
    return None


def _action_can_delay_first_immortal(action: StrategyAction) -> bool:
    return action in {
        StrategyAction.EXPAND,
        StrategyAction.ADD_GATEWAYS,
        StrategyAction.TECH_ROBO,
        StrategyAction.FORGE_UPGRADES,
        StrategyAction.BUILD_STATIC_DEFENSE,
        StrategyAction.PRODUCE_ARMY,
        StrategyAction.BOOST_WORKERS,
    }


def _gateway_scaling_needed(observation: dict[str, float]) -> bool:
    bases = max(_value(observation, "own_bases"), 1.0)
    target_gateways = bases * GATEWAYS_PER_BASE_TARGET
    effective_gateways = _value(observation, "ready_gateways") + _value(
        observation,
        "pending_gateways",
    )
    return effective_gateways < target_gateways


def _value(observation: dict[str, float], field: str) -> float:
    return float(observation.get(field, 0.0))
