from __future__ import annotations

import pytest

from bot.managers.tactic_selector import RuleTacticSelector
from rl.strategy_actions import StrategyAction
from rl.tactics import (
    DEFAULT_TACTIC_POOL,
    TacticId,
    TacticPhase,
    TacticState,
    filter_strategy_action,
    get_tactic_spec,
)


def _observation(**overrides: float) -> dict[str, float]:
    observation = {
        "game_time": 120.0,
        "minerals": 500.0,
        "vespene": 150.0,
        "workers": 22.0,
        "own_bases": 1.0,
        "pending_bases": 0.0,
        "ready_gateways": 4.0,
        "pending_gateways": 0.0,
        "ready_robo": 0.0,
        "pending_robo": 0.0,
        "ready_forge": 0.0,
        "pending_forge": 0.0,
        "ready_static_defense": 0.0,
        "pending_static_defense": 0.0,
        "has_cybernetics_core": 1.0,
        "observers": 0.0,
        "immortals": 0.0,
        "army_count": 10.0,
        "enemy_air_units_known": 0.0,
        "enemy_armored_units_known": 0.0,
        "enemy_cloaked_units_seen": 0.0,
        "worker_saturation_ratio": 1.0,
        "base_under_air_threat": 0.0,
        "base_under_ground_threat": 0.0,
        "base_under_threat": 0.0,
    }
    observation.update(overrides)
    return observation


@pytest.mark.unit
def test_default_tactic_pool_covers_first_version_tactics() -> None:
    assert set(DEFAULT_TACTIC_POOL) == {
        TacticId.SAFE_MACRO,
        TacticId.ANTI_RUSH_DEFENSE,
        TacticId.GATEWAY_PRESSURE,
        TacticId.ROBO_TIMING,
        TacticId.TECH_POWER,
        TacticId.ANTI_AIR_RESPONSE,
        TacticId.RECOVERY,
    }
    for spec in DEFAULT_TACTIC_POOL.values():
        assert spec.allowed_strategy_actions
        assert spec.preferred_strategy_actions
        assert StrategyAction.STAY_COURSE in spec.allowed_strategy_actions


@pytest.mark.unit
def test_tactic_specs_encode_aibuild_hints_without_expanding_action_space() -> None:
    rush = get_tactic_spec(TacticId.ANTI_RUSH_DEFENSE)
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert rush.opponent_ai_build_hints == ("Rush",)
    assert StrategyAction.EXPAND in rush.allowed_strategy_actions
    assert StrategyAction.EXPAND not in rush.avoid_strategy_actions
    assert rush.preferred_strategy_actions[:3] == (
        StrategyAction.PRODUCE_ARMY,
        StrategyAction.BUILD_STATIC_DEFENSE,
        StrategyAction.ADD_GATEWAYS,
    )
    assert power.opponent_ai_build_hints == ("Power",)
    assert StrategyAction.BUILD_STATIC_DEFENSE in power.avoid_strategy_actions


@pytest.mark.unit
def test_filter_strategy_action_replaces_disallowed_or_repeated_actions() -> None:
    anti_rush = get_tactic_spec(TacticId.ANTI_RUSH_DEFENSE)
    robo = get_tactic_spec(TacticId.ROBO_TIMING)

    assert (
        filter_strategy_action(
            anti_rush,
            StrategyAction.EXPAND,
            _observation(base_under_threat=1.0),
        )
        is StrategyAction.PRODUCE_ARMY
    )
    assert (
        filter_strategy_action(
            robo,
            StrategyAction.TECH_ROBO,
            _observation(pending_robo=1.0),
        )
        is StrategyAction.PRODUCE_ARMY
    )


@pytest.mark.unit
def test_filter_strategy_action_preserves_safe_teacher_expands() -> None:
    for tactic_id in (
        TacticId.ANTI_RUSH_DEFENSE,
        TacticId.GATEWAY_PRESSURE,
        TacticId.ROBO_TIMING,
        TacticId.TECH_POWER,
        TacticId.ANTI_AIR_RESPONSE,
    ):
        assert (
            filter_strategy_action(
                get_tactic_spec(tactic_id),
                StrategyAction.EXPAND,
                _observation(
                    game_time=320.0,
                    minerals=500.0,
                    army_count=12.0,
                    base_under_threat=0.0,
                ),
            )
            is StrategyAction.EXPAND
        )


@pytest.mark.unit
def test_safe_macro_preserves_early_gateway_with_one_pending_gateway() -> None:
    safe_macro = get_tactic_spec(TacticId.SAFE_MACRO)

    assert (
        filter_strategy_action(
            safe_macro,
            StrategyAction.ADD_GATEWAYS,
            _observation(
                game_time=120.0,
                minerals=150.0,
                pending_gateways=1.0,
            ),
        )
        is StrategyAction.ADD_GATEWAYS
    )


@pytest.mark.unit
def test_safe_macro_caps_early_gateway_at_two_pending_gateways() -> None:
    safe_macro = get_tactic_spec(TacticId.SAFE_MACRO)

    assert (
        filter_strategy_action(
            safe_macro,
            StrategyAction.ADD_GATEWAYS,
            _observation(
                game_time=120.0,
                minerals=150.0,
                pending_gateways=2.0,
            ),
        )
        is StrategyAction.BOOST_WORKERS
    )


@pytest.mark.unit
def test_safe_macro_preserves_early_gateway_before_robo_gas_with_two_pending() -> None:
    safe_macro = get_tactic_spec(TacticId.SAFE_MACRO)

    assert (
        filter_strategy_action(
            safe_macro,
            StrategyAction.ADD_GATEWAYS,
            _observation(
                game_time=100.0,
                minerals=270.0,
                vespene=80.0,
                ready_gateways=0.0,
                pending_gateways=2.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.ADD_GATEWAYS
    )


@pytest.mark.unit
def test_safe_macro_keeps_gateway_cap_after_robo_gas_ready() -> None:
    safe_macro = get_tactic_spec(TacticId.SAFE_MACRO)

    assert (
        filter_strategy_action(
            safe_macro,
            StrategyAction.ADD_GATEWAYS,
            _observation(
                game_time=100.0,
                minerals=270.0,
                vespene=100.0,
                ready_gateways=0.0,
                pending_gateways=2.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.BOOST_WORKERS
    )


@pytest.mark.unit
def test_safe_macro_keeps_gateway_cap_under_threat() -> None:
    safe_macro = get_tactic_spec(TacticId.SAFE_MACRO)

    assert (
        filter_strategy_action(
            safe_macro,
            StrategyAction.ADD_GATEWAYS,
            _observation(
                game_time=100.0,
                minerals=270.0,
                vespene=80.0,
                ready_gateways=0.0,
                pending_gateways=2.0,
                base_under_threat=1.0,
            ),
        )
        is StrategyAction.BOOST_WORKERS
    )


@pytest.mark.unit
def test_filter_strategy_action_saves_minerals_for_initial_robo() -> None:
    robo = get_tactic_spec(TacticId.ROBO_TIMING)

    assert (
        filter_strategy_action(
            robo,
            StrategyAction.TECH_ROBO,
            _observation(
                game_time=420.0,
                minerals=80.0,
                vespene=500.0,
                ready_robo=0.0,
                pending_robo=0.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.STAY_COURSE
    )


@pytest.mark.unit
def test_filter_strategy_action_saves_for_initial_tech_power_robo() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.TECH_ROBO,
            _observation(
                game_time=420.0,
                minerals=80.0,
                vespene=500.0,
                ready_robo=0.0,
                pending_robo=0.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.STAY_COURSE
    )


@pytest.mark.unit
def test_tech_power_starts_initial_robo_from_affordable_stay_course() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.STAY_COURSE,
            _observation(
                game_time=260.0,
                minerals=220.0,
                vespene=150.0,
                ready_robo=0.0,
                pending_robo=0.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.TECH_ROBO
    )


@pytest.mark.unit
def test_tech_power_redirects_affordable_forge_to_initial_robo() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.FORGE_UPGRADES,
            _observation(
                game_time=390.0,
                minerals=155.0,
                vespene=1100.0,
                ready_robo=0.0,
                pending_robo=0.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.TECH_ROBO
    )


@pytest.mark.unit
def test_tech_power_banks_army_for_initial_robo_when_minerals_short() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.PRODUCE_ARMY,
            _observation(
                game_time=390.0,
                minerals=120.0,
                vespene=1100.0,
                ready_robo=0.0,
                pending_robo=0.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.STAY_COURSE
    )


@pytest.mark.unit
def test_tech_power_does_not_force_initial_robo_under_threat() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.STAY_COURSE,
            _observation(
                game_time=260.0,
                minerals=220.0,
                vespene=150.0,
                ready_robo=0.0,
                pending_robo=0.0,
                base_under_threat=1.0,
            ),
        )
        is StrategyAction.STAY_COURSE
    )


@pytest.mark.unit
def test_tech_power_uses_capped_gateway_to_start_first_robo() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.ADD_GATEWAYS,
            _observation(
                game_time=260.0,
                minerals=300.0,
                vespene=150.0,
                pending_gateways=1.0,
                ready_robo=0.0,
                pending_robo=0.0,
            ),
        )
        is StrategyAction.TECH_ROBO
    )


@pytest.mark.unit
def test_tech_power_capped_gateway_falls_to_army_after_robo_started() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.ADD_GATEWAYS,
            _observation(
                game_time=420.0,
                minerals=500.0,
                vespene=250.0,
                pending_gateways=1.0,
                ready_robo=1.0,
                pending_robo=0.0,
                observers=0.0,
                immortals=0.0,
            ),
        )
        is StrategyAction.PRODUCE_ARMY
    )


@pytest.mark.unit
def test_tech_power_waits_for_pending_static_under_threat_without_ready_static() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.BUILD_STATIC_DEFENSE,
            _observation(
                game_time=420.0,
                minerals=500.0,
                vespene=250.0,
                base_under_threat=1.0,
                pending_static_defense=1.0,
                ready_robo=0.0,
                pending_robo=0.0,
            ),
        )
        is StrategyAction.STAY_COURSE
    )


@pytest.mark.unit
def test_tech_power_static_defense_cap_falls_to_army_when_static_exists() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.BUILD_STATIC_DEFENSE,
            _observation(
                game_time=420.0,
                minerals=500.0,
                vespene=250.0,
                base_under_threat=1.0,
                pending_static_defense=1.0,
                ready_static_defense=1.0,
                ready_robo=0.0,
                pending_robo=0.0,
            ),
        )
        is StrategyAction.PRODUCE_ARMY
    )


@pytest.mark.unit
def test_tech_power_preserves_static_defense_under_active_threat() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.BUILD_STATIC_DEFENSE,
            _observation(
                game_time=420.0,
                minerals=120.0,
                vespene=250.0,
                base_under_threat=1.0,
                pending_static_defense=0.0,
                ready_robo=1.0,
                pending_robo=0.0,
                observers=1.0,
                immortals=0.0,
            ),
        )
        is StrategyAction.BUILD_STATIC_DEFENSE
    )


@pytest.mark.unit
def test_tech_power_banks_for_static_defense_under_threat_when_minerals_short() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.BUILD_STATIC_DEFENSE,
            _observation(
                game_time=420.0,
                minerals=80.0,
                vespene=250.0,
                base_under_threat=1.0,
                pending_static_defense=0.0,
                ready_robo=1.0,
                pending_robo=0.0,
                observers=1.0,
                immortals=0.0,
            ),
        )
        is StrategyAction.STAY_COURSE
    )


@pytest.mark.unit
def test_tech_power_does_not_repeat_robo_when_observer_immortal_missing() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.TECH_ROBO,
            _observation(
                game_time=420.0,
                minerals=500.0,
                vespene=250.0,
                ready_robo=1.0,
                pending_robo=0.0,
                observers=0.0,
                immortals=0.0,
            ),
        )
        is StrategyAction.PRODUCE_ARMY
    )


@pytest.mark.unit
def test_tech_power_biases_affordable_ready_robo_to_first_immortal() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.ADD_GATEWAYS,
            _observation(
                game_time=520.0,
                minerals=300.0,
                vespene=150.0,
                supply_left=4.0,
                ready_robo=1.0,
                pending_robo=0.0,
                observers=1.0,
                immortals=0.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.PRODUCE_ARMY
    )


@pytest.mark.unit
def test_tech_power_preserves_underbuilt_gateway_before_first_immortal() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.ADD_GATEWAYS,
            _observation(
                game_time=520.0,
                minerals=300.0,
                vespene=150.0,
                supply_left=4.0,
                own_bases=2.0,
                ready_gateways=4.0,
                pending_gateways=0.0,
                ready_robo=1.0,
                pending_robo=0.0,
                observers=1.0,
                immortals=0.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.ADD_GATEWAYS
    )


@pytest.mark.unit
def test_tech_power_preserves_underbuilt_gateway_with_one_pending_after_robo() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.ADD_GATEWAYS,
            _observation(
                game_time=520.0,
                minerals=300.0,
                vespene=150.0,
                supply_left=4.0,
                own_bases=2.0,
                ready_gateways=4.0,
                pending_gateways=1.0,
                ready_robo=1.0,
                pending_robo=0.0,
                observers=1.0,
                immortals=1.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.ADD_GATEWAYS
    )


@pytest.mark.unit
def test_tech_power_banks_for_first_immortal_when_minerals_short() -> None:
    power = get_tactic_spec(TacticId.TECH_POWER)

    assert (
        filter_strategy_action(
            power,
            StrategyAction.PRODUCE_ARMY,
            _observation(
                game_time=520.0,
                minerals=175.0,
                vespene=150.0,
                supply_left=8.0,
                ready_robo=1.0,
                pending_robo=0.0,
                observers=1.0,
                immortals=0.0,
                base_under_threat=0.0,
            ),
        )
        is StrategyAction.STAY_COURSE
    )


@pytest.mark.unit
def test_first_immortal_bias_does_not_override_static_defense_under_threat() -> None:
    anti_air = get_tactic_spec(TacticId.ANTI_AIR_RESPONSE)

    assert (
        filter_strategy_action(
            anti_air,
            StrategyAction.BUILD_STATIC_DEFENSE,
            _observation(
                game_time=520.0,
                minerals=175.0,
                vespene=150.0,
                supply_left=8.0,
                ready_robo=1.0,
                pending_robo=0.0,
                observers=1.0,
                immortals=0.0,
                base_under_threat=1.0,
                pending_static_defense=0.0,
            ),
        )
        is StrategyAction.BUILD_STATIC_DEFENSE
    )


@pytest.mark.unit
def test_anti_air_banks_for_static_defense_under_threat_without_static() -> None:
    anti_air = get_tactic_spec(TacticId.ANTI_AIR_RESPONSE)

    assert (
        filter_strategy_action(
            anti_air,
            StrategyAction.BUILD_STATIC_DEFENSE,
            _observation(
                game_time=620.0,
                minerals=65.0,
                vespene=250.0,
                base_under_threat=1.0,
                pending_static_defense=0.0,
                ready_static_defense=0.0,
                ready_robo=1.0,
                observers=1.0,
                immortals=0.0,
            ),
        )
        is StrategyAction.STAY_COURSE
    )


@pytest.mark.unit
def test_anti_air_waits_for_pending_static_under_threat_without_ready_static() -> None:
    anti_air = get_tactic_spec(TacticId.ANTI_AIR_RESPONSE)

    assert (
        filter_strategy_action(
            anti_air,
            StrategyAction.BUILD_STATIC_DEFENSE,
            _observation(
                game_time=620.0,
                minerals=220.0,
                vespene=250.0,
                base_under_threat=1.0,
                pending_static_defense=1.0,
                ready_static_defense=0.0,
                ready_robo=1.0,
                observers=1.0,
                immortals=0.0,
            ),
        )
        is StrategyAction.STAY_COURSE
    )


@pytest.mark.unit
def test_anti_air_banks_when_air_threat_static_exists_and_minerals_short() -> None:
    anti_air = get_tactic_spec(TacticId.ANTI_AIR_RESPONSE)

    assert (
        filter_strategy_action(
            anti_air,
            StrategyAction.BUILD_STATIC_DEFENSE,
            _observation(
                game_time=620.0,
                minerals=65.0,
                vespene=250.0,
                base_under_threat=1.0,
                base_under_air_threat=1.0,
                base_under_ground_threat=0.0,
                pending_static_defense=0.0,
                ready_static_defense=1.0,
                ready_robo=1.0,
                observers=1.0,
                immortals=0.0,
            ),
        )
        is StrategyAction.STAY_COURSE
    )


@pytest.mark.unit
def test_anti_air_uses_army_for_ground_threat_when_static_exists_and_minerals_short() -> None:
    anti_air = get_tactic_spec(TacticId.ANTI_AIR_RESPONSE)

    assert (
        filter_strategy_action(
            anti_air,
            StrategyAction.BUILD_STATIC_DEFENSE,
            _observation(
                game_time=620.0,
                minerals=65.0,
                vespene=250.0,
                base_under_threat=1.0,
                base_under_air_threat=0.0,
                base_under_ground_threat=1.0,
                pending_static_defense=0.0,
                ready_static_defense=1.0,
                ready_robo=1.0,
                observers=1.0,
                immortals=0.0,
            ),
        )
        is StrategyAction.PRODUCE_ARMY
    )


@pytest.mark.unit
def test_recovery_preserves_affordable_static_defense_under_threat() -> None:
    recovery = get_tactic_spec(TacticId.RECOVERY)

    assert (
        filter_strategy_action(
            recovery,
            StrategyAction.BUILD_STATIC_DEFENSE,
            _observation(
                game_time=620.0,
                minerals=120.0,
                vespene=250.0,
                base_under_threat=1.0,
                pending_static_defense=0.0,
                ready_static_defense=0.0,
                ready_robo=1.0,
                observers=1.0,
                immortals=0.0,
                army_count=1.0,
            ),
        )
        is StrategyAction.BUILD_STATIC_DEFENSE
    )


@pytest.mark.unit
def test_recovery_filter_prioritizes_army_before_workers() -> None:
    recovery = get_tactic_spec(TacticId.RECOVERY)

    assert (
        filter_strategy_action(
            recovery,
            StrategyAction.EXPAND,
            _observation(game_time=420.0, army_count=1.0),
        )
        is StrategyAction.PRODUCE_ARMY
    )


@pytest.mark.unit
def test_rule_tactic_selector_uses_aibuild_and_threat_context() -> None:
    selector = RuleTacticSelector()

    assert (
        selector.select(_observation(game_time=80.0), opponent_ai_build="Rush")
        .current_tactic
        is TacticId.ANTI_RUSH_DEFENSE
    )
    assert (
        selector.select(_observation(enemy_air_units_known=2.0), opponent_ai_build="Macro")
        .current_tactic
        is TacticId.ANTI_AIR_RESPONSE
    )
    assert (
        selector.select(
            _observation(game_time=420.0, army_count=14.0),
            opponent_ai_build="Macro",
        )
        .current_tactic
        is TacticId.GATEWAY_PRESSURE
    )
    assert (
        selector.select(
            _observation(game_time=420.0, enemy_armored_units_known=1.0),
            opponent_ai_build="Power",
        )
        .current_tactic
        is TacticId.ROBO_TIMING
    )
    assert (
        selector.select(_observation(game_time=420.0), opponent_ai_build="Power")
        .current_tactic
        is TacticId.TECH_POWER
    )


@pytest.mark.unit
def test_rule_tactic_selector_applies_cooldown_but_allows_emergencies() -> None:
    selector = RuleTacticSelector(min_tactic_duration=90.0)
    previous = TacticState(
        current_tactic=TacticId.SAFE_MACRO,
        phase=TacticPhase.OPENING,
        started_game_time=100.0,
        last_switch_game_time=100.0,
        last_switch_reason="safe_macro",
    )

    held = selector.select(
        _observation(game_time=130.0, army_count=14.0),
        opponent_ai_build="Macro",
        previous_state=previous,
    )
    emergency = selector.select(
        _observation(game_time=130.0, workers=3.0, own_bases=0.0, army_count=1.0),
        opponent_ai_build="Macro",
        previous_state=previous,
    )

    assert held.current_tactic is TacticId.SAFE_MACRO
    assert held.last_switch_reason == "cooldown"
    assert emergency.current_tactic is TacticId.RECOVERY
    assert emergency.previous_tactic is TacticId.SAFE_MACRO
