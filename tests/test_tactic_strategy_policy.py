from __future__ import annotations

from types import SimpleNamespace

import pytest

import bot.managers.tactic_strategy_policy as tactic_policy_module
from bot.managers.tactic_strategy_policy import TacticAwareStrategyPolicy
from rl.strategy_actions import StrategyAction
from rl.tactics import TacticId


class FakeObservation:
    def __init__(self, data: dict[str, float]) -> None:
        self.data = data

    def to_dict(self) -> dict[str, float]:
        return self.data


class FakeBasePolicy:
    def __init__(self, action: StrategyAction) -> None:
        self.action = action
        self.observations: list[dict[str, float]] = []

    def decide_from_observation(
        self,
        observation: dict[str, float],
    ) -> StrategyAction:
        self.observations.append(observation)
        return self.action


def _observation(**overrides: float) -> dict[str, float]:
    observation = {
        "game_time": 420.0,
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
        "army_count": 12.0,
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
def test_tactic_aware_policy_filters_pending_robo_repeat() -> None:
    base = FakeBasePolicy(StrategyAction.TECH_ROBO)
    policy = TacticAwareStrategyPolicy(base)
    observation = _observation(pending_robo=1.0, enemy_armored_units_known=1.0)

    action = policy.decide_from_observation(
        observation,
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.PRODUCE_ARMY
    assert base.observations == [observation]
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.ROBO_TIMING
    assert policy.last_proposed_action is StrategyAction.TECH_ROBO
    assert policy.last_filtered_action is StrategyAction.PRODUCE_ARMY
    assert policy.last_decision_source == "tactic-aware-rule"
    assert policy.last_decision_reason == (
        "tactic_filter_ROBO_TIMING_TECH_ROBO_to_PRODUCE_ARMY"
    )


@pytest.mark.unit
def test_tactic_aware_power_policy_starts_robo_from_capped_gateway() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.ADD_GATEWAYS))

    action = policy.decide_from_observation(
        _observation(
            game_time=260.0,
            pending_gateways=1.0,
            ready_robo=0.0,
            pending_robo=0.0,
        ),
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.TECH_ROBO
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.TECH_POWER
    assert policy.last_proposed_action is StrategyAction.ADD_GATEWAYS
    assert policy.last_filtered_action is StrategyAction.TECH_ROBO


@pytest.mark.unit
def test_tactic_aware_power_policy_starts_initial_robo_from_stay_course() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.STAY_COURSE))

    action = policy.decide_from_observation(
        _observation(
            game_time=260.0,
            minerals=220.0,
            vespene=150.0,
            ready_robo=0.0,
            pending_robo=0.0,
            base_under_threat=0.0,
        ),
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.TECH_ROBO
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.TECH_POWER
    assert policy.last_proposed_action is StrategyAction.STAY_COURSE
    assert policy.last_filtered_action is StrategyAction.TECH_ROBO


@pytest.mark.unit
def test_tactic_aware_power_policy_redirects_forge_to_initial_robo() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.FORGE_UPGRADES))

    action = policy.decide_from_observation(
        _observation(
            game_time=390.0,
            minerals=155.0,
            vespene=1100.0,
            ready_robo=0.0,
            pending_robo=0.0,
            base_under_threat=0.0,
        ),
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.TECH_ROBO
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.TECH_POWER
    assert policy.last_proposed_action is StrategyAction.FORGE_UPGRADES
    assert policy.last_filtered_action is StrategyAction.TECH_ROBO


@pytest.mark.unit
def test_tactic_aware_power_policy_preserves_safe_macro_early_gateway() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.ADD_GATEWAYS))

    action = policy.decide_from_observation(
        _observation(
            game_time=120.0,
            minerals=150.0,
            pending_gateways=1.0,
            ready_robo=0.0,
            pending_robo=0.0,
        ),
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.ADD_GATEWAYS
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.SAFE_MACRO
    assert policy.last_proposed_action is StrategyAction.ADD_GATEWAYS
    assert policy.last_filtered_action is StrategyAction.ADD_GATEWAYS


@pytest.mark.unit
def test_tactic_aware_power_policy_preserves_pre_robo_gas_early_gateway() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.ADD_GATEWAYS))

    action = policy.decide_from_observation(
        _observation(
            game_time=100.0,
            minerals=270.0,
            vespene=80.0,
            ready_gateways=0.0,
            pending_gateways=2.0,
            ready_robo=0.0,
            pending_robo=0.0,
            base_under_threat=0.0,
        ),
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.ADD_GATEWAYS
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.SAFE_MACRO
    assert policy.last_proposed_action is StrategyAction.ADD_GATEWAYS
    assert policy.last_filtered_action is StrategyAction.ADD_GATEWAYS


@pytest.mark.unit
def test_tactic_aware_power_policy_stops_extra_gateway_after_robo_gas_ready() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.ADD_GATEWAYS))

    action = policy.decide_from_observation(
        _observation(
            game_time=100.0,
            minerals=270.0,
            vespene=100.0,
            ready_gateways=0.0,
            pending_gateways=2.0,
            ready_robo=0.0,
            pending_robo=0.0,
            base_under_threat=0.0,
        ),
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.BOOST_WORKERS
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.SAFE_MACRO
    assert policy.last_proposed_action is StrategyAction.ADD_GATEWAYS
    assert policy.last_filtered_action is StrategyAction.BOOST_WORKERS


@pytest.mark.unit
def test_tactic_aware_power_policy_biases_ready_robo_to_first_immortal() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.ADD_GATEWAYS))

    action = policy.decide_from_observation(
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
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.PRODUCE_ARMY
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.TECH_POWER
    assert policy.last_proposed_action is StrategyAction.ADD_GATEWAYS
    assert policy.last_filtered_action is StrategyAction.PRODUCE_ARMY


@pytest.mark.unit
def test_tactic_aware_power_policy_preserves_underbuilt_gateway() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.ADD_GATEWAYS))

    action = policy.decide_from_observation(
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
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.ADD_GATEWAYS
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.TECH_POWER
    assert policy.last_proposed_action is StrategyAction.ADD_GATEWAYS
    assert policy.last_filtered_action is StrategyAction.ADD_GATEWAYS


@pytest.mark.unit
def test_tactic_aware_power_policy_preserves_pending_underbuilt_gateway() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.ADD_GATEWAYS))

    action = policy.decide_from_observation(
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
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.ADD_GATEWAYS
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.TECH_POWER
    assert policy.last_proposed_action is StrategyAction.ADD_GATEWAYS
    assert policy.last_filtered_action is StrategyAction.ADD_GATEWAYS


@pytest.mark.unit
def test_tactic_aware_power_policy_preserves_static_defense_under_threat() -> None:
    policy = TacticAwareStrategyPolicy(
        FakeBasePolicy(StrategyAction.BUILD_STATIC_DEFENSE)
    )

    action = policy.decide_from_observation(
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
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.BUILD_STATIC_DEFENSE
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.TECH_POWER
    assert policy.last_proposed_action is StrategyAction.BUILD_STATIC_DEFENSE
    assert policy.last_filtered_action is StrategyAction.BUILD_STATIC_DEFENSE


@pytest.mark.unit
def test_tactic_aware_power_policy_waits_for_pending_static_under_threat() -> None:
    policy = TacticAwareStrategyPolicy(
        FakeBasePolicy(StrategyAction.BUILD_STATIC_DEFENSE)
    )

    action = policy.decide_from_observation(
        _observation(
            game_time=420.0,
            base_under_threat=1.0,
            pending_static_defense=1.0,
            ready_robo=0.0,
            pending_robo=0.0,
        ),
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.STAY_COURSE
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.TECH_POWER
    assert policy.last_proposed_action is StrategyAction.BUILD_STATIC_DEFENSE
    assert policy.last_filtered_action is StrategyAction.STAY_COURSE


@pytest.mark.unit
def test_tactic_aware_power_policy_uses_army_when_ready_static_exists() -> None:
    policy = TacticAwareStrategyPolicy(
        FakeBasePolicy(StrategyAction.BUILD_STATIC_DEFENSE)
    )

    action = policy.decide_from_observation(
        _observation(
            game_time=420.0,
            base_under_threat=1.0,
            pending_static_defense=1.0,
            ready_static_defense=1.0,
            ready_robo=0.0,
            pending_robo=0.0,
        ),
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.PRODUCE_ARMY
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.TECH_POWER
    assert policy.last_proposed_action is StrategyAction.BUILD_STATIC_DEFENSE
    assert policy.last_filtered_action is StrategyAction.PRODUCE_ARMY


@pytest.mark.unit
def test_tactic_aware_anti_air_banks_static_defense_under_threat() -> None:
    policy = TacticAwareStrategyPolicy(
        FakeBasePolicy(StrategyAction.BUILD_STATIC_DEFENSE)
    )

    action = policy.decide_from_observation(
        _observation(
            game_time=620.0,
            minerals=65.0,
            vespene=250.0,
            base_under_threat=1.0,
            enemy_air_units_known=2.0,
            pending_static_defense=0.0,
            ready_static_defense=0.0,
            ready_robo=1.0,
            pending_robo=0.0,
            observers=1.0,
            immortals=0.0,
        ),
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.STAY_COURSE
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.ANTI_AIR_RESPONSE
    assert policy.last_proposed_action is StrategyAction.BUILD_STATIC_DEFENSE
    assert policy.last_filtered_action is StrategyAction.STAY_COURSE


@pytest.mark.unit
def test_tactic_aware_anti_air_banks_when_ready_static_low_minerals() -> None:
    policy = TacticAwareStrategyPolicy(
        FakeBasePolicy(StrategyAction.BUILD_STATIC_DEFENSE)
    )

    action = policy.decide_from_observation(
        _observation(
            game_time=620.0,
            minerals=65.0,
            vespene=1800.0,
            base_under_threat=1.0,
            base_under_air_threat=1.0,
            base_under_ground_threat=0.0,
            enemy_air_units_known=2.0,
            pending_static_defense=0.0,
            ready_static_defense=1.0,
            ready_robo=1.0,
            pending_robo=0.0,
            observers=1.0,
            immortals=0.0,
        ),
        opponent_ai_build="Power",
    )

    assert action is StrategyAction.STAY_COURSE
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.ANTI_AIR_RESPONSE
    assert policy.last_proposed_action is StrategyAction.BUILD_STATIC_DEFENSE
    assert policy.last_filtered_action is StrategyAction.STAY_COURSE


@pytest.mark.unit
def test_tactic_aware_policy_keeps_allowed_macro_pressure_action() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.ADD_GATEWAYS))

    action = policy.decide_from_observation(
        _observation(army_count=14.0),
        opponent_ai_build="Macro",
    )

    assert action is StrategyAction.ADD_GATEWAYS
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.GATEWAY_PRESSURE


@pytest.mark.unit
def test_tactic_aware_policy_records_non_power_tactic_without_filtering() -> None:
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.FORGE_UPGRADES))

    action = policy.decide_from_observation(
        _observation(game_time=420.0, army_count=1.0),
        opponent_ai_build="Rush",
    )

    assert action is StrategyAction.FORGE_UPGRADES
    assert policy.last_tactic_state is not None
    assert policy.last_tactic_state.current_tactic is TacticId.RECOVERY
    assert policy.last_proposed_action is StrategyAction.FORGE_UPGRADES
    assert policy.last_filtered_action is StrategyAction.FORGE_UPGRADES


@pytest.mark.unit
def test_tactic_aware_policy_decide_strategy_writes_bot_metadata(monkeypatch) -> None:
    observation = _observation(pending_robo=1.0, enemy_armored_units_known=1.0)
    bot = SimpleNamespace(
        episode_metadata={"opponent_ai_build": "Power"},
    )
    monkeypatch.setattr(
        tactic_policy_module,
        "build_strategy_observation",
        lambda _bot: FakeObservation(observation),
    )
    policy = TacticAwareStrategyPolicy(FakeBasePolicy(StrategyAction.TECH_ROBO))

    action = policy.decide_strategy(bot)

    assert action is StrategyAction.PRODUCE_ARMY
    assert bot.last_tactic_state.current_tactic is TacticId.ROBO_TIMING
    assert bot.last_tactic_source == "rule"
    assert bot.last_strategy_action_before_tactic_filter is StrategyAction.TECH_ROBO
    assert bot.last_strategy_action_after_tactic_filter is StrategyAction.PRODUCE_ARMY
    assert bot.last_strategy_decision_source == "tactic-aware-rule"
    assert bot.last_strategy_decision_reason == (
        "tactic_filter_ROBO_TIMING_TECH_ROBO_to_PRODUCE_ARMY"
    )
