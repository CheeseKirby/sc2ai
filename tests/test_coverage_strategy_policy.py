from __future__ import annotations

from types import SimpleNamespace

import pytest

import bot.managers.coverage_strategy_policy as coverage_policy_module
from bot.managers.coverage_strategy_policy import CoverageStrategyPolicy
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS


class FakeObservation:
    def __init__(self, data: dict[str, float]) -> None:
        self.data = data

    def to_dict(self) -> dict[str, float]:
        return self.data


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "game_time": 240.0,
            "minerals": 250.0,
            "vespene": 100.0,
            "supply_left": 8.0,
            "workers": 22.0,
            "own_bases": 1.0,
            "ready_gateways": 4.0,
            "has_cybernetics_core": 1.0,
            "army_count": 8.0,
            "worker_saturation_ratio": 1.0,
        }
    )
    observation.update(overrides)
    return observation


@pytest.mark.unit
def test_coverage_strategy_builds_static_defense_for_base_threat() -> None:
    policy = CoverageStrategyPolicy()

    action = policy.decide_from_observation(
        _observation(
            own_bases=2.0,
            base_under_threat=1.0,
            ready_static_defense=1.0,
        )
    )

    assert action is StrategyAction.BUILD_STATIC_DEFENSE
    assert policy.last_decision_source == "coverage-teacher"
    assert policy.last_decision_reason == "base_threat_static_defense_gap"


@pytest.mark.unit
def test_coverage_strategy_decide_strategy_writes_bot_reason(monkeypatch) -> None:
    observation = _observation(
        own_bases=2.0,
        base_under_threat=1.0,
        ready_static_defense=1.0,
    )
    bot = SimpleNamespace()
    monkeypatch.setattr(
        coverage_policy_module,
        "build_strategy_observation",
        lambda _bot: FakeObservation(observation),
    )
    policy = CoverageStrategyPolicy()

    action = policy.decide_strategy(bot)

    assert action is StrategyAction.BUILD_STATIC_DEFENSE
    assert bot.last_strategy_decision_source == "coverage-teacher"
    assert bot.last_strategy_decision_reason == "base_threat_static_defense_gap"


@pytest.mark.unit
def test_coverage_strategy_boosts_workers_when_under_saturated() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            own_bases=2.0,
            workers=20.0,
            worker_saturation_ratio=0.45,
            ready_gateways=8.0,
        )
    )

    assert action is StrategyAction.BOOST_WORKERS


@pytest.mark.unit
def test_coverage_strategy_expands_when_saturated_and_safe() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            own_bases=1.0,
            minerals=550.0,
            worker_saturation_ratio=1.0,
        )
    )

    assert action is StrategyAction.EXPAND


@pytest.mark.unit
def test_coverage_strategy_adds_gateways_when_count_is_low() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            own_bases=2.0,
            minerals=450.0,
            ready_gateways=3.0,
            worker_saturation_ratio=1.0,
        )
    )

    assert action is StrategyAction.ADD_GATEWAYS


@pytest.mark.unit
def test_coverage_strategy_prioritizes_midgame_forge_before_extra_gateways() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            game_time=420.0,
            own_bases=2.0,
            minerals=450.0,
            vespene=150.0,
            ready_gateways=4.0,
            ready_robo=0.0,
            ready_forge=0.0,
            army_count=14.0,
        )
    )

    assert action is StrategyAction.FORGE_UPGRADES


@pytest.mark.unit
def test_coverage_strategy_techs_robo_for_armored_or_cloaked_signals() -> None:
    policy = CoverageStrategyPolicy()

    assert policy.decide_from_observation(
        _observation(
            own_bases=2.0,
            ready_gateways=8.0,
            ready_robo=0.0,
            enemy_armored_units_known=1.0,
            vespene=150.0,
        )
    ) is StrategyAction.TECH_ROBO
    assert policy.decide_from_observation(
        _observation(
            own_bases=2.0,
            ready_gateways=8.0,
            ready_robo=0.0,
            enemy_cloaked_units_seen=1.0,
            vespene=150.0,
        )
    ) is StrategyAction.TECH_ROBO


@pytest.mark.unit
def test_coverage_strategy_techs_robo_for_cloaked_signal_before_forge() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            game_time=420.0,
            own_bases=2.0,
            minerals=450.0,
            vespene=150.0,
            ready_gateways=4.0,
            ready_robo=0.0,
            ready_forge=0.0,
            enemy_cloaked_units_seen=1.0,
        )
    )

    assert action is StrategyAction.TECH_ROBO


@pytest.mark.unit
def test_coverage_strategy_builds_forge_for_midgame_upgrades() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            game_time=420.0,
            own_bases=2.0,
            minerals=450.0,
            ready_gateways=8.0,
            ready_robo=1.0,
            ready_forge=0.0,
        )
    )

    assert action is StrategyAction.FORGE_UPGRADES


@pytest.mark.unit
def test_coverage_strategy_continues_forge_upgrades_when_forge_exists() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            game_time=420.0,
            own_bases=2.0,
            minerals=450.0,
            ready_gateways=8.0,
            ready_robo=1.0,
            ready_forge=1.0,
            ground_weapon_level=0.0,
            ground_armor_level=0.0,
        )
    )

    assert action is StrategyAction.FORGE_UPGRADES


@pytest.mark.unit
def test_coverage_strategy_does_not_duplicate_pending_forge() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            game_time=420.0,
            own_bases=2.0,
            minerals=450.0,
            ready_gateways=8.0,
            ready_robo=1.0,
            ready_forge=0.0,
            pending_forge=1.0,
            gateway_idle_count=2.0,
            army_count=12.0,
        )
    )

    assert action is StrategyAction.PRODUCE_ARMY


@pytest.mark.unit
def test_coverage_strategy_does_not_duplicate_pending_forge_upgrade() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            game_time=420.0,
            own_bases=2.0,
            minerals=450.0,
            ready_gateways=8.0,
            ready_robo=1.0,
            ready_forge=1.0,
            ground_weapon_level=0.0,
            ground_weapon_upgrade_pending=1.0,
            ground_armor_level=0.0,
            gateway_idle_count=2.0,
            army_count=12.0,
        )
    )

    assert action is StrategyAction.PRODUCE_ARMY


@pytest.mark.unit
def test_coverage_strategy_does_not_duplicate_pending_robo() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            own_bases=2.0,
            minerals=250.0,
            vespene=150.0,
            ready_gateways=8.0,
            ready_robo=0.0,
            pending_robo=1.0,
            enemy_armored_units_known=1.0,
            gateway_idle_count=2.0,
            army_count=12.0,
        )
    )

    assert action is StrategyAction.PRODUCE_ARMY


@pytest.mark.unit
def test_coverage_strategy_produces_army_when_production_is_idle() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            own_bases=2.0,
            minerals=200.0,
            ready_gateways=8.0,
            ready_robo=1.0,
            ready_forge=1.0,
            gateway_idle_count=2.0,
            army_count=7.0,
        )
    )

    assert action is StrategyAction.PRODUCE_ARMY


@pytest.mark.unit
def test_coverage_strategy_stays_course_when_no_rule_fires() -> None:
    action = CoverageStrategyPolicy().decide_from_observation(
        _observation(
            own_bases=2.0,
            ready_gateways=8.0,
            ready_robo=1.0,
            ready_forge=1.0,
            ready_static_defense=4.0,
            army_count=16.0,
            gateway_idle_count=0.0,
            robo_idle_count=0.0,
        )
    )

    assert action is StrategyAction.STAY_COURSE
