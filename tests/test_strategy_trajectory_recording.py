from __future__ import annotations

from types import SimpleNamespace

import pytest

import bot.protoss_rule_bot as protoss_module
from bot.managers.army_policy import ArmyAction
from bot.managers.strategy_executor import StrategyExecutionResult
from bot.protoss_rule_bot import ProtossRuleBot
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import STRATEGY_OBSERVATION_DETAIL_FIELDS
from rl.tactics import TacticId, TacticPhase, TacticState
from rl.trajectory_recorder import StrategyTrajectoryStep, TrajectoryStep


class FakeObservation:
    def __init__(self, data: dict[str, float]) -> None:
        self.data = data

    def to_dict(self) -> dict[str, float]:
        return self.data


class FakeRecorder:
    def __init__(self) -> None:
        self.records: list[object] = []
        self.closed = False

    def record(self, step: object) -> None:
        self.records.append(step)

    def close(self) -> None:
        self.closed = True


def _strategy_observation_details(
    **overrides: float,
) -> dict[str, float]:
    details = {
        "ready_photon_cannons": 0.0,
        "pending_photon_cannons": 0.0,
        "ready_shield_batteries": 0.0,
        "pending_shield_batteries": 0.0,
    }
    details.update(overrides)
    assert tuple(details) == STRATEGY_OBSERVATION_DETAIL_FIELDS
    return details


@pytest.mark.unit
def test_bot_records_strategy_trajectory_separately(monkeypatch) -> None:
    recorder = FakeRecorder()
    bot = SimpleNamespace(
        NAME="FakeBot",
        strategy_trajectory_recorder=recorder,
        episode_id="episode-1",
        episode_metadata={
            "map_name": "AcropolisLE",
            "difficulty": "Medium",
            "opponent_race": "Zerg",
            "opponent_ai_build": "Rush",
        },
        last_army_action=ArmyAction.DEFEND_BASE,
        last_tactic_state=None,
        last_tactic_source=None,
        last_strategy_action_before_tactic_filter=None,
        last_strategy_action_after_tactic_filter=None,
        last_strategy_decision_source="coverage-teacher",
        last_strategy_decision_reason="base_threat_static_defense_gap",
        last_strategy_execution_result=None,
    )
    monkeypatch.setattr(
        protoss_module,
        "build_strategy_observation",
        lambda _bot: FakeObservation({"own_bases": 2.0}),
    )
    monkeypatch.setattr(
        protoss_module,
        "build_strategy_observation_details",
        lambda _bot: FakeObservation(
            _strategy_observation_details(
                ready_photon_cannons=1.0,
                pending_photon_cannons=1.0,
                ready_shield_batteries=2.0,
                pending_shield_batteries=0.0,
            )
        ),
    )
    monkeypatch.setattr(
        protoss_module,
        "build_observation",
        lambda _bot: FakeObservation({"army_count": 12.0}),
    )

    ProtossRuleBot._record_strategy_trajectory_step(
        bot,
        64,
        StrategyAction.TECH_ROBO,
        reward=1.25,
    )

    details = _strategy_observation_details(
        ready_photon_cannons=1.0,
        pending_photon_cannons=1.0,
        ready_shield_batteries=2.0,
        pending_shield_batteries=0.0,
    )
    assert recorder.records == [
        StrategyTrajectoryStep(
            episode_id="episode-1",
            step=64,
            map_name="AcropolisLE",
            difficulty="Medium",
            opponent_race="Zerg",
            strategy_observation={"own_bases": 2.0},
            strategy_observation_details=details,
            strategy_action=3,
            strategy_action_name="TECH_ROBO",
            strategy_policy_source="coverage-teacher",
            strategy_policy_reason="base_threat_static_defense_gap",
            army_observation={"army_count": 12.0},
            army_action=3,
            army_action_name="DEFEND_BASE",
            reward=1.25,
            done=False,
            result=None,
            opponent_ai_build="Rush",
        )
    ]
    step = recorder.records[0]
    assert isinstance(step, StrategyTrajectoryStep)
    assert tuple(step.strategy_observation_details or {}) == (
        STRATEGY_OBSERVATION_DETAIL_FIELDS
    )
    assert step.strategy_policy_source == "coverage-teacher"
    assert step.strategy_policy_reason == "base_threat_static_defense_gap"


@pytest.mark.unit
def test_bot_records_strategy_tactic_metadata(monkeypatch) -> None:
    recorder = FakeRecorder()
    bot = SimpleNamespace(
        NAME="FakeBot",
        strategy_trajectory_recorder=recorder,
        episode_id="episode-3",
        episode_metadata={
            "map_name": "AcropolisLE",
            "difficulty": "Hard",
            "opponent_race": "Terran",
            "opponent_ai_build": "Power",
        },
        last_army_action=ArmyAction.HOLD,
        last_tactic_state=TacticState(
            current_tactic=TacticId.ROBO_TIMING,
            phase=TacticPhase.POWER_SPIKE,
            started_game_time=240.0,
            last_switch_game_time=240.0,
            last_switch_reason="robo_signal",
            previous_tactic=TacticId.TECH_POWER,
        ),
        last_tactic_source="rule",
        last_strategy_action_before_tactic_filter=StrategyAction.TECH_ROBO,
        last_strategy_action_after_tactic_filter=StrategyAction.PRODUCE_ARMY,
        last_strategy_decision_source="tactic-aware-rule",
        last_strategy_decision_reason=(
            "tactic_filter_ROBO_TIMING_TECH_ROBO_to_PRODUCE_ARMY"
        ),
        last_strategy_execution_result=None,
    )
    monkeypatch.setattr(
        protoss_module,
        "build_strategy_observation",
        lambda _bot: FakeObservation({"pending_robo": 1.0}),
    )
    monkeypatch.setattr(
        protoss_module,
        "build_strategy_observation_details",
        lambda _bot: FakeObservation(_strategy_observation_details()),
    )
    monkeypatch.setattr(
        protoss_module,
        "build_observation",
        lambda _bot: FakeObservation({"army_count": 10.0}),
    )

    ProtossRuleBot._record_strategy_trajectory_step(
        bot,
        128,
        StrategyAction.PRODUCE_ARMY,
    )

    step = recorder.records[0]
    assert isinstance(step, StrategyTrajectoryStep)
    assert step.tactic_id == "ROBO_TIMING"
    assert step.tactic_phase == "POWER_SPIKE"
    assert step.tactic_source == "rule"
    assert step.tactic_started_game_time == 240.0
    assert step.tactic_switch_reason == "robo_signal"
    assert step.tactic_previous_id == "TECH_POWER"
    assert step.strategy_action_before_tactic_filter == 3
    assert step.strategy_action_before_tactic_filter_name == "TECH_ROBO"
    assert step.strategy_action_after_tactic_filter == 6
    assert step.strategy_action_after_tactic_filter_name == "PRODUCE_ARMY"
    assert step.strategy_policy_source == "tactic-aware-rule"
    assert step.strategy_policy_reason == (
        "tactic_filter_ROBO_TIMING_TECH_ROBO_to_PRODUCE_ARMY"
    )


@pytest.mark.unit
def test_bot_records_strategy_execution_metadata(monkeypatch) -> None:
    recorder = FakeRecorder()
    bot = SimpleNamespace(
        NAME="FakeBot",
        strategy_trajectory_recorder=recorder,
        episode_id="episode-4",
        episode_metadata={
            "map_name": "AcropolisLE",
            "difficulty": "Hard",
            "opponent_race": "Terran",
            "opponent_ai_build": "Power",
        },
        last_army_action=ArmyAction.HOLD,
        last_tactic_state=None,
        last_tactic_source=None,
        last_strategy_action_before_tactic_filter=None,
        last_strategy_action_after_tactic_filter=None,
        last_strategy_execution_result=StrategyExecutionResult(
            action=StrategyAction.TECH_ROBO,
            attempted=True,
            effect="build_structure",
            unit_type="ROBOTICSFACILITY",
            target="power_field",
        ),
    )
    monkeypatch.setattr(
        protoss_module,
        "build_strategy_observation",
        lambda _bot: FakeObservation({"pending_robo": 0.0}),
    )
    monkeypatch.setattr(
        protoss_module,
        "build_strategy_observation_details",
        lambda _bot: FakeObservation(_strategy_observation_details()),
    )
    monkeypatch.setattr(
        protoss_module,
        "build_observation",
        lambda _bot: FakeObservation({"army_count": 10.0}),
    )

    ProtossRuleBot._record_strategy_trajectory_step(
        bot,
        192,
        StrategyAction.TECH_ROBO,
    )

    step = recorder.records[0]
    assert isinstance(step, StrategyTrajectoryStep)
    assert step.strategy_execution_attempted is True
    assert step.strategy_execution_effect == "build_structure"
    assert step.strategy_execution_blocker is None
    assert step.strategy_execution_unit_type == "ROBOTICSFACILITY"
    assert step.strategy_execution_target == "power_field"


@pytest.mark.unit
def test_bot_records_army_trajectory_ai_build_metadata(monkeypatch) -> None:
    recorder = FakeRecorder()
    bot = SimpleNamespace(
        NAME="FakeBot",
        trajectory_recorder=recorder,
        episode_id="episode-2",
        episode_metadata={
            "map_name": "AcropolisLE",
            "difficulty": "Hard",
            "opponent_race": "Terran",
            "opponent_ai_build": "Timing",
        },
        record_decision_interval=8,
    )
    monkeypatch.setattr(
        protoss_module,
        "build_observation",
        lambda _bot: FakeObservation({"army_count": 8.0}),
    )

    ProtossRuleBot._record_trajectory_step(
        bot,
        16,
        ArmyAction.ATTACK_MAIN,
        reward=0.5,
    )

    assert recorder.records == [
        TrajectoryStep(
            episode_id="episode-2",
            step=16,
            map_name="AcropolisLE",
            difficulty="Hard",
            opponent_race="Terran",
            observation={"army_count": 8.0},
            action=1,
            action_name="ATTACK_MAIN",
            reward=0.5,
            done=False,
            result=None,
            opponent_ai_build="Timing",
        )
    ]


@pytest.mark.unit
def test_bot_strategy_trajectory_recording_is_noop_when_disabled() -> None:
    bot = SimpleNamespace(strategy_trajectory_recorder=None)

    ProtossRuleBot._record_strategy_trajectory_step(
        bot,
        64,
        StrategyAction.STAY_COURSE,
    )
