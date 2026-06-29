from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
from sc2.ids.unit_typeid import UnitTypeId

from bot.managers.army_policy import ArmyAction
from bot.managers.army_memory import ArmyMemory
from rl.actions import ACTION_NAMES, action_from_int, action_name, action_to_int
from rl.observations import (
    OBSERVATION_FIELDS,
    OBSERVATION_FIELDS_V1,
    OBSERVATION_FIELDS_V2,
    ArmyObservation,
    build_observation,
    infer_observation_schema_version,
    normalize_observation_dict,
    observation_dict_to_vector,
    validate_observation_dict,
)
from rl.strategy_observations import STRATEGY_OBSERVATION_DETAIL_FIELDS
from rl.trajectory_recorder import JsonlTrajectoryRecorder, TrajectoryStep
from rl.trajectory_recorder import StrategyTrajectoryStep


@dataclass(frozen=True)
class FakePosition:
    x: float
    y: float

    def distance_to(self, other: FakePosition) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class FakeUnits:
    def __init__(self, amount: int, center: FakePosition | None = None) -> None:
        self.amount = amount
        self.center = center or FakePosition(0, 0)

    @property
    def exists(self) -> bool:
        return self.amount > 0


class FakeUnitCollection:
    def __init__(self) -> None:
        self.counts = {
            UnitTypeId.ZEALOT: 3,
            UnitTypeId.STALKER: 2,
        }

    def of_type(self, unit_types: set[UnitTypeId] | frozenset[UnitTypeId]) -> FakeUnits:
        amount = sum(self.counts.get(unit_type, 0) for unit_type in unit_types)
        return FakeUnits(amount, FakePosition(6, 8))


class FakeStructures:
    def __init__(self, counts: dict[UnitTypeId, int]) -> None:
        self.counts = counts

    def __call__(self, unit_type: UnitTypeId) -> FakeUnits:
        return FakeUnits(self.counts.get(unit_type, 0))


@dataclass
class FakeTownhall:
    position: FakePosition


class FakeTownhalls:
    amount = 1

    @property
    def exists(self) -> bool:
        return True

    @property
    def first(self) -> FakeTownhall:
        return FakeTownhall(FakePosition(0, 0))


class FakeBot:
    time = 123.5
    minerals = 400
    vespene = 125
    supply_used = 35
    supply_cap = 46
    supply_left = 11
    is_attacking = True
    enemy_start_locations = [FakePosition(9, 12)]

    def __init__(self) -> None:
        self.units = FakeUnitCollection()
        self.workers = FakeUnits(22)
        self.townhalls = FakeTownhalls()
        self.structures = FakeStructures(
            {
                UnitTypeId.GATEWAY: 4,
                UnitTypeId.CYBERNETICSCORE: 1,
            }
        )
        self.enemy_units = FakeUnits(7)
        self.enemy_structures = FakeUnits(3)
        self.army_memory = ArmyMemory()
        self.army_memory.update(7, is_attacking=True)
        self.army_memory.update(5, is_attacking=True)


@pytest.mark.unit
def test_army_action_helpers_are_stable() -> None:
    assert action_to_int(ArmyAction.ATTACK_MAIN) == 1
    assert action_from_int(2) is ArmyAction.RETREAT_HOME
    assert action_name(ArmyAction.HOLD) == "HOLD"
    assert ACTION_NAMES[0] == "RALLY"


@pytest.mark.unit
def test_army_observation_serializes_to_dict_and_vector() -> None:
    observation = build_observation(FakeBot())

    data = observation.to_dict()
    vector = observation.to_vector()

    assert data["workers"] == 22.0
    assert data["gateways"] == 4.0
    assert data["has_cybernetics_core"] == 1.0
    assert data["zealots"] == 3.0
    assert data["stalkers"] == 2.0
    assert data["army_count"] == 5.0
    assert data["army_to_home_distance"] == 10.0
    assert data["army_to_enemy_start_distance"] == 5.0
    assert data["base_under_threat"] == 0.0
    assert data["enemy_to_home_distance"] == 0.0
    assert data["army_idle_count"] == 0.0
    assert data["army_busy_count"] == 5.0
    assert data["attack_army_peak"] == 7.0
    assert data["army_lost_from_peak"] == 2.0
    assert data["army_lost_from_peak_ratio"] == pytest.approx(2 / 7)
    assert data["army_count_delta"] == -2.0
    assert vector.dtype == np.float32
    assert vector.shape == (len(OBSERVATION_FIELDS),)
    assert tuple(ArmyObservation.__dataclass_fields__) == OBSERVATION_FIELDS


@pytest.mark.unit
def test_observation_dict_vectorization_uses_stable_field_order() -> None:
    observation = {field: float(index) for index, field in enumerate(OBSERVATION_FIELDS)}

    vector = observation_dict_to_vector(observation)

    assert vector.tolist() == [float(index) for index in range(len(OBSERVATION_FIELDS))]


@pytest.mark.unit
def test_observation_validation_rejects_missing_fields() -> None:
    observation = {field: 1.0 for field in OBSERVATION_FIELDS}
    del observation["workers"]

    with pytest.raises(ValueError, match="workers"):
        validate_observation_dict(observation)


@pytest.mark.unit
def test_observation_v1_rows_can_be_defaulted_to_current_schema() -> None:
    observation = {
        field: float(index) for index, field in enumerate(OBSERVATION_FIELDS_V1)
    }

    normalized = normalize_observation_dict(
        observation,
        allow_missing_defaults=True,
    )
    vector = observation_dict_to_vector(
        observation,
        allow_missing_defaults=True,
    )

    assert infer_observation_schema_version(observation) == 1
    assert normalized["base_under_threat"] == 0.0
    assert normalized["enemy_to_home_distance"] == 0.0
    assert normalized["attack_army_peak"] == 0.0
    assert normalized["army_lost_from_peak_ratio"] == 0.0
    assert vector.shape == (len(OBSERVATION_FIELDS),)


@pytest.mark.unit
def test_observation_v2_rows_can_be_defaulted_to_current_schema() -> None:
    observation = {
        field: float(index) for index, field in enumerate(OBSERVATION_FIELDS_V2)
    }

    normalized = normalize_observation_dict(
        observation,
        allow_missing_defaults=True,
    )
    vector = observation_dict_to_vector(
        observation,
        allow_missing_defaults=True,
    )

    assert infer_observation_schema_version(observation) == 2
    assert normalized["attack_army_peak"] == 0.0
    assert normalized["army_lost_from_peak"] == 0.0
    assert normalized["army_lost_from_peak_ratio"] == 0.0
    assert normalized["army_count_delta"] == 0.0
    assert vector.shape == (len(OBSERVATION_FIELDS),)


@pytest.mark.unit
def test_observation_v1_rows_are_rejected_without_defaulting() -> None:
    observation = {field: 1.0 for field in OBSERVATION_FIELDS_V1}

    with pytest.raises(ValueError, match="base_under_threat"):
        validate_observation_dict(observation)


@pytest.mark.unit
def test_jsonl_trajectory_recorder_writes_one_record(tmp_path) -> None:
    output = tmp_path / "trajectory.jsonl"
    step = TrajectoryStep(
        episode_id="episode-1",
        step=8,
        map_name="AcropolisLE",
        difficulty="Easy",
        opponent_race="Protoss",
        observation={"army_count": 5.0},
        action=1,
        action_name="ATTACK_MAIN",
        reward=0.5,
        opponent_ai_build="Rush",
    )

    with JsonlTrajectoryRecorder(output) as recorder:
        recorder.record(step)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "episode_id": "episode-1",
            "step": 8,
            "map_name": "AcropolisLE",
            "difficulty": "Easy",
            "opponent_race": "Protoss",
            "observation": {"army_count": 5.0},
            "action": 1,
            "action_name": "ATTACK_MAIN",
            "reward": 0.5,
            "done": False,
            "result": None,
            "opponent_ai_build": "Rush",
        }
    ]


@pytest.mark.unit
def test_jsonl_trajectory_recorder_writes_strategy_record(tmp_path) -> None:
    output = tmp_path / "strategy.jsonl"
    strategy_observation_details = {
        "ready_photon_cannons": 1.0,
        "pending_photon_cannons": 0.0,
        "ready_shield_batteries": 2.0,
        "pending_shield_batteries": 1.0,
    }
    assert tuple(strategy_observation_details) == STRATEGY_OBSERVATION_DETAIL_FIELDS
    step = StrategyTrajectoryStep(
        episode_id="episode-1",
        step=64,
        map_name="AcropolisLE",
        difficulty="Easy",
        opponent_race="Protoss",
        strategy_observation={"own_bases": 1.0},
        strategy_observation_details=strategy_observation_details,
        strategy_action=3,
        strategy_action_name="TECH_ROBO",
        strategy_policy_source="coverage-teacher",
        strategy_policy_reason="base_threat_static_defense_gap",
        army_observation={"army_count": 12.0},
        army_action=1,
        army_action_name="ATTACK_MAIN",
        reward=0.25,
        opponent_ai_build="Air",
    )

    with JsonlTrajectoryRecorder(output) as recorder:
        recorder.record(step)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "episode_id": "episode-1",
            "step": 64,
            "map_name": "AcropolisLE",
            "difficulty": "Easy",
            "opponent_race": "Protoss",
            "strategy_observation": {"own_bases": 1.0},
            "strategy_observation_details": strategy_observation_details,
            "strategy_action": 3,
            "strategy_action_name": "TECH_ROBO",
            "strategy_policy_source": "coverage-teacher",
            "strategy_policy_reason": "base_threat_static_defense_gap",
            "army_observation": {"army_count": 12.0},
            "army_action": 1,
            "army_action_name": "ATTACK_MAIN",
            "reward": 0.25,
            "done": False,
            "result": None,
            "opponent_ai_build": "Air",
            "tactic_id": None,
            "tactic_phase": None,
            "tactic_source": None,
            "tactic_started_game_time": None,
            "tactic_switch_reason": None,
            "tactic_previous_id": None,
            "strategy_action_before_tactic_filter": None,
            "strategy_action_before_tactic_filter_name": None,
            "strategy_action_after_tactic_filter": None,
            "strategy_action_after_tactic_filter_name": None,
            "strategy_execution_attempted": None,
            "strategy_execution_effect": None,
            "strategy_execution_blocker": None,
            "strategy_execution_unit_type": None,
            "strategy_execution_target": None,
        }
    ]
    assert tuple(rows[0]["strategy_observation_details"]) == (
        STRATEGY_OBSERVATION_DETAIL_FIELDS
    )
