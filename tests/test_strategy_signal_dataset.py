from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from rl.strategy_signal_dataset import build_strategy_signal_dataset
from scripts.build_strategy_signal_dataset import format_strategy_signal_dataset


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "game_time": 100.0,
            "minerals": 200.0,
            "vespene": 150.0,
            "supply_left": 8.0,
            "workers": 24.0,
            "own_bases": 1.0,
            "ready_gateways": 1.0,
            "has_cybernetics_core": 1.0,
            "army_count": 8.0,
            "worker_saturation_ratio": 0.9,
        }
    )
    observation.update(overrides)
    return observation


def _row(**overrides) -> dict:
    row = {
        "step": 64,
        "map_name": "AcropolisLE",
        "difficulty": "Hard",
        "opponent_race": "Terran",
        "opponent_ai_build": "Power",
        "strategy_observation": _observation(),
        "strategy_action": 0,
        "strategy_action_name": "STAY_COURSE",
        "done": False,
    }
    row.update(overrides)
    return row


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_strategy_signal_dataset_marks_observed_payoff_as_positive(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
                strategy_execution_attempted=True,
                strategy_execution_effect="build_structure",
                strategy_execution_unit_type="GATEWAY",
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=125.0,
                    ready_gateways=1.0,
                    pending_gateways=1.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=170.0,
                    ready_gateways=2.0,
                    pending_gateways=0.0,
                ),
            ),
            _row(done=True, result="Result.Victory"),
        ],
    )

    dataset = build_strategy_signal_dataset(trajectory)

    assert dataset.files == 1
    assert dataset.training_rows == 3
    positive = next(
        record
        for record in dataset.records
        if record.candidate_action == "ADD_GATEWAYS"
    )
    assert positive.candidate_source == "recorded"
    assert positive.immediate_executable is True
    assert positive.payoff_observed is True
    assert positive.label_quality == "good"
    assert positive.recommended_training_use == "accept_positive"
    assert "pending_gateway_seen" in positive.payoff_events_by_window["30s"]
    assert "ready_gateway_increased" in positive.payoff_events_by_window["120s"]
    assert dataset.records_by_training_use["accept_positive"] == 1
    assert asdict(dataset)["records"][0]["candidate_action"] == "ADD_GATEWAYS"


@pytest.mark.unit
def test_strategy_signal_dataset_drops_non_executable_rows(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=25.0,
                    ready_static_defense=1.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                ),
                strategy_execution_attempted=False,
                strategy_execution_effect="noop",
                strategy_execution_blocker="cannot_afford_static_defense",
            ),
            _row(
                strategy_observation=_observation(
                    game_time=160.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                    ready_gateways=2.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    dataset = build_strategy_signal_dataset(trajectory)
    static_record = next(
        record
        for record in dataset.records
        if record.candidate_action == "BUILD_STATIC_DEFENSE"
    )

    assert static_record.immediate_executable is False
    assert static_record.candidate_blocker == "cannot_afford_static_defense"
    assert static_record.label_quality == "bad"
    assert static_record.recommended_training_use == "drop_non_executable"
    assert "candidate_not_executable:cannot_afford_static_defense" in static_record.reasons


@pytest.mark.unit
def test_strategy_signal_dataset_marks_bad_production_under_threat_as_veto(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=10.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                    ready_gateways=2.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                    ready_gateways=2.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    dataset = build_strategy_signal_dataset(trajectory)
    produce = next(
        record
        for record in dataset.records
        if record.candidate_action == "PRODUCE_ARMY"
    )

    assert produce.immediate_executable is True
    assert produce.label_quality == "bad"
    assert produce.recommended_training_use == "veto_negative"
    assert "threat_persisted" in produce.negative_events_by_window["120s"]
    assert "army_count_decreased" in produce.negative_events_by_window["120s"]


@pytest.mark.unit
def test_strategy_signal_dataset_marks_waiting_under_persistent_threat_as_veto(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=10.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=10.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=10.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    dataset = build_strategy_signal_dataset(trajectory)
    wait = next(
        record
        for record in dataset.records
        if record.candidate_action == "STAY_COURSE"
    )

    assert wait.immediate_executable is True
    assert wait.label_quality == "bad"
    assert wait.recommended_training_use == "veto_negative"
    assert "threat_persisted" in wait.negative_events_by_window["120s"]
    assert "threat_persisted" in wait.reasons


@pytest.mark.unit
def test_strategy_signal_dataset_marks_waiting_with_no_macro_options_as_action_space_exhausted(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=25.0,
                    vespene=0.0,
                    supply_left=0.0,
                    own_bases=1.0,
                    ready_gateways=4.0,
                    pending_gateways=0.0,
                    has_cybernetics_core=0.0,
                    ready_forge=0.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    dataset = build_strategy_signal_dataset(trajectory)
    wait = next(
        record
        for record in dataset.records
        if record.candidate_action == "STAY_COURSE"
    )

    assert wait.label_quality == "bad"
    assert wait.recommended_training_use == "action_space_exhausted"
    assert "action_space_exhausted" in wait.reasons
    assert "threat_persisted" in wait.reasons
    assert dataset.records_by_training_use["action_space_exhausted"] == 1


@pytest.mark.unit
def test_strategy_signal_dataset_handles_before_filter_candidates(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                tactic_id="RECOVERY",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    minerals=150.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    ready_gateways=2.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=140.0,
                    base_under_threat=0.0,
                    ready_static_defense=1.0,
                    army_count=12.0,
                ),
            ),
            _row(done=True, result="Result.Victory"),
        ],
    )

    dataset = build_strategy_signal_dataset(trajectory)
    before_filter = next(
        record for record in dataset.records if record.candidate_source == "before_filter"
    )

    assert before_filter.recorded_action == "PRODUCE_ARMY"
    assert before_filter.candidate_action == "BUILD_STATIC_DEFENSE"
    assert before_filter.immediate_executable is True
    assert before_filter.label_quality == "unknown"
    assert before_filter.recommended_training_use == "needs_fresh_ab"
    assert before_filter.reasons == ["counterfactual_not_observed"]
    assert dataset.records_by_candidate_source == {"before_filter": 1, "recorded": 2}


@pytest.mark.unit
def test_format_strategy_signal_dataset_contains_summary_and_records(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_action=3,
                strategy_action_name="TECH_ROBO",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=200.0,
                    vespene=150.0,
                    pending_robo=0.0,
                    ready_robo=0.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=150.0,
                    pending_robo=1.0,
                    ready_robo=0.0,
                ),
            ),
            _row(done=True, result="Result.Tie"),
        ],
    )

    report = format_strategy_signal_dataset(
        build_strategy_signal_dataset(trajectory),
        show_records=True,
    )

    assert "Strategy signal dataset" in report
    assert "training_use:" in report
    assert "accept_positive=1" in report
    assert "candidate=TECH_ROBO" in report
    assert "pending_robo_seen" in report
