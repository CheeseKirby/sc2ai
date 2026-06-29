from __future__ import annotations

import json

import pytest

from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from rl.strategy_veto_audit import audit_strategy_veto_baseline
from scripts.audit_strategy_veto import format_strategy_veto_audit


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
            "ready_gateways": 2.0,
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
def test_strategy_veto_audit_hard_vetoes_non_executable_records(tmp_path) -> None:
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
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    audit = audit_strategy_veto_baseline(trajectory)

    assert audit.records == 1
    assert audit.hard_veto_records == 1
    assert audit.bad_records == 1
    assert audit.bad_records_hard_vetoed == 1
    assert audit.bad_capture_ratio == 1.0
    assert audit.accept_positive_false_veto_ratio == 0.0
    assert audit.hard_veto_by_training_use == {"drop_non_executable": 1}
    assert audit.hard_veto_by_action == {"BUILD_STATIC_DEFENSE": 1}
    assert audit.hard_veto_by_reason == {
        "not_executable:cannot_afford_static_defense": 1
    }


@pytest.mark.unit
def test_strategy_veto_audit_does_not_veto_positive_payoff(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=170.0,
                    ready_gateways=2.0,
                    pending_gateways=0.0,
                ),
                done=True,
                result="Result.Victory",
            ),
        ],
    )

    audit = audit_strategy_veto_baseline(trajectory)

    assert audit.records == 1
    assert audit.hard_veto_records == 0
    assert audit.accept_positive_records == 1
    assert audit.accept_positive_records_hard_vetoed == 0


@pytest.mark.unit
def test_strategy_veto_audit_marks_static_defense_review_rows(tmp_path) -> None:
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
                    minerals=150.0,
                    has_cybernetics_core=1.0,
                    pending_static_defense=0.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                ),
            ),
            _row(done=True, result="Result.Tie"),
        ],
    )

    audit = audit_strategy_veto_baseline(trajectory)

    assert audit.hard_veto_records == 0
    assert audit.review_records == 1
    assert audit.review_by_reason == {"static_defense_available_under_threat": 1}
    assert audit.review_by_action == {"PRODUCE_ARMY": 1}


@pytest.mark.unit
def test_format_strategy_veto_audit_contains_summary_and_decisions(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_action=3,
                strategy_action_name="TECH_ROBO",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=40.0,
                    vespene=150.0,
                    pending_robo=0.0,
                    ready_robo=0.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    report = format_strategy_veto_audit(
        audit_strategy_veto_baseline(trajectory),
        show_decisions=True,
    )

    assert "Strategy veto audit" in report
    assert "hard_veto_records: 1" in report
    assert "bad_capture:" in report
    assert "not_executable:cannot_afford_robo=1" in report
    assert "decisions:" in report
    assert "action=TECH_ROBO" in report
