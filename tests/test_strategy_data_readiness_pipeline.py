from __future__ import annotations

import json

import pytest

from rl.experiments import read_json
from rl.strategy_data_readiness_pipeline import run_strategy_data_readiness_pipeline
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.run_strategy_data_readiness_pipeline import (
    format_strategy_data_readiness_pipeline,
    pipeline_exit_code,
)


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "game_time": 100.0,
            "minerals": 25.0,
            "vespene": 0.0,
            "supply_left": 0.0,
            "workers": 18.0,
            "own_bases": 1.0,
            "ready_gateways": 4.0,
            "ready_static_defense": 2.0,
            "pending_static_defense": 0.0,
            "base_under_threat": 1.0,
            "base_under_air_threat": 1.0,
            "base_under_ground_threat": 1.0,
            "army_count": 3.0,
            "zealots": 3.0,
            "stalkers": 0.0,
            "sentries": 0.0,
        }
    )
    observation.update(overrides)
    return observation


def _details(**overrides: float) -> dict[str, float]:
    details = {
        "ready_photon_cannons": 1.0,
        "pending_photon_cannons": 0.0,
        "ready_shield_batteries": 1.0,
        "pending_shield_batteries": 0.0,
    }
    details.update(overrides)
    return details


def _row(**overrides) -> dict:
    row = {
        "step": 64,
        "map_name": "AcropolisLE",
        "difficulty": "Hard",
        "opponent_race": "Terran",
        "opponent_ai_build": "Power",
        "strategy_observation": _observation(),
        "strategy_observation_details": _details(),
        "strategy_action": 0,
        "strategy_action_name": "STAY_COURSE",
        "strategy_policy_source": "coverage-teacher",
        "strategy_policy_reason": "no_strategy_rule_triggered",
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
def test_data_readiness_pipeline_promotes_detail_ready_data(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(trajectory, [_row(), _row(done=True, result="Result.Defeat")])

    result = run_strategy_data_readiness_pipeline(
        trajectory,
        output_dir=tmp_path / "runs",
        prefix="unit_ready",
    )

    assert result.recommendation == "train"
    assert result.training_ready is True
    assert result.trajectory_detail_ready is True
    assert result.policy_explanation_ready is True
    assert result.observation_detail_ready is True
    assert result.blocking_reasons == []
    assert result.inputs == [str(trajectory)]
    assert result.artifacts["trajectory_detail_gate"].endswith(
        "unit_ready_trajectory_detail_gate.json"
    )
    for path in result.artifacts.values():
        assert path
        assert read_json(path)

    summary = read_json(tmp_path / "runs" / "unit_ready_summary.json")
    assert summary["training_ready"] is True
    assert summary["policy_explanation_ready"] is True
    assert summary["observation_detail_ready"] is True


@pytest.mark.unit
def test_data_readiness_pipeline_holds_old_data_without_details(tmp_path) -> None:
    trajectory = tmp_path / "old_strategy.jsonl"
    row = _row()
    row.pop("strategy_observation_details")
    _write_jsonl(trajectory, [row, _row(done=True, result="Result.Defeat")])

    result = run_strategy_data_readiness_pipeline(
        trajectory,
        output_dir=tmp_path / "runs",
        prefix="unit_hold",
    )

    assert result.recommendation == "hold"
    assert result.training_ready is False
    assert result.trajectory_detail_ready is False
    assert result.policy_explanation_ready is True
    assert result.observation_detail_ready is False
    assert result.blocking_reasons == [
        "trajectory_detail_gate_not_ready",
        "observation_detail_gate_not_ready",
    ]

    readiness = read_json(result.artifacts["training_readiness"])
    assert readiness["trajectory_detail_gate_ready"] is False
    assert readiness["policy_explanation_gate_ready"] is True
    assert readiness["observation_detail_gate_ready"] is False


@pytest.mark.unit
def test_data_readiness_pipeline_holds_data_without_policy_explanations(
    tmp_path,
) -> None:
    trajectory = tmp_path / "unexplained_strategy.jsonl"
    row = _row()
    row.pop("strategy_policy_source")
    row.pop("strategy_policy_reason")
    _write_jsonl(trajectory, [row, _row(done=True, result="Result.Defeat")])

    result = run_strategy_data_readiness_pipeline(
        trajectory,
        output_dir=tmp_path / "runs",
        prefix="unit_unexplained",
    )

    assert result.recommendation == "hold"
    assert result.training_ready is False
    assert result.trajectory_detail_ready is True
    assert result.policy_explanation_ready is False
    assert result.observation_detail_ready is True
    assert result.blocking_reasons == [
        "policy_explanation_gate_not_ready",
    ]

    readiness = read_json(result.artifacts["training_readiness"])
    assert readiness["policy_explanation_gate_ready"] is False
    assert readiness["policy_explanation_gate_blocking_reasons"] == [
        "policy_source_coverage_low",
        "policy_reason_coverage_low",
    ]


@pytest.mark.unit
def test_format_data_readiness_pipeline_includes_artifacts(tmp_path) -> None:
    trajectory = tmp_path / "old_strategy.jsonl"
    row = _row()
    row.pop("strategy_observation_details")
    _write_jsonl(trajectory, [row])

    report = format_strategy_data_readiness_pipeline(
        run_strategy_data_readiness_pipeline(
            trajectory,
            output_dir=tmp_path / "runs",
            prefix="unit_report",
        )
    )

    assert "Strategy data readiness pipeline" in report
    assert "recommendation: hold" in report
    assert "training_ready: false" in report
    assert "trajectory_detail_gate: hold" in report
    assert "policy_explanation_gate: ready" in report
    assert "observation_detail_gate: hold" in report
    assert "artifacts:" in report


@pytest.mark.unit
def test_data_readiness_pipeline_exit_code_respects_fail_on_hold(tmp_path) -> None:
    ready_trajectory = tmp_path / "ready_strategy.jsonl"
    hold_trajectory = tmp_path / "hold_strategy.jsonl"
    hold_row = _row()
    hold_row.pop("strategy_observation_details")
    _write_jsonl(ready_trajectory, [_row()])
    _write_jsonl(hold_trajectory, [hold_row])

    ready = run_strategy_data_readiness_pipeline(
        ready_trajectory,
        output_dir=tmp_path / "runs",
        prefix="unit_exit_ready",
    )
    hold = run_strategy_data_readiness_pipeline(
        hold_trajectory,
        output_dir=tmp_path / "runs",
        prefix="unit_exit_hold",
    )

    assert pipeline_exit_code(ready, fail_on_hold=True) == 0
    assert pipeline_exit_code(hold, fail_on_hold=False) == 0
    assert pipeline_exit_code(hold, fail_on_hold=True) == 1
