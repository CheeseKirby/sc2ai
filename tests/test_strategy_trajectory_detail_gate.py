from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl.strategy_trajectory_detail_gate import (
    evaluate_strategy_trajectory_detail_gate,
)
from scripts.gate_strategy_trajectory_details import (
    format_strategy_trajectory_detail_gate,
)


def _row(**overrides) -> dict:
    row = {
        "step": 64,
        "strategy_observation": {
            "ready_static_defense": 3.0,
            "pending_static_defense": 1.0,
        },
        "strategy_observation_details": {
            "ready_photon_cannons": 1.0,
            "pending_photon_cannons": 0.0,
            "ready_shield_batteries": 2.0,
            "pending_shield_batteries": 1.0,
        },
        "strategy_action": 0,
        "strategy_action_name": "STAY_COURSE",
        "done": False,
    }
    row.update(overrides)
    return row


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_trajectory_detail_gate_promotes_complete_matching_details(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(trajectory, [_row(), _row(step=128), _row(done=True)])

    result = evaluate_strategy_trajectory_detail_gate(trajectory)

    assert result.recommendation == "ready"
    assert result.ready is True
    assert result.blocking_reasons == []
    assert result.inputs == [str(trajectory)]
    assert result.rows == 2
    assert result.observation_detail_rows == 2
    assert result.observation_detail_complete_rows == 2
    assert result.ready_static_defense_mismatch_rows == 0
    assert result.pending_static_defense_mismatch_rows == 0
    assert asdict(result)["files"] == 1


@pytest.mark.unit
def test_trajectory_detail_gate_holds_missing_details(tmp_path) -> None:
    trajectory = tmp_path / "old_strategy.jsonl"
    row = _row()
    row.pop("strategy_observation_details")
    _write_jsonl(trajectory, [row])

    result = evaluate_strategy_trajectory_detail_gate(trajectory)

    assert result.recommendation == "hold"
    assert result.ready is False
    assert result.blocking_reasons == [
        "observation_detail_coverage_low",
        "observation_detail_complete_coverage_low",
    ]
    assert result.observation_detail_rows == 0


@pytest.mark.unit
def test_trajectory_detail_gate_holds_missing_fields_and_mismatches(tmp_path) -> None:
    trajectory = tmp_path / "bad_strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_observation_details={
                    "ready_photon_cannons": 1.0,
                    "pending_photon_cannons": 0.0,
                    "ready_shield_batteries": 1.0,
                },
            ),
            _row(
                step=128,
                strategy_observation={"ready_static_defense": 5.0},
            ),
        ],
    )

    result = evaluate_strategy_trajectory_detail_gate(trajectory)

    assert result.recommendation == "hold"
    assert result.ready is False
    assert result.blocking_reasons == [
        "observation_detail_complete_coverage_low",
        "ready_static_defense_detail_mismatch",
        "pending_static_defense_detail_mismatch",
    ]
    assert result.missing_detail_field_counts == {"pending_shield_batteries": 1}
    assert result.ready_static_defense_mismatch_rows == 2
    assert result.pending_static_defense_mismatch_rows == 1


@pytest.mark.unit
def test_format_trajectory_detail_gate_includes_counts(tmp_path) -> None:
    trajectory = tmp_path / "old_strategy.jsonl"
    row = _row()
    row.pop("strategy_observation_details")
    _write_jsonl(trajectory, [row])

    report = format_strategy_trajectory_detail_gate(
        evaluate_strategy_trajectory_detail_gate(trajectory)
    )

    assert "Strategy trajectory detail gate" in report
    assert "recommendation: hold" in report
    assert "inputs:" in report
    assert "observation_details: 0/1 ratio=0.000" in report
    assert "observation_detail_coverage_low" in report
