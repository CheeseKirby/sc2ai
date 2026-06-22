from __future__ import annotations

import json

import pytest

from rl.diagnostics import diagnose_trajectories
from rl.observations import OBSERVATION_FIELDS, OBSERVATION_FIELDS_V1
from scripts.diagnose_trajectories import format_diagnostics


def _observation(value: float) -> dict[str, float]:
    return {field: value for field in OBSERVATION_FIELDS}


def _v1_observation(value: float) -> dict[str, float]:
    return {field: value for field in OBSERVATION_FIELDS_V1}


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_diagnose_trajectories_reports_training_action_coverage(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(
        first,
        [
            {
                "step": 8,
                "observation": _observation(1.0),
                "action": 0,
                "done": False,
            },
            {
                "step": 16,
                "observation": _observation(2.0),
                "action": 1,
                "done": True,
                "result": "Victory",
            },
        ],
    )
    _write_jsonl(
        second,
        [
            {
                "step": 8,
                "observation": _observation(3.0),
                "action": 1,
                "done": False,
            },
        ],
    )

    diagnostics = diagnose_trajectories(tmp_path)

    assert diagnostics.files == 2
    assert diagnostics.rows == 3
    assert diagnostics.training_rows == 2
    assert diagnostics.terminal_rows == 1
    assert diagnostics.files_missing_terminal == 1
    assert diagnostics.result_counts == {"Victory": 1}
    assert diagnostics.observation_schema_counts == {"3": 3}
    assert diagnostics.rows_defaulted_observation_fields == 0
    assert diagnostics.observation_feature_stats["army_count"] == {
        "min": 1.0,
        "max": 3.0,
        "avg": 2.0,
    }
    assert diagnostics.observation_feature_stats["enemy_to_home_distance"] == {
        "min": 1.0,
        "max": 3.0,
        "avg": 2.0,
    }
    assert diagnostics.action_counts_by_name == {"RALLY": 1, "ATTACK_MAIN": 1}
    assert diagnostics.missing_action_names == [
        "RETREAT_HOME",
        "DEFEND_BASE",
        "HOLD",
    ]
    assert diagnostics.min_action_count == 10
    assert diagnostics.low_count_action_names == ["RALLY", "ATTACK_MAIN"]
    assert diagnostics.action_coverage == pytest.approx(0.4)
    assert diagnostics.rows_per_file == {"min": 1.0, "max": 2.0, "avg": 1.5}
    assert any("missing action coverage" in item for item in diagnostics.warnings)
    assert any("low action counts" in item for item in diagnostics.warnings)


@pytest.mark.unit
def test_diagnose_trajectories_low_count_warning_can_be_disabled(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 8,
                "observation": _observation(1.0),
                "action": 0,
                "done": False,
            },
            {
                "step": 16,
                "observation": _observation(2.0),
                "action": 0,
                "done": True,
                "result": "Victory",
            },
        ],
    )

    diagnostics = diagnose_trajectories(trajectory, min_action_count=0)

    assert diagnostics.min_action_count == 0
    assert diagnostics.low_count_action_names == []
    assert not any("low action counts" in item for item in diagnostics.warnings)


@pytest.mark.unit
def test_diagnose_trajectories_reports_invalid_rows(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        "\n".join(
            [
                "{bad-json",
                json.dumps(
                    {
                        "step": 8,
                        "observation": {"game_time": 1.0},
                        "action": 99,
                    }
                ),
                json.dumps(
                    {
                        "step": 16,
                        "observation": _observation(2.0),
                        "action": 0,
                        "done": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    diagnostics = diagnose_trajectories(trajectory)

    assert diagnostics.invalid_json_rows == 1
    assert diagnostics.invalid_action_rows == 1
    assert diagnostics.invalid_observation_rows == 1
    assert diagnostics.rows_missing_done == 1
    assert diagnostics.terminal_rows_missing_result == 1
    assert diagnostics.result_counts == {"NO_RESULT": 1}


@pytest.mark.unit
def test_diagnose_trajectories_reports_v1_observation_defaulting(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 8,
                "observation": _v1_observation(1.0),
                "action": 0,
                "done": False,
            },
            {
                "step": 16,
                "observation": _v1_observation(2.0),
                "action": 0,
                "done": True,
                "result": "Victory",
            },
        ],
    )

    diagnostics = diagnose_trajectories(trajectory)

    assert diagnostics.observation_schema_counts == {"1": 2}
    assert diagnostics.rows_defaulted_observation_fields == 2
    assert diagnostics.invalid_observation_rows == 0
    assert any(
        "current-schema default fields" in item for item in diagnostics.warnings
    )


@pytest.mark.unit
def test_format_diagnostics_includes_key_sections(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 8,
                "observation": _observation(1.0),
                "action": 0,
                "done": False,
            },
            {
                "step": 16,
                "observation": _observation(2.0),
                "action": 0,
                "done": True,
                "result": "Tie",
            },
        ],
    )

    report = format_diagnostics(diagnose_trajectories(trajectory), show_files=True)

    assert "Trajectory diagnostics" in report
    assert "observation_schemas:" in report
    assert "rows_defaulted_observation_fields: 0" in report
    assert "observation_feature_stats:" in report
    assert "army_count: min=1.000 max=2.000 avg=1.500" in report
    assert "enemy_to_home_distance: min=1.000 max=2.000 avg=1.500" in report
    assert "actions:" in report
    assert "RALLY: 1" in report
    assert "missing_actions:" in report
    assert "low_count_actions (<10):" in report
    assert "results:" in report
    assert "Tie: 1" in report
    assert str(trajectory) in report
