from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl.strategy_policy_explanation_gate import (
    evaluate_strategy_policy_explanation_gate,
)
from scripts.gate_strategy_policy_explanations import (
    format_strategy_policy_explanation_gate,
)


def _row(**overrides) -> dict:
    row = {
        "step": 64,
        "strategy_observation": {},
        "strategy_action": 5,
        "strategy_action_name": "BUILD_STATIC_DEFENSE",
        "strategy_policy_source": "coverage-teacher",
        "strategy_policy_reason": "base_threat_static_defense_gap",
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
def test_policy_explanation_gate_promotes_complete_metadata(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(),
            _row(
                step=128,
                strategy_policy_source="tactic-aware-rule",
                strategy_policy_reason=(
                    "tactic_filter_ROBO_TIMING_TECH_ROBO_to_PRODUCE_ARMY"
                ),
            ),
            _row(done=True),
        ],
    )

    result = evaluate_strategy_policy_explanation_gate(trajectory)

    assert result.recommendation == "ready"
    assert result.ready is True
    assert result.blocking_reasons == []
    assert result.inputs == [str(trajectory)]
    assert result.rows == 2
    assert result.policy_source_rows == 2
    assert result.policy_reason_rows == 2
    assert result.missing_policy_source_rows == 0
    assert result.missing_policy_reason_rows == 0
    assert result.policy_source_counts == {
        "coverage-teacher": 1,
        "tactic-aware-rule": 1,
    }
    assert asdict(result)["files"] == 1


@pytest.mark.unit
def test_policy_explanation_gate_holds_missing_metadata(tmp_path) -> None:
    trajectory = tmp_path / "old_strategy.jsonl"
    row = _row()
    row.pop("strategy_policy_source")
    row["strategy_policy_reason"] = ""
    _write_jsonl(trajectory, [row])

    result = evaluate_strategy_policy_explanation_gate(trajectory)

    assert result.recommendation == "hold"
    assert result.ready is False
    assert result.blocking_reasons == [
        "policy_source_coverage_low",
        "policy_reason_coverage_low",
    ]
    assert result.policy_source_rows == 0
    assert result.policy_reason_rows == 0
    assert result.missing_policy_source_rows == 1
    assert result.missing_policy_reason_rows == 1


@pytest.mark.unit
def test_format_policy_explanation_gate_includes_counts(tmp_path) -> None:
    trajectory = tmp_path / "old_strategy.jsonl"
    row = _row()
    row.pop("strategy_policy_source")
    _write_jsonl(trajectory, [row])

    report = format_strategy_policy_explanation_gate(
        evaluate_strategy_policy_explanation_gate(trajectory)
    )

    assert "Strategy policy explanation gate" in report
    assert "recommendation: hold" in report
    assert "policy_sources: 0/1 ratio=0.000" in report
    assert "policy_source_coverage_low" in report
