from __future__ import annotations

from dataclasses import asdict

import pytest

from rl.experiments import write_json
from rl.strategy_training_readiness import (
    StrategyTrainingReadinessConfig,
    evaluate_strategy_training_readiness,
)
from scripts.gate_strategy_training_readiness import format_strategy_training_readiness


@pytest.mark.unit
def test_strategy_training_readiness_promotes_ready_detail_gate(tmp_path) -> None:
    trajectory_gate = tmp_path / "trajectory_detail_gate.json"
    observation_gate = tmp_path / "observation_detail_gate.json"
    promotion_gate = tmp_path / "promotion_gate.json"
    write_json(
        trajectory_gate,
        {
            "ready": True,
            "blocking_reasons": [],
            "inputs": ["data/trajectories/detail_ready"],
        },
    )
    write_json(
        observation_gate,
        {
            "ready": True,
            "blocking_reasons": [],
            "analysis_inputs": ["data/trajectories/detail_ready"],
        },
    )
    write_json(
        promotion_gate,
        {
            "promotable": True,
            "blocking_reasons": [],
            "selected_checkpoint_path": "runs/example/checkpoints/policy.pt",
        },
    )

    result = evaluate_strategy_training_readiness(
        observation_gate,
        config=StrategyTrainingReadinessConfig(
            expected_inputs=("data/trajectories/detail_ready",),
            trajectory_detail_gate_path=str(trajectory_gate),
            promotion_gate_path=str(promotion_gate),
        ),
    )

    assert result.recommendation == "train"
    assert result.training_ready is True
    assert result.promotion_ready is True
    assert result.blocking_reasons == []
    assert result.trajectory_detail_gate_ready is True
    assert result.selected_checkpoint_path == "runs/example/checkpoints/policy.pt"
    assert asdict(result)["observation_detail_gate_ready"] is True


@pytest.mark.unit
def test_strategy_training_readiness_holds_failed_trajectory_detail_gate(
    tmp_path,
) -> None:
    trajectory_gate = tmp_path / "trajectory_detail_gate.json"
    observation_gate = tmp_path / "observation_detail_gate.json"
    write_json(
        trajectory_gate,
        {
            "ready": False,
            "blocking_reasons": [
                "observation_detail_coverage_low",
                "observation_detail_complete_coverage_low",
            ],
            "inputs": ["data/trajectories/old"],
        },
    )
    write_json(
        observation_gate,
        {
            "ready": True,
            "blocking_reasons": [],
            "analysis_inputs": ["data/trajectories/old"],
        },
    )

    result = evaluate_strategy_training_readiness(
        observation_gate,
        config=StrategyTrainingReadinessConfig(
            expected_inputs=("data/trajectories/old",),
            trajectory_detail_gate_path=str(trajectory_gate),
        ),
    )

    assert result.recommendation == "hold"
    assert result.training_ready is False
    assert result.blocking_reasons == ["trajectory_detail_gate_not_ready"]
    assert result.trajectory_detail_gate_blocking_reasons == [
        "observation_detail_coverage_low",
        "observation_detail_complete_coverage_low",
    ]


@pytest.mark.unit
def test_strategy_training_readiness_holds_failed_policy_explanation_gate(
    tmp_path,
) -> None:
    explanation_gate = tmp_path / "policy_explanation_gate.json"
    observation_gate = tmp_path / "observation_detail_gate.json"
    write_json(
        explanation_gate,
        {
            "ready": False,
            "blocking_reasons": [
                "policy_source_coverage_low",
                "policy_reason_coverage_low",
            ],
            "inputs": ["data/trajectories/old"],
        },
    )
    write_json(
        observation_gate,
        {
            "ready": True,
            "blocking_reasons": [],
            "analysis_inputs": ["data/trajectories/old"],
        },
    )

    result = evaluate_strategy_training_readiness(
        observation_gate,
        config=StrategyTrainingReadinessConfig(
            expected_inputs=("data/trajectories/old",),
            policy_explanation_gate_path=str(explanation_gate),
        ),
    )

    assert result.recommendation == "hold"
    assert result.training_ready is False
    assert result.blocking_reasons == ["policy_explanation_gate_not_ready"]
    assert result.policy_explanation_gate_ready is False
    assert result.policy_explanation_gate_blocking_reasons == [
        "policy_source_coverage_low",
        "policy_reason_coverage_low",
    ]


@pytest.mark.unit
def test_strategy_training_readiness_holds_mismatched_policy_explanation_inputs(
    tmp_path,
) -> None:
    explanation_gate = tmp_path / "policy_explanation_gate.json"
    observation_gate = tmp_path / "observation_detail_gate.json"
    write_json(
        explanation_gate,
        {
            "ready": True,
            "blocking_reasons": [],
            "inputs": ["data/trajectories/other"],
        },
    )
    write_json(
        observation_gate,
        {
            "ready": True,
            "blocking_reasons": [],
            "analysis_inputs": ["data/trajectories/current"],
        },
    )

    result = evaluate_strategy_training_readiness(
        observation_gate,
        config=StrategyTrainingReadinessConfig(
            expected_inputs=("data/trajectories/current",),
            policy_explanation_gate_path=str(explanation_gate),
        ),
    )

    assert result.recommendation == "hold"
    assert result.training_ready is False
    assert result.blocking_reasons == [
        "policy_explanation_gate_inputs_mismatch",
    ]


@pytest.mark.unit
def test_strategy_training_readiness_holds_mismatched_trajectory_detail_inputs(
    tmp_path,
) -> None:
    trajectory_gate = tmp_path / "trajectory_detail_gate.json"
    observation_gate = tmp_path / "observation_detail_gate.json"
    write_json(
        trajectory_gate,
        {
            "ready": True,
            "blocking_reasons": [],
            "inputs": ["data/trajectories/other"],
        },
    )
    write_json(
        observation_gate,
        {
            "ready": True,
            "blocking_reasons": [],
            "analysis_inputs": ["data/trajectories/current"],
        },
    )

    result = evaluate_strategy_training_readiness(
        observation_gate,
        config=StrategyTrainingReadinessConfig(
            expected_inputs=("data/trajectories/current",),
            trajectory_detail_gate_path=str(trajectory_gate),
        ),
    )

    assert result.recommendation == "hold"
    assert result.training_ready is False
    assert result.blocking_reasons == [
        "trajectory_detail_gate_inputs_mismatch",
    ]


@pytest.mark.unit
def test_strategy_training_readiness_holds_failed_detail_gate(tmp_path) -> None:
    observation_gate = tmp_path / "observation_detail_gate.json"
    write_json(
        observation_gate,
        {
            "ready": False,
            "blocking_reasons": [
                "observation_detail_coverage_low",
                "static_defense_type_ambiguous_rows_high",
            ],
            "analysis_inputs": ["data/trajectories/old"],
        },
    )

    result = evaluate_strategy_training_readiness(
        observation_gate,
        config=StrategyTrainingReadinessConfig(
            expected_inputs=("data/trajectories/old",),
        ),
    )

    assert result.recommendation == "hold"
    assert result.training_ready is False
    assert result.blocking_reasons == ["observation_detail_gate_not_ready"]
    assert result.observation_detail_gate_blocking_reasons == [
        "observation_detail_coverage_low",
        "static_defense_type_ambiguous_rows_high",
    ]


@pytest.mark.unit
def test_strategy_training_readiness_holds_mismatched_inputs(tmp_path) -> None:
    observation_gate = tmp_path / "observation_detail_gate.json"
    write_json(
        observation_gate,
        {
            "ready": True,
            "blocking_reasons": [],
            "analysis_inputs": ["data/trajectories/other"],
        },
    )

    result = evaluate_strategy_training_readiness(
        observation_gate,
        config=StrategyTrainingReadinessConfig(
            expected_inputs=("data/trajectories/current",),
        ),
    )

    assert result.recommendation == "hold"
    assert result.training_ready is False
    assert result.blocking_reasons == [
        "observation_detail_gate_inputs_mismatch",
    ]


@pytest.mark.unit
def test_format_strategy_training_readiness_includes_gate_state(tmp_path) -> None:
    observation_gate = tmp_path / "observation_detail_gate.json"
    write_json(
        observation_gate,
        {
            "ready": False,
            "blocking_reasons": ["observation_detail_coverage_low"],
            "analysis_inputs": ["data/trajectories/old"],
        },
    )

    report = format_strategy_training_readiness(
        evaluate_strategy_training_readiness(
            observation_gate,
            config=StrategyTrainingReadinessConfig(
                expected_inputs=("data/trajectories/old",),
            ),
        )
    )

    assert "Strategy training readiness" in report
    assert "recommendation: hold" in report
    assert "training_ready: false" in report
    assert "expected_inputs: data/trajectories/old" in report
    assert "trajectory_detail_gate: unchecked" in report
    assert "policy_explanation_gate: unchecked" in report
    assert "observation_detail_gate: hold" in report
    assert "observation_detail_coverage_low" in report
