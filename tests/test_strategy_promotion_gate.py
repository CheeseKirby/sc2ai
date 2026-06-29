from __future__ import annotations

from dataclasses import asdict

import pytest

from rl.experiments import write_json
from rl.strategy_promotion_gate import (
    StrategyPromotionGateConfig,
    evaluate_strategy_promotion_gate,
)
from scripts.gate_strategy_promotion import format_strategy_promotion_gate


def _audit(**overrides) -> dict:
    payload = {
        "inputs": ["data/trajectories/detail_ready"],
        "checkpoint_path": "runs/example/checkpoints/policy.pt",
        "prediction_mode": "action-critic-mask",
        "rows": 10,
        "warnings": [],
        "prediction_matches_recorded": 7,
        "prediction_match_ratio": 0.7,
        "accept_positive_prediction_matches": 3,
        "accept_positive_rows": 4,
        "veto_negative_prediction_matches": 0,
        "veto_negative_rows": 2,
        "drop_non_executable_prediction_matches": 0,
        "drop_non_executable_rows": 2,
        "action_space_exhausted_prediction_matches": 0,
        "action_space_exhausted_rows": 0,
        "predicted_non_executable_rows": 0,
        "predicted_non_executable_ratio": 0.0,
        "action_critic_fallback_rows": 0,
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_strategy_promotion_gate_promotes_clean_audit(tmp_path) -> None:
    audit_path = tmp_path / "clean.json"
    write_json(audit_path, _audit())

    result = evaluate_strategy_promotion_gate([audit_path])

    assert result.recommendation == "promote"
    assert result.promotable is True
    assert result.selected_audit_path == str(audit_path)
    assert result.blocking_reasons == []
    assert result.candidates[0].promotable is True
    assert asdict(result)["candidates"][0]["prediction_matches_recorded"] == 7


@pytest.mark.unit
def test_strategy_promotion_gate_promotes_with_matching_observation_detail_gate(
    tmp_path,
) -> None:
    audit_path = tmp_path / "clean.json"
    gate_path = tmp_path / "observation_detail_gate.json"
    write_json(audit_path, _audit())
    write_json(
        gate_path,
        {
            "ready": True,
            "blocking_reasons": [],
            "analysis_inputs": ["data/trajectories/detail_ready"],
        },
    )

    result = evaluate_strategy_promotion_gate(
        [audit_path],
        config=StrategyPromotionGateConfig(
            observation_detail_gate_path=str(gate_path),
        ),
    )

    assert result.recommendation == "promote"
    assert result.promotable is True
    assert result.candidates[0].observation_detail_gate_ready is True
    assert result.candidates[0].observation_detail_gate_inputs == [
        "data/trajectories/detail_ready",
    ]


@pytest.mark.unit
def test_strategy_promotion_gate_blocks_failed_observation_detail_gate(
    tmp_path,
) -> None:
    audit_path = tmp_path / "clean.json"
    gate_path = tmp_path / "observation_detail_gate.json"
    write_json(audit_path, _audit())
    write_json(
        gate_path,
        {
            "ready": False,
            "blocking_reasons": ["observation_detail_coverage_low"],
            "analysis_inputs": ["data/trajectories/detail_ready"],
        },
    )

    result = evaluate_strategy_promotion_gate(
        [audit_path],
        config=StrategyPromotionGateConfig(
            observation_detail_gate_path=str(gate_path),
        ),
    )

    assert result.recommendation == "hold"
    assert result.promotable is False
    assert result.candidates[0].blocking_reasons == [
        "observation_detail_gate_not_ready",
    ]
    assert result.candidates[0].observation_detail_gate_blocking_reasons == [
        "observation_detail_coverage_low",
    ]


@pytest.mark.unit
def test_strategy_promotion_gate_blocks_mismatched_observation_detail_gate_inputs(
    tmp_path,
) -> None:
    audit_path = tmp_path / "clean.json"
    gate_path = tmp_path / "observation_detail_gate.json"
    write_json(audit_path, _audit())
    write_json(
        gate_path,
        {
            "ready": True,
            "blocking_reasons": [],
            "analysis_inputs": ["data/trajectories/other_dataset"],
        },
    )

    result = evaluate_strategy_promotion_gate(
        [audit_path],
        config=StrategyPromotionGateConfig(
            observation_detail_gate_path=str(gate_path),
        ),
    )

    assert result.recommendation == "hold"
    assert result.promotable is False
    assert result.candidates[0].blocking_reasons == [
        "observation_detail_gate_inputs_mismatch",
    ]


@pytest.mark.unit
def test_strategy_promotion_gate_blocks_bad_and_fallback_matches(tmp_path) -> None:
    audit_path = tmp_path / "blocked.json"
    write_json(
        audit_path,
        _audit(
            veto_negative_prediction_matches=1,
            action_space_exhausted_prediction_matches=1,
            action_space_exhausted_rows=1,
            action_critic_fallback_rows=5,
            warnings=["missing_signal_rows:1"],
        ),
    )

    result = evaluate_strategy_promotion_gate([audit_path])

    assert result.recommendation == "hold"
    assert result.promotable is False
    assert result.candidates[0].blocking_reasons == [
        "predicted_matches_veto_negative_labels",
        "predicted_matches_action_space_exhausted_labels",
        "action_critic_fallback_rows_high",
        "audit_warnings_present",
    ]
    assert result.blocking_reasons == [
        "action_critic_fallback_rows_high",
        "audit_warnings_present",
        "predicted_matches_action_space_exhausted_labels",
        "predicted_matches_veto_negative_labels",
    ]


@pytest.mark.unit
def test_strategy_promotion_gate_ranks_less_bad_candidate(tmp_path) -> None:
    worse = tmp_path / "worse.json"
    better = tmp_path / "better.json"
    write_json(
        worse,
        _audit(
            checkpoint_path="runs/worse/checkpoints/policy.pt",
            veto_negative_prediction_matches=5,
            action_critic_fallback_rows=20,
            prediction_match_ratio=0.8,
        ),
    )
    write_json(
        better,
        _audit(
            checkpoint_path="runs/better/checkpoints/policy.pt",
            veto_negative_prediction_matches=1,
            action_critic_fallback_rows=0,
            prediction_match_ratio=0.4,
        ),
    )

    result = evaluate_strategy_promotion_gate([worse, better])

    assert result.recommendation == "hold"
    assert result.selected_audit_path == str(better)
    assert result.selected_checkpoint_path == "runs/better/checkpoints/policy.pt"
    assert result.candidates[0].veto_negative_prediction_matches == 1
    assert result.candidates[1].veto_negative_prediction_matches == 5


@pytest.mark.unit
def test_format_strategy_promotion_gate_includes_candidate_metrics(tmp_path) -> None:
    audit_path = tmp_path / "blocked.json"
    write_json(audit_path, _audit(veto_negative_prediction_matches=1))

    report = format_strategy_promotion_gate(
        evaluate_strategy_promotion_gate([audit_path])
    )

    assert "Strategy promotion gate" in report
    assert "recommendation: hold" in report
    assert "veto=1/2" in report
    assert "space=0/0" in report
    assert "predicted_matches_veto_negative_labels" in report
