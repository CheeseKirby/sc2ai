"""Aggregate pre-training readiness gates for strategy learning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.experiments import read_json


@dataclass(frozen=True)
class StrategyTrainingReadinessConfig:
    """Inputs expected by the training run being checked."""

    expected_inputs: tuple[str, ...] = ()
    trajectory_detail_gate_path: str | None = None
    policy_explanation_gate_path: str | None = None
    promotion_gate_path: str | None = None


@dataclass(frozen=True)
class StrategyTrainingReadinessResult:
    """One compact readiness decision before strategy training."""

    recommendation: str
    training_ready: bool
    promotion_ready: bool | None
    blocking_reasons: list[str]
    expected_inputs: list[str]
    trajectory_detail_gate_path: str | None
    trajectory_detail_gate_ready: bool | None
    trajectory_detail_gate_inputs: list[str]
    trajectory_detail_gate_blocking_reasons: list[str]
    policy_explanation_gate_path: str | None
    policy_explanation_gate_ready: bool | None
    policy_explanation_gate_inputs: list[str]
    policy_explanation_gate_blocking_reasons: list[str]
    observation_detail_gate_path: str
    observation_detail_gate_ready: bool
    observation_detail_gate_inputs: list[str]
    observation_detail_gate_blocking_reasons: list[str]
    promotion_gate_path: str | None
    promotion_gate_promotable: bool | None
    promotion_gate_blocking_reasons: list[str]
    selected_checkpoint_path: str | None


def evaluate_strategy_training_readiness(
    observation_detail_gate_path: str | Path,
    *,
    config: StrategyTrainingReadinessConfig | None = None,
) -> StrategyTrainingReadinessResult:
    """Return whether the current strategy data is ready for gated training."""
    readiness_config = config or StrategyTrainingReadinessConfig()
    observation_path = Path(observation_detail_gate_path)
    trajectory_gate = (
        read_json(readiness_config.trajectory_detail_gate_path)
        if readiness_config.trajectory_detail_gate_path is not None
        else None
    )
    explanation_gate = (
        read_json(readiness_config.policy_explanation_gate_path)
        if readiness_config.policy_explanation_gate_path is not None
        else None
    )
    observation_gate = read_json(observation_path)
    promotion_gate = (
        read_json(readiness_config.promotion_gate_path)
        if readiness_config.promotion_gate_path is not None
        else None
    )

    expected_inputs = [str(value) for value in readiness_config.expected_inputs]
    trajectory_inputs: list[str] = []
    trajectory_ready: bool | None = None
    trajectory_reasons: list[str] = []
    if trajectory_gate is not None:
        trajectory_inputs = _string_list(trajectory_gate.get("inputs"))
        trajectory_ready = bool(trajectory_gate.get("ready", False))
        trajectory_reasons = _string_list(trajectory_gate.get("blocking_reasons"))
    explanation_inputs: list[str] = []
    explanation_ready: bool | None = None
    explanation_reasons: list[str] = []
    if explanation_gate is not None:
        explanation_inputs = _string_list(explanation_gate.get("inputs"))
        explanation_ready = bool(explanation_gate.get("ready", False))
        explanation_reasons = _string_list(explanation_gate.get("blocking_reasons"))
    observation_inputs = _string_list(observation_gate.get("analysis_inputs"))
    observation_ready = bool(observation_gate.get("ready", False))
    observation_reasons = _string_list(observation_gate.get("blocking_reasons"))

    blocking_reasons = _training_blocking_reasons(
        trajectory_ready=trajectory_ready,
        trajectory_inputs=trajectory_inputs,
        explanation_ready=explanation_ready,
        explanation_inputs=explanation_inputs,
        observation_ready=observation_ready,
        expected_inputs=expected_inputs,
        observation_inputs=observation_inputs,
    )
    training_ready = not blocking_reasons

    promotion_ready: bool | None = None
    promotion_reasons: list[str] = []
    selected_checkpoint: str | None = None
    if promotion_gate is not None:
        promotion_ready = bool(promotion_gate.get("promotable", False))
        promotion_reasons = _string_list(promotion_gate.get("blocking_reasons"))
        selected_checkpoint = _optional_string(
            promotion_gate.get("selected_checkpoint_path")
        )

    return StrategyTrainingReadinessResult(
        recommendation="train" if training_ready else "hold",
        training_ready=training_ready,
        promotion_ready=promotion_ready,
        blocking_reasons=blocking_reasons,
        expected_inputs=expected_inputs,
        trajectory_detail_gate_path=readiness_config.trajectory_detail_gate_path,
        trajectory_detail_gate_ready=trajectory_ready,
        trajectory_detail_gate_inputs=trajectory_inputs,
        trajectory_detail_gate_blocking_reasons=trajectory_reasons,
        policy_explanation_gate_path=readiness_config.policy_explanation_gate_path,
        policy_explanation_gate_ready=explanation_ready,
        policy_explanation_gate_inputs=explanation_inputs,
        policy_explanation_gate_blocking_reasons=explanation_reasons,
        observation_detail_gate_path=str(observation_path),
        observation_detail_gate_ready=observation_ready,
        observation_detail_gate_inputs=observation_inputs,
        observation_detail_gate_blocking_reasons=observation_reasons,
        promotion_gate_path=readiness_config.promotion_gate_path,
        promotion_gate_promotable=promotion_ready,
        promotion_gate_blocking_reasons=promotion_reasons,
        selected_checkpoint_path=selected_checkpoint,
    )


def _training_blocking_reasons(
    *,
    trajectory_ready: bool | None,
    trajectory_inputs: list[str],
    explanation_ready: bool | None,
    explanation_inputs: list[str],
    observation_ready: bool,
    expected_inputs: list[str],
    observation_inputs: list[str],
) -> list[str]:
    reasons: list[str] = []
    if trajectory_ready is False:
        reasons.append("trajectory_detail_gate_not_ready")
    if (
        trajectory_ready is not None
        and expected_inputs
        and trajectory_inputs != expected_inputs
    ):
        reasons.append("trajectory_detail_gate_inputs_mismatch")
    if explanation_ready is False:
        reasons.append("policy_explanation_gate_not_ready")
    if (
        explanation_ready is not None
        and expected_inputs
        and explanation_inputs != expected_inputs
    ):
        reasons.append("policy_explanation_gate_inputs_mismatch")
    if not observation_ready:
        reasons.append("observation_detail_gate_not_ready")
    if expected_inputs and observation_inputs != expected_inputs:
        reasons.append("observation_detail_gate_inputs_mismatch")
    return reasons


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
