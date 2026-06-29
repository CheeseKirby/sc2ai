"""Threshold sweep utilities for action-critic masked strategy audits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from rl.strategy_checkpoint_signal_audit import (
    ACTION_CRITIC_FALLBACK_POLICIES,
    StrategyCheckpointSignalAudit,
    audit_strategy_checkpoint_signals,
)
from rl.strategy_datasets import StrategyTrajectoryPathInput
from rl.strategy_signal_critic import (
    DEFAULT_CRITIC_MAX_BAD_RATE,
    DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    DEFAULT_CRITIC_MIN_SAMPLES,
)


DEFAULT_ACTION_CRITIC_SWEEP_THRESHOLDS: tuple[float, ...] = (
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
)


@dataclass(frozen=True)
class StrategyActionCriticThresholdTrial:
    """One action critic threshold/fallback trial summarized for comparison."""

    threshold: float
    fallback_policy: str
    rows: int
    signal_healthy: bool
    blocking_reasons: list[str]
    warnings: list[str]
    prediction_matches_recorded: int
    prediction_match_ratio: float
    accept_positive_prediction_matches: int
    accept_positive_rows: int
    veto_negative_prediction_matches: int
    veto_negative_rows: int
    drop_non_executable_prediction_matches: int
    drop_non_executable_rows: int
    predicted_non_executable_rows: int
    predicted_non_executable_ratio: float
    action_critic_fallback_rows: int
    action_critic_selected_unsafe_probability_avg: float | None
    action_critic_selected_unsafe_probability_max: float | None
    rank_score: float


@dataclass(frozen=True)
class StrategyActionCriticThresholdSweep:
    """Sweep summary for selecting safer action critic thresholds."""

    inputs: list[str]
    checkpoint_path: str
    action_critic_checkpoint_path: str
    thresholds: list[float]
    fallback_policies: list[str]
    trials: list[StrategyActionCriticThresholdTrial]
    selected_trial: StrategyActionCriticThresholdTrial | None
    recommendation: str
    blocking_reasons: list[str]


def sweep_strategy_action_critic_thresholds(
    paths: StrategyTrajectoryPathInput,
    checkpoint_path: str | Path,
    action_critic_checkpoint_path: str | Path,
    *,
    thresholds: Iterable[float] = DEFAULT_ACTION_CRITIC_SWEEP_THRESHOLDS,
    fallback_policies: Iterable[str] = ("lowest-risk",),
    device: str | torch.device = "cpu",
    critic_min_samples: int = DEFAULT_CRITIC_MIN_SAMPLES,
    critic_max_bad_rate: float = DEFAULT_CRITIC_MAX_BAD_RATE,
    critic_max_veto_negative_rate: float = DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
) -> StrategyActionCriticThresholdSweep:
    """Run action-critic masked audits across thresholds and fallback policies."""
    threshold_values = _validate_thresholds(thresholds)
    fallback_policy_values = _validate_fallback_policies(fallback_policies)

    trials: list[StrategyActionCriticThresholdTrial] = []
    inputs: list[str] = []
    for fallback_policy in fallback_policy_values:
        for threshold in threshold_values:
            audit = audit_strategy_checkpoint_signals(
                paths,
                checkpoint_path,
                device=device,
                prediction_mode="action-critic-mask",
                critic_min_samples=critic_min_samples,
                critic_max_bad_rate=critic_max_bad_rate,
                critic_max_veto_negative_rate=critic_max_veto_negative_rate,
                action_critic_checkpoint_path=action_critic_checkpoint_path,
                action_critic_threshold=threshold,
                action_critic_fallback_policy=fallback_policy,
            )
            if not inputs:
                inputs = list(audit.inputs)
            trials.append(
                action_critic_threshold_trial_from_audit(
                    audit,
                    threshold=threshold,
                    fallback_policy=fallback_policy,
                )
            )

    sorted_trials = sorted(trials, key=_trial_sort_key)
    selected = sorted_trials[0] if sorted_trials else None
    return StrategyActionCriticThresholdSweep(
        inputs=inputs,
        checkpoint_path=str(checkpoint_path),
        action_critic_checkpoint_path=str(action_critic_checkpoint_path),
        thresholds=list(threshold_values),
        fallback_policies=list(fallback_policy_values),
        trials=sorted_trials,
        selected_trial=selected,
        recommendation=(
            "promotion_candidate" if selected and selected.signal_healthy else "hold"
        ),
        blocking_reasons=[] if selected and selected.signal_healthy else (
            list(selected.blocking_reasons) if selected else ["no_trials"]
        ),
    )


def action_critic_threshold_trial_from_audit(
    audit: StrategyCheckpointSignalAudit | dict[str, Any],
    *,
    threshold: float,
    fallback_policy: str,
) -> StrategyActionCriticThresholdTrial:
    """Build one comparable threshold trial from an audit object or payload."""
    veto_matches = _int(audit, "veto_negative_prediction_matches")
    drop_matches = _int(audit, "drop_non_executable_prediction_matches")
    predicted_non_executable_rows = _int(audit, "predicted_non_executable_rows")
    fallback_rows = _int(audit, "action_critic_fallback_rows")
    prediction_match_ratio = _float(audit, "prediction_match_ratio")
    accept_matches = _int(audit, "accept_positive_prediction_matches")
    rank_score = _rank_score(
        veto_negative_matches=veto_matches,
        drop_non_executable_matches=drop_matches,
        predicted_non_executable_rows=predicted_non_executable_rows,
        action_critic_fallback_rows=fallback_rows,
        prediction_match_ratio=prediction_match_ratio,
        accept_positive_matches=accept_matches,
        warnings=len(_list(audit, "warnings")),
    )
    return StrategyActionCriticThresholdTrial(
        threshold=float(threshold),
        fallback_policy=fallback_policy,
        rows=_int(audit, "rows"),
        signal_healthy=bool(_get(audit, "signal_healthy", False)),
        blocking_reasons=_list(audit, "blocking_reasons"),
        warnings=_list(audit, "warnings"),
        prediction_matches_recorded=_int(audit, "prediction_matches_recorded"),
        prediction_match_ratio=prediction_match_ratio,
        accept_positive_prediction_matches=accept_matches,
        accept_positive_rows=_int(audit, "accept_positive_rows"),
        veto_negative_prediction_matches=veto_matches,
        veto_negative_rows=_int(audit, "veto_negative_rows"),
        drop_non_executable_prediction_matches=drop_matches,
        drop_non_executable_rows=_int(audit, "drop_non_executable_rows"),
        predicted_non_executable_rows=predicted_non_executable_rows,
        predicted_non_executable_ratio=_float(audit, "predicted_non_executable_ratio"),
        action_critic_fallback_rows=fallback_rows,
        action_critic_selected_unsafe_probability_avg=_optional_float(
            audit,
            "action_critic_selected_unsafe_probability_avg",
        ),
        action_critic_selected_unsafe_probability_max=_optional_float(
            audit,
            "action_critic_selected_unsafe_probability_max",
        ),
        rank_score=rank_score,
    )


def _validate_thresholds(thresholds: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in thresholds)
    if not values:
        raise ValueError("at least one threshold is required")
    invalid = [value for value in values if not 0.0 <= value <= 1.0]
    if invalid:
        raise ValueError(f"thresholds must be in [0.0, 1.0]: {invalid}")
    return values


def _validate_fallback_policies(policies: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in policies)
    if not values:
        raise ValueError("at least one fallback policy is required")
    invalid = [value for value in values if value not in ACTION_CRITIC_FALLBACK_POLICIES]
    if invalid:
        names = ", ".join(ACTION_CRITIC_FALLBACK_POLICIES)
        raise ValueError(f"unknown fallback policies {invalid}; expected {names}")
    return values


def _trial_sort_key(trial: StrategyActionCriticThresholdTrial) -> tuple:
    return (
        0 if trial.signal_healthy else 1,
        trial.rank_score,
        -trial.prediction_match_ratio,
        -trial.accept_positive_prediction_matches,
        trial.threshold,
        trial.fallback_policy,
    )


def _rank_score(
    *,
    veto_negative_matches: int,
    drop_non_executable_matches: int,
    predicted_non_executable_rows: int,
    action_critic_fallback_rows: int,
    prediction_match_ratio: float,
    accept_positive_matches: int,
    warnings: int,
) -> float:
    return (
        (warnings * 1_000_000.0)
        + (predicted_non_executable_rows * 100_000.0)
        + (veto_negative_matches * 10_000.0)
        + (drop_non_executable_matches * 5_000.0)
        + (action_critic_fallback_rows * 10.0)
        - (prediction_match_ratio * 10.0)
        - (accept_positive_matches * 0.01)
    )


def _get(payload: StrategyCheckpointSignalAudit | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _int(payload: StrategyCheckpointSignalAudit | dict[str, Any], key: str) -> int:
    return int(_get(payload, key, 0) or 0)


def _float(payload: StrategyCheckpointSignalAudit | dict[str, Any], key: str) -> float:
    return float(_get(payload, key, 0.0) or 0.0)


def _optional_float(
    payload: StrategyCheckpointSignalAudit | dict[str, Any],
    key: str,
) -> float | None:
    value = _get(payload, key, None)
    if value is None:
        return None
    return float(value)


def _list(payload: StrategyCheckpointSignalAudit | dict[str, Any], key: str) -> list[str]:
    return [str(value) for value in (_get(payload, key, []) or [])]
