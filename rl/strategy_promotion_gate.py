"""Promotion gates for offline strategy checkpoint signal audits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.experiments import read_json


@dataclass(frozen=True)
class StrategyPromotionGateConfig:
    """Fail-closed thresholds for promoting a strategy checkpoint."""

    min_rows: int = 1
    max_predicted_non_executable_rows: int = 0
    max_predicted_non_executable_ratio: float = 0.0
    max_veto_negative_matches: int = 0
    max_drop_non_executable_matches: int = 0
    max_action_space_exhausted_matches: int = 0
    max_action_critic_fallback_rows: int = 0
    fail_on_warnings: bool = True
    observation_detail_gate_path: str | None = None


@dataclass(frozen=True)
class StrategyPromotionCandidate:
    """One audited strategy checkpoint candidate and its gate decision."""

    audit_path: str
    inputs: list[str]
    checkpoint_path: str
    prediction_mode: str
    rows: int
    promotable: bool
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
    action_space_exhausted_prediction_matches: int
    action_space_exhausted_rows: int
    predicted_non_executable_rows: int
    predicted_non_executable_ratio: float
    action_critic_fallback_rows: int
    observation_detail_gate_path: str | None
    observation_detail_gate_ready: bool | None
    observation_detail_gate_inputs: list[str] | None
    observation_detail_gate_blocking_reasons: list[str]
    rank_score: float


@dataclass(frozen=True)
class StrategyPromotionGateResult:
    """Promotion recommendation across one or more audit candidates."""

    recommendation: str
    promotable: bool
    selected_audit_path: str | None
    selected_checkpoint_path: str | None
    blocking_reasons: list[str]
    candidates: list[StrategyPromotionCandidate]
    config: StrategyPromotionGateConfig


@dataclass(frozen=True)
class _ObservationDetailGateState:
    path: str
    ready: bool
    inputs: list[str] | None
    blocking_reasons: list[str]


def evaluate_strategy_promotion_gate(
    audit_paths: list[str | Path],
    *,
    config: StrategyPromotionGateConfig | None = None,
) -> StrategyPromotionGateResult:
    """Evaluate promotion readiness from one or more checkpoint audit JSON files."""
    gate_config = config or StrategyPromotionGateConfig()
    observation_detail_gate = _load_observation_detail_gate(
        gate_config.observation_detail_gate_path
    )
    candidates = [
        _candidate_from_audit_path(
            path,
            config=gate_config,
            observation_detail_gate=observation_detail_gate,
        )
        for path in audit_paths
    ]
    candidates = sorted(candidates, key=_candidate_sort_key)
    selected = candidates[0] if candidates else None
    promotable = bool(selected and selected.promotable)
    return StrategyPromotionGateResult(
        recommendation="promote" if promotable else "hold",
        promotable=promotable,
        selected_audit_path=selected.audit_path if selected else None,
        selected_checkpoint_path=selected.checkpoint_path if selected else None,
        blocking_reasons=[] if promotable else _result_blocking_reasons(candidates),
        candidates=candidates,
        config=gate_config,
    )


def _candidate_from_audit_path(
    path: str | Path,
    *,
    config: StrategyPromotionGateConfig,
    observation_detail_gate: _ObservationDetailGateState | None,
) -> StrategyPromotionCandidate:
    audit_path = Path(path)
    audit = read_json(audit_path)
    return candidate_from_audit(
        audit,
        audit_path=str(audit_path),
        config=config,
        observation_detail_gate=observation_detail_gate,
    )


def candidate_from_audit(
    audit: dict[str, Any],
    *,
    audit_path: str,
    config: StrategyPromotionGateConfig,
    observation_detail_gate: _ObservationDetailGateState | None = None,
) -> StrategyPromotionCandidate:
    """Return one gate decision from an already-loaded audit payload."""
    inputs = _string_list(audit.get("inputs"))
    rows = _int(audit, "rows")
    predicted_non_executable_rows = _int(audit, "predicted_non_executable_rows")
    predicted_non_executable_ratio = _float(
        audit,
        "predicted_non_executable_ratio",
    )
    veto_negative_matches = _int(audit, "veto_negative_prediction_matches")
    drop_non_executable_matches = _int(
        audit,
        "drop_non_executable_prediction_matches",
    )
    action_space_exhausted_matches = _int(
        audit,
        "action_space_exhausted_prediction_matches",
    )
    fallback_rows = _int(audit, "action_critic_fallback_rows")
    warnings = list(audit.get("warnings") or [])
    observation_detail_gate_reasons = _observation_detail_gate_blocking_reasons(
        inputs=inputs,
        observation_detail_gate=observation_detail_gate,
    )
    blocking_reasons = _candidate_blocking_reasons(
        rows=rows,
        predicted_non_executable_rows=predicted_non_executable_rows,
        predicted_non_executable_ratio=predicted_non_executable_ratio,
        veto_negative_matches=veto_negative_matches,
        drop_non_executable_matches=drop_non_executable_matches,
        action_space_exhausted_matches=action_space_exhausted_matches,
        action_critic_fallback_rows=fallback_rows,
        warnings=warnings,
        observation_detail_gate_reasons=observation_detail_gate_reasons,
        config=config,
    )
    rank_score = _rank_score(
        predicted_non_executable_rows=predicted_non_executable_rows,
        veto_negative_matches=veto_negative_matches,
        drop_non_executable_matches=drop_non_executable_matches,
        action_space_exhausted_matches=action_space_exhausted_matches,
        action_critic_fallback_rows=fallback_rows,
        prediction_match_ratio=_float(audit, "prediction_match_ratio"),
        accept_positive_matches=_int(audit, "accept_positive_prediction_matches"),
    )
    return StrategyPromotionCandidate(
        audit_path=audit_path,
        inputs=inputs,
        checkpoint_path=str(audit.get("checkpoint_path", "")),
        prediction_mode=str(audit.get("prediction_mode", "")),
        rows=rows,
        promotable=not blocking_reasons,
        blocking_reasons=blocking_reasons,
        warnings=warnings,
        prediction_matches_recorded=_int(audit, "prediction_matches_recorded"),
        prediction_match_ratio=_float(audit, "prediction_match_ratio"),
        accept_positive_prediction_matches=_int(
            audit,
            "accept_positive_prediction_matches",
        ),
        accept_positive_rows=_int(audit, "accept_positive_rows"),
        veto_negative_prediction_matches=veto_negative_matches,
        veto_negative_rows=_int(audit, "veto_negative_rows"),
        drop_non_executable_prediction_matches=drop_non_executable_matches,
        drop_non_executable_rows=_int(audit, "drop_non_executable_rows"),
        action_space_exhausted_prediction_matches=action_space_exhausted_matches,
        action_space_exhausted_rows=_int(audit, "action_space_exhausted_rows"),
        predicted_non_executable_rows=predicted_non_executable_rows,
        predicted_non_executable_ratio=predicted_non_executable_ratio,
        action_critic_fallback_rows=fallback_rows,
        observation_detail_gate_path=(
            observation_detail_gate.path
            if observation_detail_gate is not None
            else None
        ),
        observation_detail_gate_ready=(
            observation_detail_gate.ready
            if observation_detail_gate is not None
            else None
        ),
        observation_detail_gate_inputs=(
            observation_detail_gate.inputs
            if observation_detail_gate is not None
            else None
        ),
        observation_detail_gate_blocking_reasons=(
            observation_detail_gate.blocking_reasons
            if observation_detail_gate is not None
            else []
        ),
        rank_score=rank_score,
    )


def _candidate_blocking_reasons(
    *,
    rows: int,
    predicted_non_executable_rows: int,
    predicted_non_executable_ratio: float,
    veto_negative_matches: int,
    drop_non_executable_matches: int,
    action_space_exhausted_matches: int,
    action_critic_fallback_rows: int,
    warnings: list[str],
    observation_detail_gate_reasons: list[str],
    config: StrategyPromotionGateConfig,
) -> list[str]:
    reasons: list[str] = []
    if rows < config.min_rows:
        reasons.append("insufficient_rows")
    if predicted_non_executable_rows > config.max_predicted_non_executable_rows:
        reasons.append("predicted_non_executable_rows_high")
    if predicted_non_executable_ratio > config.max_predicted_non_executable_ratio:
        reasons.append("predicted_non_executable_ratio_high")
    if veto_negative_matches > config.max_veto_negative_matches:
        reasons.append("predicted_matches_veto_negative_labels")
    if drop_non_executable_matches > config.max_drop_non_executable_matches:
        reasons.append("predicted_matches_drop_non_executable_labels")
    if action_space_exhausted_matches > config.max_action_space_exhausted_matches:
        reasons.append("predicted_matches_action_space_exhausted_labels")
    if action_critic_fallback_rows > config.max_action_critic_fallback_rows:
        reasons.append("action_critic_fallback_rows_high")
    if config.fail_on_warnings and warnings:
        reasons.append("audit_warnings_present")
    reasons.extend(observation_detail_gate_reasons)
    return reasons


def _load_observation_detail_gate(
    gate_path: str | None,
) -> _ObservationDetailGateState | None:
    if gate_path is None:
        return None
    gate = read_json(Path(gate_path))
    inputs = gate.get("analysis_inputs")
    return _ObservationDetailGateState(
        path=str(gate_path),
        ready=bool(gate.get("ready", False)),
        inputs=_string_list(inputs) if isinstance(inputs, list) else None,
        blocking_reasons=_string_list(gate.get("blocking_reasons")),
    )


def _observation_detail_gate_blocking_reasons(
    *,
    inputs: list[str],
    observation_detail_gate: _ObservationDetailGateState | None,
) -> list[str]:
    if observation_detail_gate is None:
        return []
    if not observation_detail_gate.ready:
        return ["observation_detail_gate_not_ready"]
    if observation_detail_gate.inputs is None:
        return ["observation_detail_gate_inputs_missing"]
    if not inputs:
        return ["audit_inputs_missing"]
    if observation_detail_gate.inputs != inputs:
        return ["observation_detail_gate_inputs_mismatch"]
    return []


def _candidate_sort_key(candidate: StrategyPromotionCandidate) -> tuple:
    return (
        0 if candidate.promotable else 1,
        len(candidate.blocking_reasons),
        candidate.rank_score,
        -candidate.prediction_match_ratio,
        -candidate.accept_positive_prediction_matches,
        candidate.audit_path,
    )


def _rank_score(
    *,
    predicted_non_executable_rows: int,
    veto_negative_matches: int,
    drop_non_executable_matches: int,
    action_space_exhausted_matches: int,
    action_critic_fallback_rows: int,
    prediction_match_ratio: float,
    accept_positive_matches: int,
) -> float:
    return (
        (predicted_non_executable_rows * 100000.0)
        + (veto_negative_matches * 10000.0)
        + (action_space_exhausted_matches * 8000.0)
        + (drop_non_executable_matches * 5000.0)
        + (action_critic_fallback_rows * 10.0)
        - (prediction_match_ratio * 10.0)
        - (accept_positive_matches * 0.01)
    )


def _result_blocking_reasons(
    candidates: list[StrategyPromotionCandidate],
) -> list[str]:
    if not candidates:
        return ["no_candidates"]
    reasons = {
        reason
        for candidate in candidates
        for reason in candidate.blocking_reasons
    }
    return sorted(reasons)


def _int(payload: dict[str, Any], key: str) -> int:
    return int(payload.get(key, 0) or 0)


def _float(payload: dict[str, Any], key: str) -> float:
    return float(payload.get(key, 0.0) or 0.0)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
