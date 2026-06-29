"""Promotion-gate audit for baseline and candidate strategy trajectories."""
from __future__ import annotations

from dataclasses import dataclass

from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_datasets import StrategyTrajectoryPathInput
from rl.strategy_outcome_diagnostics import (
    DEFAULT_LOOKAHEAD_SECONDS,
    StrategyOutcomeDiagnostics,
    diagnose_strategy_outcomes,
)


PROMOTION_REQUIRED_ACTIONS: tuple[str, ...] = (
    "ADD_GATEWAYS",
    "TECH_ROBO",
    "BUILD_STATIC_DEFENSE",
)


@dataclass(frozen=True)
class StrategyCandidateSideSummary:
    """Compact side-by-side summary used by the candidate promotion gate."""

    inputs: list[str]
    files: int
    rows: int
    training_rows: int
    result_counts: dict[str, int]
    result_score: float
    action_counts_by_name: dict[str, int]
    action_first_game_time_by_name: dict[str, float]
    filter_change_rows: int
    base_threat_rows: int
    execution_effect_counts: dict[str, int]
    execution_blocker_counts: dict[str, int]
    execution_blocker_total: int


@dataclass(frozen=True)
class StrategyCandidateAudit:
    """Comparison result for deciding whether a candidate is promotable."""

    baseline: StrategyCandidateSideSummary
    candidate: StrategyCandidateSideSummary
    promotable: bool
    blocking_reasons: list[str]
    warnings: list[str]
    result_score_delta: float
    base_threat_rows_delta: int
    action_count_delta_by_name: dict[str, int]
    execution_blocker_delta: int


def audit_strategy_candidate(
    baseline_paths: StrategyTrajectoryPathInput,
    candidate_paths: StrategyTrajectoryPathInput,
) -> StrategyCandidateAudit:
    """Audit a candidate strategy trajectory set against a frozen baseline."""
    baseline_diagnostics = diagnose_strategy_outcomes(
        baseline_paths,
        lookahead_seconds=DEFAULT_LOOKAHEAD_SECONDS,
    )
    candidate_diagnostics = diagnose_strategy_outcomes(
        candidate_paths,
        lookahead_seconds=DEFAULT_LOOKAHEAD_SECONDS,
    )
    baseline = _side_summary(baseline_diagnostics)
    candidate = _side_summary(candidate_diagnostics)

    result_score_delta = candidate.result_score - baseline.result_score
    base_threat_rows_delta = candidate.base_threat_rows - baseline.base_threat_rows
    action_count_delta_by_name = _action_count_delta(
        baseline.action_counts_by_name,
        candidate.action_counts_by_name,
    )
    execution_blocker_delta = (
        candidate.execution_blocker_total - baseline.execution_blocker_total
    )

    blocking_reasons = _blocking_reasons(
        result_score_delta=result_score_delta,
        base_threat_rows_delta=base_threat_rows_delta,
        action_count_delta_by_name=action_count_delta_by_name,
        execution_blocker_delta=execution_blocker_delta,
    )
    warnings = _warnings(candidate)

    return StrategyCandidateAudit(
        baseline=baseline,
        candidate=candidate,
        promotable=not blocking_reasons,
        blocking_reasons=blocking_reasons,
        warnings=warnings,
        result_score_delta=result_score_delta,
        base_threat_rows_delta=base_threat_rows_delta,
        action_count_delta_by_name=action_count_delta_by_name,
        execution_blocker_delta=execution_blocker_delta,
    )


def _side_summary(
    diagnostics: StrategyOutcomeDiagnostics,
) -> StrategyCandidateSideSummary:
    action_counts_by_name = {
        action_name: diagnostics.action_summaries_by_name.get(action_name).count
        if action_name in diagnostics.action_summaries_by_name
        else 0
        for action_name in STRATEGY_ACTION_NAMES.values()
    }
    action_first_game_time_by_name = {
        name: summary.first_game_time
        for name, summary in diagnostics.action_summaries_by_name.items()
        if summary.first_game_time is not None
    }
    filter_change_rows = sum(
        summary.filter_change_rows for summary in diagnostics.source_summaries
    )
    base_threat_rows = sum(
        summary.base_threat_rows for summary in diagnostics.file_summaries
    )
    execution_blocker_total = sum(diagnostics.execution_blocker_counts.values())

    return StrategyCandidateSideSummary(
        inputs=diagnostics.inputs,
        files=diagnostics.files,
        rows=diagnostics.rows,
        training_rows=diagnostics.training_rows,
        result_counts=diagnostics.result_counts,
        result_score=_result_score(diagnostics.result_counts),
        action_counts_by_name=action_counts_by_name,
        action_first_game_time_by_name=action_first_game_time_by_name,
        filter_change_rows=filter_change_rows,
        base_threat_rows=base_threat_rows,
        execution_effect_counts=diagnostics.execution_effect_counts,
        execution_blocker_counts=diagnostics.execution_blocker_counts,
        execution_blocker_total=execution_blocker_total,
    )


def _result_score(result_counts: dict[str, int]) -> float:
    games = sum(result_counts.values())
    if games <= 0:
        return 0.0
    victories = sum(
        count for result, count in result_counts.items() if _is_victory(result)
    )
    return victories / games


def _is_victory(result: str) -> bool:
    return result == "Victory" or result.endswith(".Victory")


def _action_count_delta(
    baseline_counts: dict[str, int],
    candidate_counts: dict[str, int],
) -> dict[str, int]:
    return {
        action_name: int(candidate_counts.get(action_name, 0))
        - int(baseline_counts.get(action_name, 0))
        for action_name in STRATEGY_ACTION_NAMES.values()
    }


def _blocking_reasons(
    *,
    result_score_delta: float,
    base_threat_rows_delta: int,
    action_count_delta_by_name: dict[str, int],
    execution_blocker_delta: int,
) -> list[str]:
    reasons: list[str] = []
    if result_score_delta < 0.0:
        reasons.append("candidate_result_worse_than_baseline")
    if base_threat_rows_delta > 0:
        reasons.append("base_threat_rows_regressed")
    for action_name in PROMOTION_REQUIRED_ACTIONS:
        if action_count_delta_by_name.get(action_name, 0) < 0:
            reasons.append(f"{action_name.lower()}_count_regressed")
    if execution_blocker_delta > 0:
        reasons.append("execution_blockers_increased")
    return reasons


def _warnings(candidate: StrategyCandidateSideSummary) -> list[str]:
    warnings: list[str] = []
    if candidate.filter_change_rows > 0:
        warnings.append("candidate_has_filter_changes")
    return warnings
