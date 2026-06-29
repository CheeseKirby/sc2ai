"""Signal-quality audit for baseline and candidate strategy trajectories."""
from __future__ import annotations

from dataclasses import dataclass

from rl.strategy_datasets import StrategyTrajectoryPathInput
from rl.strategy_signal_dataset import StrategySignalDataset, build_strategy_signal_dataset


@dataclass(frozen=True)
class StrategySignalSideSummary:
    """Compact signal-quality summary for one trajectory set."""

    inputs: list[str]
    files: int
    rows: int
    training_rows: int
    records: int
    records_by_training_use: dict[str, int]
    records_by_label_quality: dict[str, int]
    records_by_candidate_action: dict[str, int]
    records_by_candidate_source: dict[str, int]
    accept_positive_ratio: float
    bad_signal_ratio: float
    drop_non_executable_ratio: float
    veto_negative_ratio: float
    weak_context_ratio: float
    needs_fresh_ab_count: int


@dataclass(frozen=True)
class StrategySignalAudit:
    """Comparison result for deciding whether signal quality regressed."""

    baseline: StrategySignalSideSummary
    candidate: StrategySignalSideSummary
    signal_healthy: bool
    blocking_reasons: list[str]
    warnings: list[str]
    accept_positive_ratio_delta: float
    bad_signal_ratio_delta: float
    drop_non_executable_ratio_delta: float
    veto_negative_ratio_delta: float
    weak_context_ratio_delta: float


def audit_strategy_signals(
    baseline_paths: StrategyTrajectoryPathInput,
    candidate_paths: StrategyTrajectoryPathInput,
    *,
    include_before_filter_candidates: bool = False,
) -> StrategySignalAudit:
    """Audit candidate row-level signal quality against a baseline."""
    baseline_dataset = build_strategy_signal_dataset(
        baseline_paths,
        include_before_filter_candidates=include_before_filter_candidates,
    )
    candidate_dataset = build_strategy_signal_dataset(
        candidate_paths,
        include_before_filter_candidates=include_before_filter_candidates,
    )
    baseline = _side_summary(baseline_dataset)
    candidate = _side_summary(candidate_dataset)

    accept_positive_ratio_delta = (
        candidate.accept_positive_ratio - baseline.accept_positive_ratio
    )
    bad_signal_ratio_delta = candidate.bad_signal_ratio - baseline.bad_signal_ratio
    drop_non_executable_ratio_delta = (
        candidate.drop_non_executable_ratio - baseline.drop_non_executable_ratio
    )
    veto_negative_ratio_delta = (
        candidate.veto_negative_ratio - baseline.veto_negative_ratio
    )
    weak_context_ratio_delta = candidate.weak_context_ratio - baseline.weak_context_ratio
    blocking_reasons = _blocking_reasons(
        accept_positive_ratio_delta=accept_positive_ratio_delta,
        bad_signal_ratio_delta=bad_signal_ratio_delta,
        drop_non_executable_ratio_delta=drop_non_executable_ratio_delta,
        veto_negative_ratio_delta=veto_negative_ratio_delta,
    )
    warnings = _warnings(candidate)

    return StrategySignalAudit(
        baseline=baseline,
        candidate=candidate,
        signal_healthy=not blocking_reasons,
        blocking_reasons=blocking_reasons,
        warnings=warnings,
        accept_positive_ratio_delta=accept_positive_ratio_delta,
        bad_signal_ratio_delta=bad_signal_ratio_delta,
        drop_non_executable_ratio_delta=drop_non_executable_ratio_delta,
        veto_negative_ratio_delta=veto_negative_ratio_delta,
        weak_context_ratio_delta=weak_context_ratio_delta,
    )


def _side_summary(dataset: StrategySignalDataset) -> StrategySignalSideSummary:
    records = len(dataset.records)
    return StrategySignalSideSummary(
        inputs=dataset.inputs,
        files=dataset.files,
        rows=dataset.rows,
        training_rows=dataset.training_rows,
        records=records,
        records_by_training_use=dataset.records_by_training_use,
        records_by_label_quality=dataset.records_by_label_quality,
        records_by_candidate_action=dataset.records_by_candidate_action,
        records_by_candidate_source=dataset.records_by_candidate_source,
        accept_positive_ratio=_ratio(
            dataset.records_by_training_use.get("accept_positive", 0),
            records,
        ),
        bad_signal_ratio=_ratio(
            dataset.records_by_label_quality.get("bad", 0),
            records,
        ),
        drop_non_executable_ratio=_ratio(
            dataset.records_by_training_use.get("drop_non_executable", 0),
            records,
        ),
        veto_negative_ratio=_ratio(
            dataset.records_by_training_use.get("veto_negative", 0),
            records,
        ),
        weak_context_ratio=_ratio(
            dataset.records_by_training_use.get("weak_context", 0),
            records,
        ),
        needs_fresh_ab_count=dataset.records_by_training_use.get("needs_fresh_ab", 0),
    )


def _blocking_reasons(
    *,
    accept_positive_ratio_delta: float,
    bad_signal_ratio_delta: float,
    drop_non_executable_ratio_delta: float,
    veto_negative_ratio_delta: float,
) -> list[str]:
    reasons: list[str] = []
    if accept_positive_ratio_delta < 0.0:
        reasons.append("accept_positive_ratio_regressed")
    if bad_signal_ratio_delta > 0.0:
        reasons.append("bad_signal_ratio_regressed")
    if drop_non_executable_ratio_delta > 0.0:
        reasons.append("non_executable_ratio_regressed")
    if veto_negative_ratio_delta > 0.0:
        reasons.append("veto_negative_ratio_regressed")
    return reasons


def _warnings(candidate: StrategySignalSideSummary) -> list[str]:
    warnings: list[str] = []
    if candidate.needs_fresh_ab_count > 0:
        warnings.append("candidate_has_counterfactual_rows_needing_fresh_ab")
    return warnings


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
