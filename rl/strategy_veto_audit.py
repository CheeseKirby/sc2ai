"""Offline audit for a conservative strategy veto baseline."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rl.strategy_datasets import StrategyTrajectoryPathInput
from rl.strategy_signal_dataset import StrategySignalRecord, build_strategy_signal_dataset


@dataclass(frozen=True)
class StrategyVetoRecordDecision:
    """One offline veto/review decision for a signal record."""

    path: str
    step: int
    game_time: float
    candidate_action: str
    recommended_training_use: str
    label_quality: str
    hard_veto: bool
    hard_veto_reasons: list[str]
    review_reasons: list[str]


@dataclass(frozen=True)
class StrategyVetoAudit:
    """Dataset-level summary for a rule-based veto baseline."""

    inputs: list[str]
    files: int
    records: int
    hard_veto_records: int
    review_records: int
    bad_records: int
    bad_records_hard_vetoed: int
    accept_positive_records: int
    accept_positive_records_hard_vetoed: int
    bad_capture_ratio: float
    accept_positive_false_veto_ratio: float
    hard_veto_by_reason: dict[str, int]
    review_by_reason: dict[str, int]
    hard_veto_by_training_use: dict[str, int]
    review_by_training_use: dict[str, int]
    hard_veto_by_action: dict[str, int]
    review_by_action: dict[str, int]
    decisions: list[StrategyVetoRecordDecision]


def audit_strategy_veto_baseline(
    paths: StrategyTrajectoryPathInput,
    *,
    include_before_filter_candidates: bool = False,
) -> StrategyVetoAudit:
    """Audit a conservative veto baseline over strategy signal records."""
    dataset = build_strategy_signal_dataset(
        paths,
        include_before_filter_candidates=include_before_filter_candidates,
    )
    decisions = [_decision(record) for record in dataset.records]
    hard_vetoed = [decision for decision in decisions if decision.hard_veto]
    review = [decision for decision in decisions if decision.review_reasons]
    bad_records = [
        record
        for record in dataset.records
        if record.label_quality == "bad"
    ]
    bad_hard_vetoed = [
        decision
        for decision in hard_vetoed
        if decision.label_quality == "bad"
    ]
    accept_positive = [
        record
        for record in dataset.records
        if record.recommended_training_use == "accept_positive"
    ]
    accept_positive_hard_vetoed = [
        decision
        for decision in hard_vetoed
        if decision.recommended_training_use == "accept_positive"
    ]

    return StrategyVetoAudit(
        inputs=dataset.inputs,
        files=dataset.files,
        records=len(dataset.records),
        hard_veto_records=len(hard_vetoed),
        review_records=len(review),
        bad_records=len(bad_records),
        bad_records_hard_vetoed=len(bad_hard_vetoed),
        accept_positive_records=len(accept_positive),
        accept_positive_records_hard_vetoed=len(accept_positive_hard_vetoed),
        bad_capture_ratio=_ratio(len(bad_hard_vetoed), len(bad_records)),
        accept_positive_false_veto_ratio=_ratio(
            len(accept_positive_hard_vetoed),
            len(accept_positive),
        ),
        hard_veto_by_reason=_count_reasons(
            decision.hard_veto_reasons for decision in hard_vetoed
        ),
        review_by_reason=_count_reasons(
            decision.review_reasons for decision in review
        ),
        hard_veto_by_training_use=_count(
            decision.recommended_training_use for decision in hard_vetoed
        ),
        review_by_training_use=_count(
            decision.recommended_training_use for decision in review
        ),
        hard_veto_by_action=_count(decision.candidate_action for decision in hard_vetoed),
        review_by_action=_count(decision.candidate_action for decision in review),
        decisions=decisions,
    )


def _decision(record: StrategySignalRecord) -> StrategyVetoRecordDecision:
    hard_reasons = _hard_veto_reasons(record)
    review_reasons = _review_reasons(record)
    return StrategyVetoRecordDecision(
        path=record.path,
        step=record.step,
        game_time=record.game_time,
        candidate_action=record.candidate_action,
        recommended_training_use=record.recommended_training_use,
        label_quality=record.label_quality,
        hard_veto=bool(hard_reasons),
        hard_veto_reasons=hard_reasons,
        review_reasons=review_reasons,
    )


def _hard_veto_reasons(record: StrategySignalRecord) -> list[str]:
    reasons: list[str] = []
    if not record.immediate_executable:
        reasons.append(f"not_executable:{record.candidate_blocker or 'unknown'}")
    return reasons


def _review_reasons(record: StrategySignalRecord) -> list[str]:
    reasons: list[str] = []
    if _static_defense_available_under_threat(record):
        reasons.append("static_defense_available_under_threat")
    return reasons


def _static_defense_available_under_threat(record: StrategySignalRecord) -> bool:
    metrics = record.start_metrics
    return (
        record.candidate_action == "PRODUCE_ARMY"
        and record.threat_state != "no_threat"
        and metrics.get("base_under_threat", 0.0) > 0.0
        and metrics.get("minerals", 0.0) >= 100.0
        and (
            metrics.get("has_cybernetics_core", 0.0) > 0.0
            or metrics.get("ready_forge", 0.0) > 0.0
        )
        and metrics.get("pending_static_defense", 0.0) <= 0.0
    )


def _count(values) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in Counter(values).items()))


def _count_reasons(reason_lists) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for reasons in reason_lists:
        counter.update(reasons)
    return dict(sorted((name, int(count)) for name, count in counter.items()))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
