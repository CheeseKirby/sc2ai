"""Cluster veto-negative strategy signals for offline error analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.experiments import read_json
from rl.strategy_datasets import StrategyTrajectoryPathInput
from rl.strategy_signal_dataset import (
    StrategySignalRecord,
    build_strategy_signal_dataset,
)


VETO_START_METRICS: tuple[str, ...] = (
    "army_count",
    "workers",
    "ready_gateways",
    "gateway_idle_count",
    "ready_static_defense",
    "pending_static_defense",
    "base_under_air_threat",
    "base_under_ground_threat",
    "minerals",
    "vespene",
    "supply_left",
)


@dataclass(frozen=True)
class StrategyVetoNegativeExample:
    """One representative veto-negative signal row."""

    source_path: str
    source: str
    step: int
    game_time: float
    recorded_action: str
    candidate_action: str
    context: str
    threat_state: str
    reasons: list[str]
    start_metrics: dict[str, float]
    last_window: str
    last_window_metrics: dict[str, float]
    last_window_negative_events: list[str]
    payoff_events_by_window: dict[str, list[str]]
    matched_audits: list[str]
    matched_predicted_actions: list[str]
    audit_fallback_selected: bool


@dataclass(frozen=True)
class StrategyVetoNegativeAnalysis:
    """Aggregated clusters for veto-negative strategy signal rows."""

    inputs: list[str]
    audit_paths: list[str]
    files: int
    trajectory_rows: int
    training_rows: int
    signal_records: int
    veto_negative_records: int
    audit_decisions: int
    matched_by_audit_decisions: int
    matched_by_any_audit_records: int
    by_action: dict[str, int]
    by_threat_state: dict[str, int]
    by_context: dict[str, int]
    by_reason: dict[str, int]
    by_file: dict[str, int]
    by_source: dict[str, int]
    by_action_threat: dict[str, int]
    by_action_context_threat: dict[str, int]
    start_metric_averages: dict[str, float]
    start_metric_buckets: dict[str, dict[str, int]]
    negative_events_by_window: dict[str, dict[str, int]]
    payoff_events_by_window: dict[str, dict[str, int]]
    last_window_metric_averages: dict[str, float]
    last_window_metric_buckets: dict[str, dict[str, int]]
    matched_by_audit_action: dict[str, int]
    matched_by_audit_threat_state: dict[str, int]
    matched_by_audit_context: dict[str, int]
    matched_by_audit_path: dict[str, int]
    matched_by_audit_fallback_selected: dict[str, int]
    examples: list[StrategyVetoNegativeExample]


@dataclass(frozen=True)
class _AuditMatch:
    audit_path: str
    predicted_action: str
    threat_state: str
    context: str
    fallback_selected: bool


def analyze_strategy_veto_negatives(
    paths: StrategyTrajectoryPathInput,
    *,
    audit_paths: Iterable[str | Path] | None = None,
    max_examples: int = 12,
    include_before_filter_candidates: bool = False,
) -> StrategyVetoNegativeAnalysis:
    """Return veto-negative clusters, optionally joined with audit matches."""
    if max_examples < 0:
        raise ValueError("max_examples must be >= 0")

    dataset = build_strategy_signal_dataset(
        paths,
        include_before_filter_candidates=include_before_filter_candidates,
    )
    records = [
        record
        for record in dataset.records
        if record.recommended_training_use == "veto_negative"
    ]
    audit_lookup, audit_path_strings, audit_decisions = _load_audit_matches(
        audit_paths or []
    )
    matched_records = [
        record for record in records if _record_key(record) in audit_lookup
    ]
    matched_decisions = [
        match
        for record in records
        for match in audit_lookup.get(_record_key(record), [])
    ]

    return StrategyVetoNegativeAnalysis(
        inputs=dataset.inputs,
        audit_paths=audit_path_strings,
        files=dataset.files,
        trajectory_rows=dataset.rows,
        training_rows=dataset.training_rows,
        signal_records=len(dataset.records),
        veto_negative_records=len(records),
        audit_decisions=audit_decisions,
        matched_by_audit_decisions=len(matched_decisions),
        matched_by_any_audit_records=len(matched_records),
        by_action=_count(record.candidate_action for record in records),
        by_threat_state=_count(record.threat_state for record in records),
        by_context=_count(record.context for record in records),
        by_reason=_count(reason for record in records for reason in record.reasons),
        by_file=_count(Path(record.path).name for record in records),
        by_source=_count(record.source for record in records),
        by_action_threat=_count(
            f"{record.candidate_action}|{record.threat_state}"
            for record in records
        ),
        by_action_context_threat=_count(
            f"{record.candidate_action}|{record.context}|{record.threat_state}"
            for record in records
        ),
        start_metric_averages=_metric_averages(records, VETO_START_METRICS),
        start_metric_buckets=_start_metric_buckets(records),
        negative_events_by_window=_event_counts(
            records,
            "negative_events_by_window",
        ),
        payoff_events_by_window=_event_counts(records, "payoff_events_by_window"),
        last_window_metric_averages=_last_window_metric_averages(records),
        last_window_metric_buckets=_last_window_metric_buckets(records),
        matched_by_audit_action=_count(
            match.predicted_action for match in matched_decisions
        ),
        matched_by_audit_threat_state=_count(
            match.threat_state for match in matched_decisions
        ),
        matched_by_audit_context=_count(match.context for match in matched_decisions),
        matched_by_audit_path=_count(match.audit_path for match in matched_decisions),
        matched_by_audit_fallback_selected=_count(
            "true" if match.fallback_selected else "false"
            for match in matched_decisions
        ),
        examples=_examples(
            records,
            audit_lookup=audit_lookup,
            max_examples=max_examples,
        ),
    )


def _load_audit_matches(
    audit_paths: Iterable[str | Path],
) -> tuple[dict[tuple[str, int, str], list[_AuditMatch]], list[str], int]:
    lookup: dict[tuple[str, int, str], list[_AuditMatch]] = defaultdict(list)
    audit_path_strings: list[str] = []
    decision_count = 0
    for audit_path in audit_paths:
        audit_path_string = str(audit_path)
        audit_path_strings.append(audit_path_string)
        audit = read_json(audit_path)
        for decision in audit.get("decisions") or []:
            decision_count += 1
            if not bool(decision.get("prediction_matches_veto_negative", False)):
                continue
            key = _decision_key(decision)
            lookup[key].append(
                _AuditMatch(
                    audit_path=audit_path_string,
                    predicted_action=str(decision.get("predicted_action", "")),
                    threat_state=str(decision.get("threat_state", "")),
                    context=str(decision.get("context", "")),
                    fallback_selected=bool(
                        decision.get("action_critic_fallback_selected", False)
                    ),
                )
            )
    return dict(lookup), audit_path_strings, decision_count


def _examples(
    records: list[StrategySignalRecord],
    *,
    audit_lookup: dict[tuple[str, int, str], list[_AuditMatch]],
    max_examples: int,
) -> list[StrategyVetoNegativeExample]:
    ranked = sorted(
        records,
        key=lambda record: (
            -len(audit_lookup.get(_record_key(record), [])),
            record.candidate_action,
            record.threat_state,
            str(Path(record.path).name),
            record.step,
        ),
    )
    examples: list[StrategyVetoNegativeExample] = []
    for record in ranked[:max_examples]:
        matches = audit_lookup.get(_record_key(record), [])
        last_window = _last_window(record.metrics_by_window)
        examples.append(
            StrategyVetoNegativeExample(
                source_path=record.path,
                source=record.source,
                step=record.step,
                game_time=record.game_time,
                recorded_action=record.recorded_action,
                candidate_action=record.candidate_action,
                context=record.context,
                threat_state=record.threat_state,
                reasons=list(record.reasons),
                start_metrics={
                    metric: float(record.start_metrics.get(metric, 0.0))
                    for metric in VETO_START_METRICS
                },
                last_window=last_window,
                last_window_metrics={
                    name: float(value)
                    for name, value in record.metrics_by_window.get(
                        last_window,
                        {},
                    ).items()
                },
                last_window_negative_events=list(
                    record.negative_events_by_window.get(last_window, [])
                ),
                payoff_events_by_window={
                    window: list(events)
                    for window, events in record.payoff_events_by_window.items()
                    if events
                },
                matched_audits=[match.audit_path for match in matches],
                matched_predicted_actions=[
                    match.predicted_action for match in matches
                ],
                audit_fallback_selected=any(
                    match.fallback_selected for match in matches
                ),
            )
        )
    return examples


def _record_key(record: StrategySignalRecord) -> tuple[str, int, str]:
    return (
        str(Path(record.path).resolve()),
        int(record.step),
        str(record.recorded_action),
    )


def _decision_key(decision: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(Path(str(decision.get("path", ""))).resolve()),
        int(decision.get("step", 0) or 0),
        str(decision.get("recorded_action", "")),
    )


def _start_metric_buckets(
    records: list[StrategySignalRecord],
) -> dict[str, dict[str, int]]:
    return {
        metric: _count(
            _start_metric_bucket(metric, record.start_metrics.get(metric, 0.0))
            for record in records
        )
        for metric in VETO_START_METRICS
    }


def _last_window_metric_buckets(
    records: list[StrategySignalRecord],
) -> dict[str, dict[str, int]]:
    metrics = sorted(
        {
            metric
            for record in records
            for metric in record.metrics_by_window.get(
                _last_window(record.metrics_by_window),
                {},
            )
        }
    )
    return {
        metric: _count(
            _window_metric_bucket(
                metric,
                record.metrics_by_window.get(
                    _last_window(record.metrics_by_window),
                    {},
                ).get(metric, 0.0),
            )
            for record in records
        )
        for metric in metrics
    }


def _metric_averages(
    records: list[StrategySignalRecord],
    metrics: Iterable[str],
) -> dict[str, float]:
    averages: dict[str, float] = {}
    for metric in metrics:
        values = [float(record.start_metrics.get(metric, 0.0)) for record in records]
        if values:
            averages[metric] = float(sum(values) / len(values))
    return dict(sorted(averages.items()))


def _last_window_metric_averages(
    records: list[StrategySignalRecord],
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for record in records:
        metrics = record.metrics_by_window.get(_last_window(record.metrics_by_window), {})
        for name, value in metrics.items():
            totals[name] += float(value)
            counts[name] += 1
    return {
        name: float(totals[name] / counts[name])
        for name in sorted(totals)
        if counts[name]
    }


def _event_counts(
    records: list[StrategySignalRecord],
    attr: str,
) -> dict[str, dict[str, int]]:
    by_window: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        events_by_window = getattr(record, attr)
        for window, events in events_by_window.items():
            by_window[window].update(str(event) for event in events)
    return {
        window: _sorted_counter(counter)
        for window, counter in sorted(by_window.items(), key=lambda item: _window_seconds(item[0]))
    }


def _last_window(metrics_by_window: dict[str, dict[str, float]]) -> str:
    if not metrics_by_window:
        return ""
    return sorted(metrics_by_window, key=_window_seconds)[-1]


def _start_metric_bucket(metric: str, value: float | int | None) -> str:
    number = float(value or 0.0)
    if metric.startswith("base_under_"):
        return "true" if number > 0.0 else "false"
    if metric in {"minerals", "vespene"}:
        return _range_bucket(
            number,
            ((50.0, "0-49"), (100.0, "50-99"), (150.0, "100-149"), (200.0, "150-199"), (400.0, "200-399")),
            "400+",
        )
    if metric == "supply_left":
        return _range_bucket(
            number,
            ((1.0, "0"), (3.0, "1-2"), (6.0, "3-5"), (10.0, "6-9")),
            "10+",
        )
    if metric == "workers":
        return _range_bucket(
            number,
            ((16.0, "0-15"), (24.0, "16-23"), (36.0, "24-35")),
            "36+",
        )
    if metric == "army_count":
        return _range_bucket(
            number,
            ((5.0, "0-4"), (10.0, "5-9"), (20.0, "10-19")),
            "20+",
        )
    return _range_bucket(
        number,
        ((1.0, "0"), (2.0, "1"), (3.0, "2")),
        "3+",
    )


def _window_metric_bucket(metric: str, value: float | int | None) -> str:
    number = float(value or 0.0)
    if metric.endswith("_delta"):
        if number < -5.0:
            return "<-5"
        if number < 0.0:
            return "-5..-1"
        if number == 0.0:
            return "0"
        if number <= 5.0:
            return "1..5"
        return ">5"
    if metric.endswith("_after"):
        base_metric = metric.removesuffix("_after")
        return _start_metric_bucket(base_metric, number)
    return _start_metric_bucket(metric, number)


def _range_bucket(
    value: float,
    thresholds: Iterable[tuple[float, str]],
    overflow: str,
) -> str:
    for upper_bound, label in thresholds:
        if value < upper_bound:
            return label
    return overflow


def _window_seconds(key: str) -> float:
    if key.endswith("s"):
        return float(key[:-1])
    if not key:
        return 0.0
    return float(key)


def _count(values: Iterable[str]) -> dict[str, int]:
    return _sorted_counter(Counter(str(value) for value in values))


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in counter.items()))
