"""Active-threat outcome diagnostics for tactic-filtered static defense rows."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rl.strategy_datasets import (
    StrategyTrajectoryPathInput,
    discover_strategy_trajectory_files,
)
from rl.strategy_outcome_diagnostics import (
    DEFAULT_LOOKAHEAD_SECONDS,
    OutcomeWindowSummary,
    _OutcomeAccumulator,
    _StrategyOutcomeRow,
    _input_paths,
    _iter_valid_strategy_rows,
    _source_for_file,
    _value,
    _window_outcome,
    window_key,
)


STATIC_DEFENSE_MINERALS = 100.0
STATIC_DEFENSE_CONTEXT_ORDER: tuple[str, ...] = (
    "no_static_affordable",
    "no_static_mineral_short",
    "pending_static_waiting",
    "pending_static_with_ready",
    "ready_static_low_minerals",
    "ready_static_affordable",
    "other",
)
START_METRIC_FIELDS: tuple[str, ...] = (
    "minerals",
    "vespene",
    "supply_left",
    "army_count",
    "ready_gateways",
    "gateway_idle_count",
    "pending_static_defense",
    "ready_static_defense",
    "base_under_threat",
    "base_under_air_threat",
    "base_under_ground_threat",
)


@dataclass(frozen=True)
class ActiveThreatFilterOutcomeSummary:
    """Outcome summary for one active-threat static-defense filter bucket."""

    opponent_ai_build: str
    tactic_id: str
    before_action: str
    after_action: str
    context: str
    count: int
    avg_start_metrics: dict[str, float]
    outcomes_by_window: dict[str, OutcomeWindowSummary]


@dataclass(frozen=True)
class FileActiveThreatOutcomeSummary:
    """Per-file active-threat static-defense filter summary."""

    path: str
    source: str
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    result: str | None
    rows: int
    training_rows: int
    active_threat_filter_rows: int
    first_active_threat_filter_time: float | None
    context_counts: dict[str, int]
    filter_change_counts: dict[str, int]


@dataclass(frozen=True)
class ActiveThreatOutcomeDiagnostics:
    """Dataset-level outcome diagnostics for active-threat static-defense rewrites."""

    inputs: list[str]
    lookahead_seconds: list[float]
    files: int
    rows: int
    training_rows: int
    active_threat_filter_rows: int
    result_counts: dict[str, int]
    context_summaries: list[ActiveThreatFilterOutcomeSummary]
    file_summaries: list[FileActiveThreatOutcomeSummary]


class _StartMetricAccumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.totals: dict[str, float] = defaultdict(float)

    def add(self, row: _StrategyOutcomeRow) -> None:
        self.samples += 1
        for field in START_METRIC_FIELDS:
            self.totals[field] += _value(row.observation, field)

    def averages(self) -> dict[str, float]:
        if not self.samples:
            return {}
        return {
            field: total / self.samples
            for field, total in sorted(self.totals.items())
        }


def diagnose_active_threat_outcomes(
    paths: StrategyTrajectoryPathInput,
    *,
    lookahead_seconds: Iterable[float] = DEFAULT_LOOKAHEAD_SECONDS,
) -> ActiveThreatOutcomeDiagnostics:
    """Diagnose active-threat outcomes for BUILD_STATIC_DEFENSE filter rewrites."""
    input_paths = _input_paths(paths)
    lookaheads = tuple(float(value) for value in lookahead_seconds)
    files = discover_strategy_trajectory_files(paths)
    input_strings = [str(path) for path in input_paths]

    rows_total = 0
    training_rows_total = 0
    active_rows_total = 0
    result_counts: Counter[str] = Counter()
    bucket_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    bucket_start_acc: dict[
        tuple[str, str, str, str, str],
        _StartMetricAccumulator,
    ] = defaultdict(_StartMetricAccumulator)
    bucket_window_acc: dict[
        tuple[str, str, str, str, str],
        dict[str, _OutcomeAccumulator],
    ] = defaultdict(
        lambda: {window_key(seconds): _OutcomeAccumulator() for seconds in lookaheads}
    )
    file_summaries: list[FileActiveThreatOutcomeSummary] = []

    for path in files:
        source = _source_for_file(path, input_paths)
        file_rows = list(_iter_valid_strategy_rows(path, source=source))
        training_rows = [row for row in file_rows if not row.done]
        rows_total += len(file_rows)
        training_rows_total += len(training_rows)
        for row in file_rows:
            if row.done:
                result_counts[_result_key(row.result)] += 1

        file_active_rows: list[_StrategyOutcomeRow] = []
        file_context_counts: Counter[str] = Counter()
        file_change_counts: Counter[str] = Counter()
        for index, row in enumerate(file_rows):
            if row.done or not _is_active_static_filter_row(row):
                continue
            context = classify_static_defense_filter_context(row)
            key = (
                row.opponent_ai_build or "RandomBuild",
                row.tactic_id or "<none>",
                row.before_action or "<none>",
                row.after_action or "<none>",
                context,
            )
            active_rows_total += 1
            file_active_rows.append(row)
            file_context_counts[context] += 1
            file_change_counts[f"{row.before_action}->{row.after_action}"] += 1
            bucket_counts[key] += 1
            bucket_start_acc[key].add(row)
            for seconds in lookaheads:
                bucket_window_acc[key][window_key(seconds)].add(
                    _window_outcome(file_rows, index, seconds)
                )

        file_summaries.append(
            _summarize_file(
                path=path,
                source=source,
                rows=file_rows,
                training_rows=training_rows,
                active_rows=file_active_rows,
                context_counts=file_context_counts,
                filter_change_counts=file_change_counts,
            )
        )

    return ActiveThreatOutcomeDiagnostics(
        inputs=input_strings,
        lookahead_seconds=list(lookaheads),
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        active_threat_filter_rows=active_rows_total,
        result_counts=dict(sorted(result_counts.items())),
        context_summaries=_context_summaries(
            bucket_counts,
            bucket_start_acc,
            bucket_window_acc,
        ),
        file_summaries=file_summaries,
    )


def classify_static_defense_filter_context(row: _StrategyOutcomeRow) -> str:
    """Return context for an active-threat BUILD_STATIC_DEFENSE rewrite."""
    ready_static = _value(row.observation, "ready_static_defense") > 0.0
    pending_static = _value(row.observation, "pending_static_defense") > 0.0
    has_minerals = _value(row.observation, "minerals") >= STATIC_DEFENSE_MINERALS
    if pending_static:
        if ready_static:
            return "pending_static_with_ready"
        return "pending_static_waiting"
    if ready_static and has_minerals:
        return "ready_static_affordable"
    if ready_static:
        return "ready_static_low_minerals"
    if has_minerals:
        return "no_static_affordable"
    return "no_static_mineral_short"


def _context_summaries(
    counts: Counter[tuple[str, str, str, str, str]],
    start_acc: dict[tuple[str, str, str, str, str], _StartMetricAccumulator],
    window_acc: dict[
        tuple[str, str, str, str, str],
        dict[str, _OutcomeAccumulator],
    ],
) -> list[ActiveThreatFilterOutcomeSummary]:
    summaries = [
        ActiveThreatFilterOutcomeSummary(
            opponent_ai_build=build,
            tactic_id=tactic_id,
            before_action=before_action,
            after_action=after_action,
            context=context,
            count=int(count),
            avg_start_metrics=start_acc[key].averages(),
            outcomes_by_window={
                window: accumulator.summary(_seconds_from_window_key(window))
                for window, accumulator in window_acc[key].items()
            },
        )
        for key, count in counts.items()
        for build, tactic_id, before_action, after_action, context in (key,)
    ]
    return sorted(
        summaries,
        key=lambda item: (
            -item.count,
            item.opponent_ai_build,
            item.tactic_id,
            _context_sort_index(item.context),
            item.after_action,
        ),
    )


def _summarize_file(
    *,
    path: Path,
    source: str,
    rows: list[_StrategyOutcomeRow],
    training_rows: list[_StrategyOutcomeRow],
    active_rows: list[_StrategyOutcomeRow],
    context_counts: Counter[str],
    filter_change_counts: Counter[str],
) -> FileActiveThreatOutcomeSummary:
    return FileActiveThreatOutcomeSummary(
        path=str(path),
        source=source,
        map_name=_first_non_empty(rows, "map_name"),
        difficulty=_first_non_empty(rows, "difficulty"),
        opponent_race=_first_non_empty(rows, "opponent_race"),
        opponent_ai_build=_first_non_empty(rows, "opponent_ai_build", default="RandomBuild"),
        result=next((row.result for row in rows if row.done and row.result), None),
        rows=len(rows),
        training_rows=len(training_rows),
        active_threat_filter_rows=len(active_rows),
        first_active_threat_filter_time=(
            min(row.game_time for row in active_rows) if active_rows else None
        ),
        context_counts=dict(sorted(context_counts.items())),
        filter_change_counts=dict(sorted(filter_change_counts.items())),
    )


def _is_active_static_filter_row(row: _StrategyOutcomeRow) -> bool:
    return (
        row.before_action == "BUILD_STATIC_DEFENSE"
        and row.after_action is not None
        and row.after_action != "BUILD_STATIC_DEFENSE"
        and _value(row.observation, "base_under_threat") > 0.0
    )


def _first_non_empty(
    rows: list[_StrategyOutcomeRow],
    attr: str,
    *,
    default: str = "",
) -> str:
    for row in rows:
        value = getattr(row, attr)
        if value:
            return str(value)
    return default


def _context_sort_index(context: str) -> int:
    try:
        return STATIC_DEFENSE_CONTEXT_ORDER.index(context)
    except ValueError:
        return len(STATIC_DEFENSE_CONTEXT_ORDER)


def _seconds_from_window_key(key: str) -> float:
    if key.endswith("s"):
        return float(key[:-1])
    return float(key)


def _result_key(value: object) -> str:
    if value is None or value == "":
        return "NO_RESULT"
    return str(value)
