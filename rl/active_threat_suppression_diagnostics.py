"""Diagnostics for tactic-filter suppression around active-threat failures."""
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


TARGET_TACTICS = frozenset({"RECOVERY", "TECH_POWER"})
TARGET_BEFORE_ACTIONS = frozenset({"BUILD_STATIC_DEFENSE", "TECH_ROBO"})
TARGET_AFTER_ACTIONS = frozenset({"PRODUCE_ARMY", "STAY_COURSE"})
STATIC_DEFENSE_MINERALS = 100.0
INITIAL_ROBO_MINERALS = 150.0
INITIAL_ROBO_VESPENE = 100.0
START_METRIC_FIELDS: tuple[str, ...] = (
    "minerals",
    "vespene",
    "supply_left",
    "army_count",
    "ready_gateways",
    "pending_gateways",
    "gateway_idle_count",
    "pending_robo",
    "ready_robo",
    "robo_idle_count",
    "pending_static_defense",
    "ready_static_defense",
    "base_under_threat",
    "base_under_air_threat",
    "base_under_ground_threat",
)
EVENT_METRIC_FIELDS: tuple[str, ...] = (
    "army_count_delta",
    "static_defense_delta",
    "pending_static_defense_delta",
    "ready_robo_delta",
    "observer_delta",
    "immortal_delta",
    "base_under_threat_after",
    "minerals_after",
    "gateway_idle_after",
    "robo_idle_after",
)


@dataclass(frozen=True)
class SuppressionEventWindow:
    """Single-row outcome snapshot for a lookahead window."""

    lookahead_seconds: float
    metrics: dict[str, float]
    events: dict[str, bool]
    event_times: dict[str, float]


@dataclass(frozen=True)
class SuppressionTimelineEvent:
    """One tactic-filter suppression event in a file timeline."""

    step: int
    game_time: float
    opponent_ai_build: str
    tactic_id: str
    before_action: str
    after_action: str
    candidate_action: str
    context: str
    threat_state: str
    immediate_candidate_executable: bool
    start_metrics: dict[str, float]
    outcomes_by_window: dict[str, SuppressionEventWindow]


@dataclass(frozen=True)
class SuppressionContextOutcomeSummary:
    """Aggregated lookahead outcomes for one suppression context."""

    source: str
    opponent_ai_build: str
    tactic_id: str
    before_action: str
    after_action: str
    candidate_action: str
    context: str
    threat_state: str
    count: int
    immediate_candidate_executable_rows: int
    avg_start_metrics: dict[str, float]
    replay_action_delta_by_name: dict[str, int]
    outcomes_by_window: dict[str, OutcomeWindowSummary]


@dataclass(frozen=True)
class ReplayCandidateContextImpact:
    """Replay-only action delta for one context if suppression is passed through."""

    source: str
    opponent_ai_build: str
    tactic_id: str
    before_action: str
    after_action: str
    candidate_action: str
    context: str
    threat_state: str
    count: int
    immediate_candidate_executable_rows: int
    action_delta_by_name: dict[str, int]
    avg_start_metrics: dict[str, float]


@dataclass(frozen=True)
class ReplayCandidateImpact:
    """Dataset-level replay-only impact for restoring before_action."""

    name: str
    affected_rows: int
    immediate_candidate_executable_rows: int
    action_delta_by_name: dict[str, int]
    context_impacts: list[ReplayCandidateContextImpact]


@dataclass(frozen=True)
class SourceSuppressionSummary:
    """Per-input source overview for comparison with no-filter trajectories."""

    source: str
    files: int
    rows: int
    training_rows: int
    result_counts: dict[str, int]
    action_counts_by_name: dict[str, int]
    threat_action_counts_by_name: dict[str, int]
    filter_change_rows: int
    target_suppression_rows: int


@dataclass(frozen=True)
class FileSuppressionSummary:
    """Per-file suppression counts and timeline."""

    path: str
    source: str
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    result: str | None
    rows: int
    training_rows: int
    target_suppression_rows: int
    first_target_suppression_time: float | None
    context_counts: dict[str, int]
    filter_change_counts: dict[str, int]
    replay_action_delta_by_name: dict[str, int]
    timeline_events: list[SuppressionTimelineEvent]


@dataclass(frozen=True)
class ActiveThreatSuppressionDiagnostics:
    """Diagnostics for suppression rows after a failed tactic-rule A/B."""

    inputs: list[str]
    lookahead_seconds: list[float]
    files: int
    rows: int
    training_rows: int
    target_suppression_rows: int
    result_counts: dict[str, int]
    source_summaries: list[SourceSuppressionSummary]
    context_summaries: list[SuppressionContextOutcomeSummary]
    replay_candidate_impact: ReplayCandidateImpact
    file_summaries: list[FileSuppressionSummary]


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


@dataclass
class _SourceAccumulator:
    source: str
    files: int = 0
    rows: int = 0
    training_rows: int = 0
    filter_change_rows: int = 0
    target_suppression_rows: int = 0

    def __post_init__(self) -> None:
        self.result_counts: Counter[str] = Counter()
        self.action_counts: Counter[str] = Counter()
        self.threat_action_counts: Counter[str] = Counter()

    def to_summary(self) -> SourceSuppressionSummary:
        return SourceSuppressionSummary(
            source=self.source,
            files=self.files,
            rows=self.rows,
            training_rows=self.training_rows,
            result_counts=dict(sorted(self.result_counts.items())),
            action_counts_by_name=dict(sorted(self.action_counts.items())),
            threat_action_counts_by_name=dict(sorted(self.threat_action_counts.items())),
            filter_change_rows=self.filter_change_rows,
            target_suppression_rows=self.target_suppression_rows,
        )


def diagnose_active_threat_suppression(
    paths: StrategyTrajectoryPathInput,
    *,
    lookahead_seconds: Iterable[float] = DEFAULT_LOOKAHEAD_SECONDS,
) -> ActiveThreatSuppressionDiagnostics:
    """Diagnose RECOVERY/TECH_POWER suppression rows and replay-only impact."""
    input_paths = _input_paths(paths)
    lookaheads = tuple(float(value) for value in lookahead_seconds)
    files = discover_strategy_trajectory_files(paths)
    input_strings = [str(path) for path in input_paths]

    rows_total = 0
    training_rows_total = 0
    target_rows_total = 0
    result_counts: Counter[str] = Counter()
    source_acc: dict[str, _SourceAccumulator] = {}
    context_counts: Counter[tuple[str, str, str, str, str, str, str, str]] = Counter()
    context_executable_counts: Counter[
        tuple[str, str, str, str, str, str, str, str]
    ] = Counter()
    context_start_acc: dict[
        tuple[str, str, str, str, str, str, str, str],
        _StartMetricAccumulator,
    ] = defaultdict(_StartMetricAccumulator)
    context_window_acc: dict[
        tuple[str, str, str, str, str, str, str, str],
        dict[str, _OutcomeAccumulator],
    ] = defaultdict(
        lambda: {window_key(seconds): _OutcomeAccumulator() for seconds in lookaheads}
    )
    file_summaries: list[FileSuppressionSummary] = []

    for path in files:
        source = _source_for_file(path, input_paths)
        source_summary = source_acc.setdefault(source, _SourceAccumulator(source))
        file_rows = list(_iter_valid_strategy_rows(path, source=source))
        training_rows = [row for row in file_rows if not row.done]
        rows_total += len(file_rows)
        training_rows_total += len(training_rows)
        source_summary.files += 1
        source_summary.rows += len(file_rows)
        source_summary.training_rows += len(training_rows)

        file_events: list[SuppressionTimelineEvent] = []
        file_context_counts: Counter[str] = Counter()
        file_change_counts: Counter[str] = Counter()
        file_action_delta: Counter[str] = Counter()

        for row in file_rows:
            if row.done:
                key = _result_key(row.result)
                result_counts[key] += 1
                source_summary.result_counts[key] += 1

        for index, row in enumerate(file_rows):
            if row.done:
                continue
            source_summary.action_counts[row.action_name] += 1
            if _value(row.observation, "base_under_threat") > 0.0:
                source_summary.threat_action_counts[row.action_name] += 1
            if _is_filter_change(row):
                source_summary.filter_change_rows += 1
            if not _is_target_suppression_row(row):
                continue

            target_rows_total += 1
            source_summary.target_suppression_rows += 1
            context = classify_suppression_context(row)
            threat_state = classify_threat_state(row)
            candidate_action = str(row.before_action)
            immediate_executable = is_candidate_immediately_executable(row)
            key = (
                source,
                row.opponent_ai_build or "RandomBuild",
                row.tactic_id or "<none>",
                str(row.before_action),
                str(row.after_action),
                candidate_action,
                context,
                threat_state,
            )
            context_counts[key] += 1
            if immediate_executable:
                context_executable_counts[key] += 1
            context_start_acc[key].add(row)
            file_context_counts[_context_label(row, context, threat_state)] += 1
            file_change_counts[f"{row.before_action}->{row.after_action}"] += 1
            _add_action_delta(file_action_delta, row.after_action, candidate_action)

            outcomes_by_window: dict[str, SuppressionEventWindow] = {}
            for seconds in lookaheads:
                outcome = _window_outcome(file_rows, index, seconds)
                key_name = window_key(seconds)
                context_window_acc[key][key_name].add(outcome)
                outcomes_by_window[key_name] = SuppressionEventWindow(
                    lookahead_seconds=seconds,
                    metrics=_select_event_metrics(outcome.metrics),
                    events=dict(sorted(outcome.events.items())),
                    event_times=dict(sorted(outcome.event_times.items())),
                )

            file_events.append(
                SuppressionTimelineEvent(
                    step=row.step,
                    game_time=row.game_time,
                    opponent_ai_build=row.opponent_ai_build or "RandomBuild",
                    tactic_id=row.tactic_id or "<none>",
                    before_action=str(row.before_action),
                    after_action=str(row.after_action),
                    candidate_action=candidate_action,
                    context=context,
                    threat_state=threat_state,
                    immediate_candidate_executable=immediate_executable,
                    start_metrics=_start_metrics(row),
                    outcomes_by_window=outcomes_by_window,
                )
            )

        file_summaries.append(
            _summarize_file(
                path=path,
                source=source,
                rows=file_rows,
                training_rows=training_rows,
                timeline_events=file_events,
                context_counts=file_context_counts,
                filter_change_counts=file_change_counts,
                replay_action_delta=file_action_delta,
            )
        )

    context_summaries = _context_summaries(
        context_counts,
        context_executable_counts,
        context_start_acc,
        context_window_acc,
    )
    return ActiveThreatSuppressionDiagnostics(
        inputs=input_strings,
        lookahead_seconds=list(lookaheads),
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        target_suppression_rows=target_rows_total,
        result_counts=dict(sorted(result_counts.items())),
        source_summaries=[
            acc.to_summary()
            for _, acc in sorted(source_acc.items(), key=lambda item: item[0])
        ],
        context_summaries=context_summaries,
        replay_candidate_impact=_replay_candidate_impact(context_summaries),
        file_summaries=file_summaries,
    )


def classify_suppression_context(row: _StrategyOutcomeRow) -> str:
    """Classify the state around a target suppression row."""
    if row.before_action == "BUILD_STATIC_DEFENSE":
        return classify_static_defense_context(row)
    if row.before_action == "TECH_ROBO":
        return classify_tech_robo_context(row)
    return "other"


def classify_static_defense_context(row: _StrategyOutcomeRow) -> str:
    """Return context for BUILD_STATIC_DEFENSE suppression."""
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


def classify_tech_robo_context(row: _StrategyOutcomeRow) -> str:
    """Return context for TECH_ROBO suppression."""
    if _value(row.observation, "ready_robo") > 0.0:
        return "ready_robo_already_exists"
    if _value(row.observation, "pending_robo") > 0.0:
        return "pending_robo_cap"
    if _value(row.observation, "base_under_threat") > 0.0:
        return "base_under_threat"
    if _value(row.observation, "has_cybernetics_core") <= 0.0:
        return "no_cybernetics_core"
    has_minerals = _value(row.observation, "minerals") >= INITIAL_ROBO_MINERALS
    has_vespene = _value(row.observation, "vespene") >= INITIAL_ROBO_VESPENE
    if has_minerals and has_vespene:
        return "first_robo_affordable"
    if not has_minerals and has_vespene:
        return "first_robo_mineral_short"
    if has_minerals and not has_vespene:
        return "first_robo_vespene_short"
    return "first_robo_resource_short"


def classify_threat_state(row: _StrategyOutcomeRow) -> str:
    """Classify threat state at the start row."""
    if _value(row.observation, "base_under_threat") <= 0.0:
        return "no_threat"
    air = _value(row.observation, "base_under_air_threat") > 0.0
    ground = _value(row.observation, "base_under_ground_threat") > 0.0
    if air and ground:
        return "air_and_ground_threat"
    if air:
        return "air_threat"
    if ground:
        return "ground_threat"
    return "generic_threat"


def is_candidate_immediately_executable(row: _StrategyOutcomeRow) -> bool:
    """Return whether the replay candidate looks executable from observation only."""
    if row.before_action == "BUILD_STATIC_DEFENSE":
        return _value(row.observation, "minerals") >= STATIC_DEFENSE_MINERALS
    if row.before_action == "TECH_ROBO":
        return (
            _value(row.observation, "minerals") >= INITIAL_ROBO_MINERALS
            and _value(row.observation, "vespene") >= INITIAL_ROBO_VESPENE
            and _value(row.observation, "has_cybernetics_core") > 0.0
            and _value(row.observation, "pending_robo") <= 0.0
            and _value(row.observation, "ready_robo") <= 0.0
            and _value(row.observation, "base_under_threat") <= 0.0
        )
    return False


def _context_summaries(
    counts: Counter[tuple[str, str, str, str, str, str, str, str]],
    executable_counts: Counter[tuple[str, str, str, str, str, str, str, str]],
    start_acc: dict[
        tuple[str, str, str, str, str, str, str, str],
        _StartMetricAccumulator,
    ],
    window_acc: dict[
        tuple[str, str, str, str, str, str, str, str],
        dict[str, _OutcomeAccumulator],
    ],
) -> list[SuppressionContextOutcomeSummary]:
    summaries = []
    for key, count in counts.items():
        (
            source,
            build,
            tactic_id,
            before_action,
            after_action,
            candidate_action,
            context,
            threat_state,
        ) = key
        summaries.append(
            SuppressionContextOutcomeSummary(
                source=source,
                opponent_ai_build=build,
                tactic_id=tactic_id,
                before_action=before_action,
                after_action=after_action,
                candidate_action=candidate_action,
                context=context,
                threat_state=threat_state,
                count=int(count),
                immediate_candidate_executable_rows=int(executable_counts[key]),
                avg_start_metrics=start_acc[key].averages(),
                replay_action_delta_by_name=_action_delta_for_context(
                    after_action,
                    candidate_action,
                    int(count),
                ),
                outcomes_by_window={
                    window: accumulator.summary(_seconds_from_window_key(window))
                    for window, accumulator in window_acc[key].items()
                },
            )
        )
    return sorted(
        summaries,
        key=lambda item: (
            -item.count,
            item.source,
            item.tactic_id,
            item.before_action,
            item.after_action,
            item.context,
            item.threat_state,
        ),
    )


def _replay_candidate_impact(
    context_summaries: list[SuppressionContextOutcomeSummary],
) -> ReplayCandidateImpact:
    action_delta: Counter[str] = Counter()
    affected_rows = 0
    executable_rows = 0
    impacts: list[ReplayCandidateContextImpact] = []
    for summary in context_summaries:
        affected_rows += summary.count
        executable_rows += summary.immediate_candidate_executable_rows
        action_delta.update(summary.replay_action_delta_by_name)
        impacts.append(
            ReplayCandidateContextImpact(
                source=summary.source,
                opponent_ai_build=summary.opponent_ai_build,
                tactic_id=summary.tactic_id,
                before_action=summary.before_action,
                after_action=summary.after_action,
                candidate_action=summary.candidate_action,
                context=summary.context,
                threat_state=summary.threat_state,
                count=summary.count,
                immediate_candidate_executable_rows=(
                    summary.immediate_candidate_executable_rows
                ),
                action_delta_by_name=summary.replay_action_delta_by_name,
                avg_start_metrics=summary.avg_start_metrics,
            )
        )
    return ReplayCandidateImpact(
        name="pass_through_before_action",
        affected_rows=affected_rows,
        immediate_candidate_executable_rows=executable_rows,
        action_delta_by_name=_sorted_nonzero_counts(action_delta),
        context_impacts=impacts,
    )


def _summarize_file(
    *,
    path: Path,
    source: str,
    rows: list[_StrategyOutcomeRow],
    training_rows: list[_StrategyOutcomeRow],
    timeline_events: list[SuppressionTimelineEvent],
    context_counts: Counter[str],
    filter_change_counts: Counter[str],
    replay_action_delta: Counter[str],
) -> FileSuppressionSummary:
    return FileSuppressionSummary(
        path=str(path),
        source=source,
        map_name=_first_non_empty(rows, "map_name"),
        difficulty=_first_non_empty(rows, "difficulty"),
        opponent_race=_first_non_empty(rows, "opponent_race"),
        opponent_ai_build=_first_non_empty(rows, "opponent_ai_build", default="RandomBuild"),
        result=next((row.result for row in rows if row.done and row.result), None),
        rows=len(rows),
        training_rows=len(training_rows),
        target_suppression_rows=len(timeline_events),
        first_target_suppression_time=(
            min(event.game_time for event in timeline_events)
            if timeline_events
            else None
        ),
        context_counts=dict(sorted(context_counts.items())),
        filter_change_counts=dict(sorted(filter_change_counts.items())),
        replay_action_delta_by_name=_sorted_nonzero_counts(replay_action_delta),
        timeline_events=timeline_events,
    )


def _is_target_suppression_row(row: _StrategyOutcomeRow) -> bool:
    if row.tactic_id not in TARGET_TACTICS:
        return False
    if row.before_action not in TARGET_BEFORE_ACTIONS:
        return False
    if row.after_action not in TARGET_AFTER_ACTIONS:
        return False
    if row.before_action == row.after_action:
        return False
    if row.before_action == "BUILD_STATIC_DEFENSE":
        return _value(row.observation, "base_under_threat") > 0.0
    return True


def _is_filter_change(row: _StrategyOutcomeRow) -> bool:
    return (
        row.before_action is not None
        and row.after_action is not None
        and row.before_action != row.after_action
    )


def _start_metrics(row: _StrategyOutcomeRow) -> dict[str, float]:
    return {
        field: _value(row.observation, field)
        for field in START_METRIC_FIELDS
    }


def _select_event_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        field: float(metrics[field])
        for field in EVENT_METRIC_FIELDS
        if field in metrics
    }


def _context_label(
    row: _StrategyOutcomeRow,
    context: str,
    threat_state: str,
) -> str:
    return f"{row.tactic_id}/{row.before_action}->{row.after_action}/{context}/{threat_state}"


def _add_action_delta(
    counts: Counter[str],
    recorded_action: str | None,
    candidate_action: str,
) -> None:
    if recorded_action:
        counts[str(recorded_action)] -= 1
    counts[candidate_action] += 1


def _action_delta_for_context(
    recorded_action: str,
    candidate_action: str,
    count: int,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    counts[recorded_action] -= count
    counts[candidate_action] += count
    return _sorted_nonzero_counts(counts)


def _sorted_nonzero_counts(counts: Counter[str]) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in sorted(counts.items())
        if value
    }


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


def _result_key(value: object) -> str:
    if value is None or value == "":
        return "NO_RESULT"
    return str(value)


def _seconds_from_window_key(key: str) -> float:
    if key.endswith("s"):
        return float(key[:-1])
    return float(key)
