"""Replay-only diagnostics for strategy candidate actions."""
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
    _OutcomeSample,
    _StrategyOutcomeRow,
    _input_paths,
    _iter_valid_strategy_rows,
    _source_for_file,
    _value,
    _window_outcome,
    window_key,
)


DEFAULT_CANDIDATE_SOURCE = "before_filter"
EXPAND_MINERALS = 400.0
GATEWAY_MINERALS = 150.0
ROBO_MINERALS = 150.0
ROBO_VESPENE = 100.0
FORGE_MINERALS = 150.0
STATIC_DEFENSE_MINERALS = 100.0
OBSERVER_MINERALS = 25.0
OBSERVER_VESPENE = 75.0
OBSERVER_SUPPLY = 1.0
IMMORTAL_MINERALS = 275.0
IMMORTAL_VESPENE = 100.0
IMMORTAL_SUPPLY = 4.0
PROBE_MINERALS = 50.0
DEFAULT_TARGET_BASES = 2.0
DEFAULT_GATEWAYS_PER_BASE = 4.0
DEFAULT_STATIC_DEFENSE_PER_BASE = 2.0
PATCH_REVIEW_MAX_CHANGED_ROWS = 20
PATCH_REVIEW_MAX_LARGEST_GROUP_ROWS = 10
PATCH_REVIEW_MIN_EXECUTABLE_RATIO = 0.70

START_METRIC_FIELDS: tuple[str, ...] = (
    "minerals",
    "vespene",
    "supply_left",
    "workers",
    "own_bases",
    "pending_bases",
    "ready_gateways",
    "pending_gateways",
    "gateway_idle_count",
    "ready_robo",
    "pending_robo",
    "robo_idle_count",
    "ready_forge",
    "pending_forge",
    "ready_static_defense",
    "pending_static_defense",
    "has_cybernetics_core",
    "army_count",
    "observers",
    "immortals",
    "base_under_threat",
    "base_under_air_threat",
    "base_under_ground_threat",
)


@dataclass(frozen=True)
class ReplayCandidateTimelineEvent:
    """One changed replay candidate row with recorded future outcomes."""

    path: str
    source: str
    step: int
    game_time: float
    opponent_ai_build: str
    tactic_id: str
    recorded_action: str
    candidate_action: str
    candidate_source: str
    context: str
    threat_state: str
    immediate_candidate_executable: bool
    candidate_blocker: str | None
    start_metrics: dict[str, float]
    outcomes_by_window: dict[str, OutcomeWindowSummary]


@dataclass(frozen=True)
class ReplayCandidateGroupSummary:
    """Aggregated replay-candidate outcomes for one changed-row group."""

    source: str
    opponent_ai_build: str
    tactic_id: str
    recorded_action: str
    candidate_action: str
    candidate_source: str
    context: str
    threat_state: str
    count: int
    immediate_candidate_executable_rows: int
    action_delta_by_name: dict[str, int]
    avg_start_metrics: dict[str, float]
    outcomes_by_window: dict[str, OutcomeWindowSummary]


@dataclass(frozen=True)
class ReplayCandidateFileSummary:
    """Per-file replay-candidate changed-row summary."""

    path: str
    source: str
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    result: str | None
    rows: int
    training_rows: int
    candidate_rows: int
    changed_rows: int
    immediate_candidate_executable_rows: int
    action_delta_by_name: dict[str, int]
    timeline_events: list[ReplayCandidateTimelineEvent]


@dataclass(frozen=True)
class ReplayCandidateGateDecision:
    """Replay-only decision on whether a runtime tactic patch is narrow enough."""

    recommendation: str
    runtime_patch_candidate: bool
    blocking_reasons: list[str]
    warnings: list[str]
    changed_rows: int
    immediate_candidate_executable_rows: int
    executable_ratio: float
    largest_group_count: int
    largest_group_executable_ratio: float
    max_changed_rows_for_patch: int
    max_largest_group_rows_for_patch: int
    min_executable_ratio_for_patch: float


@dataclass(frozen=True)
class StrategyReplayCandidateDiagnostics:
    """Dataset-level replay-only diagnostics for candidate actions."""

    inputs: list[str]
    candidate_source: str
    gate_decision: ReplayCandidateGateDecision
    lookahead_seconds: list[float]
    files: int
    rows: int
    training_rows: int
    candidate_rows: int
    changed_rows: int
    immediate_candidate_executable_rows: int
    action_delta_by_name: dict[str, int]
    group_summaries: list[ReplayCandidateGroupSummary]
    file_summaries: list[ReplayCandidateFileSummary]


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


def diagnose_strategy_replay_candidate(
    paths: StrategyTrajectoryPathInput,
    *,
    candidate_source: str = DEFAULT_CANDIDATE_SOURCE,
    lookahead_seconds: Iterable[float] = DEFAULT_LOOKAHEAD_SECONDS,
) -> StrategyReplayCandidateDiagnostics:
    """Replay candidate actions against recorded rows and future outcome slices."""
    if candidate_source != DEFAULT_CANDIDATE_SOURCE:
        raise ValueError("only candidate_source='before_filter' is supported")

    input_paths = _input_paths(paths)
    input_strings = [str(path) for path in input_paths]
    lookaheads = tuple(float(value) for value in lookahead_seconds)
    files = discover_strategy_trajectory_files(paths)

    rows_total = 0
    training_rows_total = 0
    candidate_rows_total = 0
    changed_rows_total = 0
    executable_rows_total = 0
    action_delta: Counter[str] = Counter()
    group_counts: Counter[tuple[str, str, str, str, str, str, str, str]] = Counter()
    group_executable_counts: Counter[
        tuple[str, str, str, str, str, str, str, str]
    ] = Counter()
    group_start_acc: dict[
        tuple[str, str, str, str, str, str, str, str],
        _StartMetricAccumulator,
    ] = defaultdict(_StartMetricAccumulator)
    group_window_acc: dict[
        tuple[str, str, str, str, str, str, str, str],
        dict[str, _OutcomeAccumulator],
    ] = defaultdict(
        lambda: {window_key(seconds): _OutcomeAccumulator() for seconds in lookaheads}
    )
    file_summaries: list[ReplayCandidateFileSummary] = []

    for path in files:
        source = _source_for_file(path, input_paths)
        file_rows = list(_iter_valid_strategy_rows(path, source=source))
        training_rows = [row for row in file_rows if not row.done]
        rows_total += len(file_rows)
        training_rows_total += len(training_rows)
        file_events: list[ReplayCandidateTimelineEvent] = []
        file_delta: Counter[str] = Counter()
        file_executable_rows = 0
        file_candidate_rows = 0

        for index, row in enumerate(file_rows):
            if row.done:
                continue
            candidate_action = _candidate_action(row, candidate_source)
            if candidate_action is None:
                continue
            candidate_rows_total += 1
            file_candidate_rows += 1
            if candidate_action == row.action_name:
                continue

            changed_rows_total += 1
            executable, blocker = candidate_executability(row, candidate_action)
            if executable:
                executable_rows_total += 1
                file_executable_rows += 1
            _add_action_delta(action_delta, row.action_name, candidate_action)
            _add_action_delta(file_delta, row.action_name, candidate_action)

            context = classify_replay_context(row, candidate_action)
            threat_state = classify_threat_state(row)
            group_key = (
                source,
                row.opponent_ai_build or "RandomBuild",
                row.tactic_id or "<none>",
                row.action_name,
                candidate_action,
                candidate_source,
                context,
                threat_state,
            )
            group_counts[group_key] += 1
            if executable:
                group_executable_counts[group_key] += 1
            group_start_acc[group_key].add(row)

            event_windows: dict[str, OutcomeWindowSummary] = {}
            for seconds in lookaheads:
                sample = _window_outcome(file_rows, index, seconds)
                key = window_key(seconds)
                group_window_acc[group_key][key].add(sample)
                event_windows[key] = _single_window_summary(sample, seconds)

            file_events.append(
                ReplayCandidateTimelineEvent(
                    path=str(path),
                    source=source,
                    step=row.step,
                    game_time=row.game_time,
                    opponent_ai_build=row.opponent_ai_build or "RandomBuild",
                    tactic_id=row.tactic_id or "<none>",
                    recorded_action=row.action_name,
                    candidate_action=candidate_action,
                    candidate_source=candidate_source,
                    context=context,
                    threat_state=threat_state,
                    immediate_candidate_executable=executable,
                    candidate_blocker=blocker,
                    start_metrics=_start_metrics(row),
                    outcomes_by_window=event_windows,
                )
            )

        file_summaries.append(
            _summarize_file(
                path=path,
                source=source,
                rows=file_rows,
                training_rows=training_rows,
                timeline_events=file_events,
                candidate_rows=file_candidate_rows,
                executable_rows=file_executable_rows,
                action_delta=file_delta,
            )
        )

    group_summaries = _group_summaries(
        group_counts,
        group_executable_counts,
        group_start_acc,
        group_window_acc,
    )

    return StrategyReplayCandidateDiagnostics(
        inputs=input_strings,
        candidate_source=candidate_source,
        gate_decision=decide_replay_candidate_gate(
            changed_rows=changed_rows_total,
            immediate_candidate_executable_rows=executable_rows_total,
            group_summaries=group_summaries,
        ),
        lookahead_seconds=list(lookaheads),
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        candidate_rows=candidate_rows_total,
        changed_rows=changed_rows_total,
        immediate_candidate_executable_rows=executable_rows_total,
        action_delta_by_name=_sorted_nonzero_counts(action_delta),
        group_summaries=group_summaries,
        file_summaries=file_summaries,
    )


def decide_replay_candidate_gate(
    *,
    changed_rows: int,
    immediate_candidate_executable_rows: int,
    group_summaries: list[ReplayCandidateGroupSummary],
) -> ReplayCandidateGateDecision:
    """Return a conservative replay-only gate decision for runtime patch review."""
    executable_ratio = _ratio(immediate_candidate_executable_rows, changed_rows)
    largest_group = max(group_summaries, key=lambda summary: summary.count, default=None)
    largest_group_count = largest_group.count if largest_group else 0
    largest_group_executable_ratio = (
        _ratio(largest_group.immediate_candidate_executable_rows, largest_group.count)
        if largest_group
        else 0.0
    )

    if changed_rows <= 0:
        return ReplayCandidateGateDecision(
            recommendation="no_candidate_changes",
            runtime_patch_candidate=False,
            blocking_reasons=[],
            warnings=[],
            changed_rows=0,
            immediate_candidate_executable_rows=0,
            executable_ratio=0.0,
            largest_group_count=0,
            largest_group_executable_ratio=0.0,
            max_changed_rows_for_patch=PATCH_REVIEW_MAX_CHANGED_ROWS,
            max_largest_group_rows_for_patch=PATCH_REVIEW_MAX_LARGEST_GROUP_ROWS,
            min_executable_ratio_for_patch=PATCH_REVIEW_MIN_EXECUTABLE_RATIO,
        )

    blocking_reasons: list[str] = []
    if changed_rows > PATCH_REVIEW_MAX_CHANGED_ROWS:
        blocking_reasons.append("candidate_surface_too_broad")
    if executable_ratio < PATCH_REVIEW_MIN_EXECUTABLE_RATIO:
        blocking_reasons.append("candidate_executability_low")
    if largest_group_count > PATCH_REVIEW_MAX_LARGEST_GROUP_ROWS:
        blocking_reasons.append("largest_group_surface_too_broad")
    if largest_group_executable_ratio < PATCH_REVIEW_MIN_EXECUTABLE_RATIO:
        blocking_reasons.append("largest_group_executability_low")

    runtime_patch_candidate = not blocking_reasons
    return ReplayCandidateGateDecision(
        recommendation=(
            "review_narrow_runtime_patch"
            if runtime_patch_candidate
            else "hold_runtime_patch"
        ),
        runtime_patch_candidate=runtime_patch_candidate,
        blocking_reasons=blocking_reasons,
        warnings=[],
        changed_rows=changed_rows,
        immediate_candidate_executable_rows=immediate_candidate_executable_rows,
        executable_ratio=executable_ratio,
        largest_group_count=largest_group_count,
        largest_group_executable_ratio=largest_group_executable_ratio,
        max_changed_rows_for_patch=PATCH_REVIEW_MAX_CHANGED_ROWS,
        max_largest_group_rows_for_patch=PATCH_REVIEW_MAX_LARGEST_GROUP_ROWS,
        min_executable_ratio_for_patch=PATCH_REVIEW_MIN_EXECUTABLE_RATIO,
    )


def candidate_executability(
    row: _StrategyOutcomeRow,
    candidate_action: str,
) -> tuple[bool, str | None]:
    """Return whether a candidate action appears executable from observation."""
    if candidate_action == "STAY_COURSE":
        return True, None
    if candidate_action == "EXPAND":
        if _value(row.observation, "own_bases") + _value(
            row.observation,
            "pending_bases",
        ) >= DEFAULT_TARGET_BASES:
            return False, "target_bases_reached"
        if _value(row.observation, "minerals") < EXPAND_MINERALS:
            return False, "cannot_afford_nexus"
        return True, None
    if candidate_action == "ADD_GATEWAYS":
        bases = max(_value(row.observation, "own_bases"), 1.0)
        target_gateways = bases * DEFAULT_GATEWAYS_PER_BASE
        total_gateways = _value(row.observation, "ready_gateways") + _value(
            row.observation,
            "pending_gateways",
        )
        if total_gateways >= target_gateways:
            return False, "target_gateways_reached"
        if _value(row.observation, "minerals") < GATEWAY_MINERALS:
            return False, "cannot_afford_gateway"
        return True, None
    if candidate_action == "TECH_ROBO":
        if _value(row.observation, "has_cybernetics_core") <= 0.0:
            return False, "missing_cybernetics_core"
        if (
            _value(row.observation, "ready_robo")
            + _value(row.observation, "pending_robo")
            > 0.0
        ):
            return False, "robo_already_started"
        if (
            _value(row.observation, "minerals") < ROBO_MINERALS
            or _value(row.observation, "vespene") < ROBO_VESPENE
        ):
            return False, "cannot_afford_robo"
        return True, None
    if candidate_action == "FORGE_UPGRADES":
        if (
            _value(row.observation, "ready_forge")
            + _value(row.observation, "pending_forge")
            <= 0.0
        ):
            if _value(row.observation, "minerals") < FORGE_MINERALS:
                return False, "cannot_afford_forge"
            return True, None
        if (
            _value(row.observation, "ground_weapon_upgrade_pending") > 0.0
            and _value(row.observation, "ground_armor_upgrade_pending") > 0.0
        ):
            return False, "upgrade_already_pending"
        if (
            _value(row.observation, "minerals") < STATIC_DEFENSE_MINERALS
            or _value(row.observation, "vespene") < ROBO_VESPENE
        ):
            return False, "cannot_afford_upgrade"
        return True, None
    if candidate_action == "BUILD_STATIC_DEFENSE":
        bases = max(_value(row.observation, "own_bases"), 1.0)
        static_total = _value(row.observation, "ready_static_defense") + _value(
            row.observation,
            "pending_static_defense",
        )
        if static_total >= bases * DEFAULT_STATIC_DEFENSE_PER_BASE:
            return False, "static_defense_cap_reached"
        if (
            _value(row.observation, "has_cybernetics_core") <= 0.0
            and _value(row.observation, "ready_forge") <= 0.0
        ):
            return False, "missing_static_defense_tech"
        if _value(row.observation, "minerals") < STATIC_DEFENSE_MINERALS:
            return False, "cannot_afford_static_defense"
        return True, None
    if candidate_action == "PRODUCE_ARMY":
        if _value(row.observation, "supply_left") <= 0.0:
            return False, "supply_blocked_army"
        if _can_train_robo_unit(row) or _can_train_gateway_unit(row):
            return True, None
        if (
            _value(row.observation, "ready_robo") <= 0.0
            and _value(row.observation, "ready_gateways") <= 0.0
        ):
            return False, "no_ready_production"
        return False, "cannot_afford_army"
    if candidate_action == "BOOST_WORKERS":
        if _value(row.observation, "supply_left") <= 0.0:
            return False, "supply_blocked_probe"
        if _value(row.observation, "own_bases") <= 0.0:
            return False, "no_ready_nexus"
        if _value(row.observation, "minerals") < PROBE_MINERALS:
            return False, "cannot_afford_probe"
        return True, None
    return False, "unknown_candidate_action"


def classify_replay_context(
    row: _StrategyOutcomeRow,
    candidate_action: str,
) -> str:
    """Classify the start-row context for a replay candidate action."""
    if candidate_action == "BUILD_STATIC_DEFENSE":
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
    if candidate_action == "TECH_ROBO":
        if _value(row.observation, "ready_robo") > 0.0:
            return "ready_robo_already_exists"
        if _value(row.observation, "pending_robo") > 0.0:
            return "pending_robo_cap"
        if _value(row.observation, "base_under_threat") > 0.0:
            return "base_under_threat"
        if _value(row.observation, "has_cybernetics_core") <= 0.0:
            return "no_cybernetics_core"
        has_minerals = _value(row.observation, "minerals") >= ROBO_MINERALS
        has_vespene = _value(row.observation, "vespene") >= ROBO_VESPENE
        if has_minerals and has_vespene:
            return "first_robo_affordable"
        if not has_minerals and has_vespene:
            return "first_robo_mineral_short"
        if has_minerals and not has_vespene:
            return "first_robo_vespene_short"
        return "first_robo_resource_short"
    return "other"


def classify_threat_state(row: _StrategyOutcomeRow) -> str:
    """Classify the start-row threat state."""
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


def _can_train_robo_unit(row: _StrategyOutcomeRow) -> bool:
    if (
        _value(row.observation, "ready_robo") <= 0.0
        or _value(row.observation, "robo_idle_count") <= 0.0
    ):
        return False
    if _value(row.observation, "observers") <= 0.0:
        return (
            _value(row.observation, "minerals") >= OBSERVER_MINERALS
            and _value(row.observation, "vespene") >= OBSERVER_VESPENE
            and _value(row.observation, "supply_left") >= OBSERVER_SUPPLY
        )
    return (
        _value(row.observation, "minerals") >= IMMORTAL_MINERALS
        and _value(row.observation, "vespene") >= IMMORTAL_VESPENE
        and _value(row.observation, "supply_left") >= IMMORTAL_SUPPLY
    )


def _can_train_gateway_unit(row: _StrategyOutcomeRow) -> bool:
    return (
        _value(row.observation, "ready_gateways") > 0.0
        and _value(row.observation, "gateway_idle_count") > 0.0
        and _value(row.observation, "minerals") >= STATIC_DEFENSE_MINERALS
        and _value(row.observation, "supply_left") > 0.0
    )


def _group_summaries(
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
) -> list[ReplayCandidateGroupSummary]:
    summaries = []
    for key, count in counts.items():
        (
            source,
            build,
            tactic_id,
            recorded_action,
            candidate_action,
            candidate_source,
            context,
            threat_state,
        ) = key
        summaries.append(
            ReplayCandidateGroupSummary(
                source=source,
                opponent_ai_build=build,
                tactic_id=tactic_id,
                recorded_action=recorded_action,
                candidate_action=candidate_action,
                candidate_source=candidate_source,
                context=context,
                threat_state=threat_state,
                count=int(count),
                immediate_candidate_executable_rows=int(executable_counts[key]),
                action_delta_by_name=_action_delta_for_group(
                    recorded_action,
                    candidate_action,
                    int(count),
                ),
                avg_start_metrics=start_acc[key].averages(),
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
            item.recorded_action,
            item.candidate_action,
            item.context,
            item.threat_state,
        ),
    )


def _summarize_file(
    *,
    path: Path,
    source: str,
    rows: list[_StrategyOutcomeRow],
    training_rows: list[_StrategyOutcomeRow],
    timeline_events: list[ReplayCandidateTimelineEvent],
    candidate_rows: int,
    executable_rows: int,
    action_delta: Counter[str],
) -> ReplayCandidateFileSummary:
    return ReplayCandidateFileSummary(
        path=str(path),
        source=source,
        map_name=_first_non_empty(rows, "map_name"),
        difficulty=_first_non_empty(rows, "difficulty"),
        opponent_race=_first_non_empty(rows, "opponent_race"),
        opponent_ai_build=_first_non_empty(
            rows,
            "opponent_ai_build",
            default="RandomBuild",
        ),
        result=next((row.result for row in rows if row.done and row.result), None),
        rows=len(rows),
        training_rows=len(training_rows),
        candidate_rows=candidate_rows,
        changed_rows=len(timeline_events),
        immediate_candidate_executable_rows=executable_rows,
        action_delta_by_name=_sorted_nonzero_counts(action_delta),
        timeline_events=timeline_events,
    )


def _candidate_action(
    row: _StrategyOutcomeRow,
    candidate_source: str,
) -> str | None:
    if candidate_source == DEFAULT_CANDIDATE_SOURCE:
        return row.before_action
    return None


def _start_metrics(row: _StrategyOutcomeRow) -> dict[str, float]:
    return {
        field: _value(row.observation, field)
        for field in START_METRIC_FIELDS
    }


def _single_window_summary(
    sample: _OutcomeSample,
    seconds: float,
) -> OutcomeWindowSummary:
    accumulator = _OutcomeAccumulator()
    accumulator.add(sample)
    return accumulator.summary(seconds)


def _add_action_delta(
    counts: Counter[str],
    recorded_action: str,
    candidate_action: str,
) -> None:
    counts[recorded_action] -= 1
    counts[candidate_action] += 1


def _action_delta_for_group(
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


def _seconds_from_window_key(key: str) -> float:
    if key.endswith("s"):
        return float(key[:-1])
    return float(key)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
