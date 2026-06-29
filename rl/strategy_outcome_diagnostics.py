"""Lookahead outcome diagnostics for strategy trajectory JSONL files."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_datasets import StrategyTrajectoryPathInput, discover_strategy_trajectory_files
from rl.strategy_observations import (
    STRATEGY_OBSERVATION_DEFAULTS,
    normalize_strategy_observation_dict,
    validate_strategy_observation_dict,
)


DEFAULT_LOOKAHEAD_SECONDS: tuple[float, ...] = (30.0, 60.0, 90.0, 120.0)
EARLY_GATEWAY_CUTOFF_SECONDS = 240.0
ROBO_PAYOFF_ACTIONS = frozenset({"TECH_ROBO", "PRODUCE_ARMY"})
OBSERVER_MINERALS = 25.0
OBSERVER_VESPENE = 75.0
OBSERVER_SUPPLY = 1.0
IMMORTAL_MINERALS = 275.0
IMMORTAL_VESPENE = 100.0
IMMORTAL_SUPPLY = 4.0


@dataclass(frozen=True)
class OutcomeWindowSummary:
    """Aggregated action outcomes for one lookahead window."""

    lookahead_seconds: float
    samples: int
    avg_metrics: dict[str, float]
    max_metrics: dict[str, float]
    event_counts: dict[str, int]
    event_rates: dict[str, float]
    avg_event_times: dict[str, float]


@dataclass(frozen=True)
class ActionOutcomeSummary:
    """Basic timing summary for one strategy action."""

    action_name: str
    count: int
    first_game_time: float | None
    min_game_time: float | None
    max_game_time: float | None
    avg_game_time: float | None
    early_before_240_count: int


@dataclass(frozen=True)
class FilterChangeOutcomeSummary:
    """Outcome summary for one tactic filter before/after action change."""

    opponent_ai_build: str
    tactic_id: str
    before_action: str
    after_action: str
    count: int
    early_before_240_count: int
    outcomes_by_window: dict[str, OutcomeWindowSummary]


@dataclass(frozen=True)
class SourceOutcomeSummary:
    """Per-input summary, useful when comparing two trajectory directories."""

    source: str
    files: int
    rows: int
    training_rows: int
    result_counts: dict[str, int]
    action_counts_by_name: dict[str, int]
    action_first_game_time_by_name: dict[str, float]
    filter_change_rows: int
    execution_effect_counts: dict[str, int]
    execution_blocker_counts: dict[str, int]


@dataclass(frozen=True)
class RoboPayoffSummary:
    """Per-file payoff and blocker classification after a Robo becomes ready."""

    ready_robo_first_game_time: float | None
    observer_first_game_time: float | None
    immortal_first_game_time: float | None
    observer_after_ready_game_time: float | None
    immortal_after_ready_game_time: float | None
    observer_after_ready_delay_seconds: float | None
    immortal_after_ready_delay_seconds: float | None
    observer_status: str
    observer_blocker: str
    immortal_status: str
    immortal_blocker: str
    robo_action_rows_after_ready: int
    robo_idle_rows_after_ready: int
    observer_candidate_rows_after_ready: int
    immortal_candidate_rows_after_ready: int
    immortal_affordable_candidate_rows_after_ready: int
    immortal_mineral_blocked_candidate_rows: int
    immortal_vespene_blocked_candidate_rows: int
    immortal_supply_blocked_candidate_rows: int


@dataclass(frozen=True)
class FileOutcomeSummary:
    """Per-file strategy outcome summary."""

    path: str
    source: str
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    result: str | None
    rows: int
    training_rows: int
    first_game_time: float | None
    last_game_time: float | None
    action_counts_by_name: dict[str, int]
    action_first_game_time_by_name: dict[str, float]
    filter_change_rows: int
    ready_robo_first_game_time: float | None
    observer_first_game_time: float | None
    immortal_first_game_time: float | None
    base_threat_rows: int
    execution_effect_counts: dict[str, int]
    execution_blocker_counts: dict[str, int]
    robo_payoff: RoboPayoffSummary


@dataclass(frozen=True)
class StrategyOutcomeDiagnostics:
    """Dataset-level lookahead diagnostics for strategy actions."""

    inputs: list[str]
    lookahead_seconds: list[float]
    files: int
    rows: int
    training_rows: int
    result_counts: dict[str, int]
    action_summaries_by_name: dict[str, ActionOutcomeSummary]
    action_window_summaries: dict[str, dict[str, OutcomeWindowSummary]]
    filter_change_summaries: list[FilterChangeOutcomeSummary]
    source_summaries: list[SourceOutcomeSummary]
    file_summaries: list[FileOutcomeSummary]
    execution_effect_counts: dict[str, int]
    execution_blocker_counts: dict[str, int]


@dataclass(frozen=True)
class _StrategyOutcomeRow:
    path: Path
    source: str
    step: int
    game_time: float
    action_id: int
    action_name: str
    done: bool
    result: str | None
    observation: dict[str, float]
    observation_details: dict[str, float]
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    tactic_id: str | None
    before_action: str | None
    after_action: str | None
    execution_attempted: bool | None
    execution_effect: str | None
    execution_blocker: str | None
    execution_unit_type: str | None
    execution_target: str | None


@dataclass(frozen=True)
class _OutcomeSample:
    metrics: dict[str, float]
    events: dict[str, bool]
    event_times: dict[str, float]


class _OutcomeAccumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.metric_totals: dict[str, float] = defaultdict(float)
        self.metric_max: dict[str, float] = {}
        self.event_counts: Counter[str] = Counter()
        self.event_time_totals: dict[str, float] = defaultdict(float)
        self.event_time_counts: Counter[str] = Counter()

    def add(self, sample: _OutcomeSample) -> None:
        self.samples += 1
        for name, value in sample.metrics.items():
            value = float(value)
            self.metric_totals[name] += value
            if name not in self.metric_max or value > self.metric_max[name]:
                self.metric_max[name] = value
        for name, value in sample.events.items():
            if value:
                self.event_counts[name] += 1
        for name, value in sample.event_times.items():
            self.event_time_totals[name] += float(value)
            self.event_time_counts[name] += 1

    def summary(self, lookahead_seconds: float) -> OutcomeWindowSummary:
        avg_metrics = {
            name: total / self.samples
            for name, total in sorted(self.metric_totals.items())
            if self.samples
        }
        event_counts = dict(sorted((name, int(count)) for name, count in self.event_counts.items()))
        return OutcomeWindowSummary(
            lookahead_seconds=lookahead_seconds,
            samples=self.samples,
            avg_metrics=avg_metrics,
            max_metrics=dict(sorted(self.metric_max.items())),
            event_counts=event_counts,
            event_rates={
                name: count / self.samples
                for name, count in event_counts.items()
                if self.samples
            },
            avg_event_times={
                name: total / self.event_time_counts[name]
                for name, total in sorted(self.event_time_totals.items())
                if self.event_time_counts[name]
            },
        )


def diagnose_strategy_outcomes(
    paths: StrategyTrajectoryPathInput,
    *,
    lookahead_seconds: Iterable[float] = DEFAULT_LOOKAHEAD_SECONDS,
) -> StrategyOutcomeDiagnostics:
    """Return lookahead outcome diagnostics for strategy trajectory JSONL files."""
    input_paths = _input_paths(paths)
    lookaheads = tuple(float(value) for value in lookahead_seconds)
    files = discover_strategy_trajectory_files(paths)
    input_strings = [str(path) for path in input_paths]

    rows_total = 0
    training_rows_total = 0
    result_counts: Counter[str] = Counter()
    action_times: dict[int, list[float]] = defaultdict(list)
    action_window_acc: dict[int, dict[str, _OutcomeAccumulator]] = defaultdict(
        lambda: {window_key(seconds): _OutcomeAccumulator() for seconds in lookaheads}
    )
    filter_counts: Counter[tuple[str, str, str, str]] = Counter()
    filter_early_counts: Counter[tuple[str, str, str, str]] = Counter()
    filter_window_acc: dict[
        tuple[str, str, str, str],
        dict[str, _OutcomeAccumulator],
    ] = defaultdict(
        lambda: {window_key(seconds): _OutcomeAccumulator() for seconds in lookaheads}
    )
    source_acc: dict[str, _SourceAccumulator] = {}
    file_summaries: list[FileOutcomeSummary] = []
    execution_effect_counts: Counter[str] = Counter()
    execution_blocker_counts: Counter[str] = Counter()

    for path in files:
        source = _source_for_file(path, input_paths)
        source_summary = source_acc.setdefault(source, _SourceAccumulator(source))
        file_rows = list(_iter_valid_strategy_rows(path, source=source))
        rows_total += len(file_rows)
        training_rows = [row for row in file_rows if not row.done]
        training_rows_total += len(training_rows)
        source_summary.files += 1
        source_summary.rows += len(file_rows)
        source_summary.training_rows += len(training_rows)

        for row in file_rows:
            if row.done:
                result_counts[_result_key(row.result)] += 1
                source_summary.result_counts[_result_key(row.result)] += 1

        for index, row in enumerate(file_rows):
            if row.done:
                continue
            if row.execution_effect:
                execution_effect_counts[row.execution_effect] += 1
                source_summary.execution_effect_counts[row.execution_effect] += 1
            if row.execution_blocker:
                execution_blocker_counts[row.execution_blocker] += 1
                source_summary.execution_blocker_counts[row.execution_blocker] += 1
            action_times[row.action_id].append(row.game_time)
            source_summary.action_counts[row.action_id] += 1
            source_summary.record_action_time(row.action_id, row.game_time)
            per_window_samples = {
                window_key(seconds): _window_outcome(file_rows, index, seconds)
                for seconds in lookaheads
            }
            for key, sample in per_window_samples.items():
                action_window_acc[row.action_id][key].add(sample)

            filter_key = _filter_key(row)
            if filter_key is not None:
                source_summary.filter_change_rows += 1
                filter_counts[filter_key] += 1
                if row.game_time < EARLY_GATEWAY_CUTOFF_SECONDS:
                    filter_early_counts[filter_key] += 1
                for key, sample in per_window_samples.items():
                    filter_window_acc[filter_key][key].add(sample)

        file_summaries.append(_summarize_file(path, source, file_rows, training_rows))

    return StrategyOutcomeDiagnostics(
        inputs=input_strings,
        lookahead_seconds=list(lookaheads),
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        result_counts=dict(sorted(result_counts.items())),
        action_summaries_by_name={
            STRATEGY_ACTION_NAMES[action_id]: _action_summary(action_id, times)
            for action_id, times in sorted(action_times.items())
        },
        action_window_summaries={
            STRATEGY_ACTION_NAMES[action_id]: {
                key: accumulator.summary(_seconds_from_window_key(key))
                for key, accumulator in window_acc.items()
            }
            for action_id, window_acc in sorted(action_window_acc.items())
        },
        filter_change_summaries=_filter_summaries(
            filter_counts,
            filter_early_counts,
            filter_window_acc,
        ),
        source_summaries=[
            summary.to_summary()
            for _, summary in sorted(source_acc.items(), key=lambda item: item[0])
        ],
        file_summaries=file_summaries,
        execution_effect_counts=dict(sorted(execution_effect_counts.items())),
        execution_blocker_counts=dict(sorted(execution_blocker_counts.items())),
    )


def window_key(seconds: float) -> str:
    """Return the stable key used for a lookahead window."""
    return f"{seconds:g}s"


def _window_outcome(
    rows: list[_StrategyOutcomeRow],
    index: int,
    lookahead_seconds: float,
) -> _OutcomeSample:
    start = rows[index]
    deadline = start.game_time + lookahead_seconds
    future = [
        row
        for row in rows[index + 1 :]
        if row.game_time <= deadline
    ]
    endpoint = future[-1] if future else start
    start_obs = start.observation
    end_obs = endpoint.observation

    observer_delta = _delta(start_obs, end_obs, "observers")
    immortal_delta = _delta(start_obs, end_obs, "immortals")
    metrics = {
        "ready_gateway_delta": _delta(start_obs, end_obs, "ready_gateways"),
        "pending_gateway_delta": _delta(start_obs, end_obs, "pending_gateways"),
        "pending_gateway_after": _value(end_obs, "pending_gateways"),
        "pending_robo_delta": _delta(start_obs, end_obs, "pending_robo"),
        "ready_robo_delta": _delta(start_obs, end_obs, "ready_robo"),
        "observer_delta": observer_delta,
        "immortal_delta": immortal_delta,
        "observer_immortal_delta": observer_delta + immortal_delta,
        "army_count_delta": _delta(start_obs, end_obs, "army_count"),
        "zealot_delta": _delta(start_obs, end_obs, "zealots"),
        "stalker_delta": _delta(start_obs, end_obs, "stalkers"),
        "sentry_delta": _delta(start_obs, end_obs, "sentries"),
        "static_defense_delta": _delta(start_obs, end_obs, "ready_static_defense"),
        "pending_static_defense_delta": _delta(
            start_obs,
            end_obs,
            "pending_static_defense",
        ),
        "forge_delta": _delta(start_obs, end_obs, "ready_forge"),
        "pending_forge_delta": _delta(start_obs, end_obs, "pending_forge"),
        "upgrade_level_delta": (
            _delta(start_obs, end_obs, "ground_weapon_level")
            + _delta(start_obs, end_obs, "ground_armor_level")
        ),
        "base_count_delta": _delta(start_obs, end_obs, "own_bases"),
        "pending_nexus_delta": _delta(start_obs, end_obs, "pending_bases"),
        "worker_delta": _delta(start_obs, end_obs, "workers"),
        "worker_saturation_delta": _delta(
            start_obs,
            end_obs,
            "worker_saturation_ratio",
        ),
        "minerals_after": _value(end_obs, "minerals"),
        "vespene_after": _value(end_obs, "vespene"),
        "gateway_idle_after": _value(end_obs, "gateway_idle_count"),
        "robo_idle_after": _value(end_obs, "robo_idle_count"),
        "base_under_threat_after": _value(end_obs, "base_under_threat"),
        "worker_saturation_after": _value(end_obs, "worker_saturation_ratio"),
    }

    start_threat = _value(start_obs, "base_under_threat") > 0.0
    end_threat = _value(end_obs, "base_under_threat") > 0.0
    events = {
        "pending_gateway_seen": _any_future(future, "pending_gateways"),
        "ready_gateway_increased": _any_future_gt(future, "ready_gateways", start_obs),
        "pending_robo_seen": _any_future(future, "pending_robo"),
        "ready_robo_seen": _any_future(future, "ready_robo"),
        "observer_increased": _any_future_gt(future, "observers", start_obs),
        "immortal_increased": _any_future_gt(future, "immortals", start_obs),
        "robo_unit_produced": any(
            _value(row.observation, "observers") > _value(start_obs, "observers")
            or _value(row.observation, "immortals") > _value(start_obs, "immortals")
            for row in future
        ),
        "army_count_increased": _any_future_gt(future, "army_count", start_obs),
        "static_defense_increased": _any_future_gt(
            future,
            "ready_static_defense",
            start_obs,
        ),
        "base_under_threat_after": end_threat,
        "threat_persisted": start_threat and end_threat,
        "threat_cleared": start_threat and not end_threat,
        "forge_pending_seen": _any_future(future, "pending_forge"),
        "forge_ready_seen": _any_future(future, "ready_forge"),
        "upgrade_pending_seen": (
            _any_future(future, "ground_weapon_upgrade_pending")
            or _any_future(future, "ground_armor_upgrade_pending")
        ),
        "upgrade_level_increased": (
            _any_future_gt(future, "ground_weapon_level", start_obs)
            or _any_future_gt(future, "ground_armor_level", start_obs)
        ),
        "pending_nexus_seen": _any_future(future, "pending_bases"),
        "base_count_increased": _any_future_gt(future, "own_bases", start_obs),
        "worker_count_increased": _any_future_gt(future, "workers", start_obs),
    }
    event_times = {
        "first_pending_gateway_after_action": _first_delay(
            start,
            future,
            lambda row: _value(row.observation, "pending_gateways") > 0.0,
        ),
        "first_ready_gateway_delta_time": _first_delay(
            start,
            future,
            lambda row: _value(row.observation, "ready_gateways")
            > _value(start_obs, "ready_gateways"),
        ),
        "first_pending_robo_after_action": _first_delay(
            start,
            future,
            lambda row: _value(row.observation, "pending_robo") > 0.0,
        ),
        "first_ready_robo_after_action": _first_delay(
            start,
            future,
            lambda row: _value(row.observation, "ready_robo") > 0.0,
        ),
        "first_observer_after_action": _first_delay(
            start,
            future,
            lambda row: _value(row.observation, "observers")
            > _value(start_obs, "observers"),
        ),
        "first_immortal_after_action": _first_delay(
            start,
            future,
            lambda row: _value(row.observation, "immortals")
            > _value(start_obs, "immortals"),
        ),
        "first_static_defense_delta_time": _first_delay(
            start,
            future,
            lambda row: _value(row.observation, "ready_static_defense")
            > _value(start_obs, "ready_static_defense"),
        ),
        "first_threat_clear_time": _first_delay(
            start,
            future,
            lambda row: start_threat and _value(row.observation, "base_under_threat") <= 0.0,
        ),
    }
    return _OutcomeSample(
        metrics=metrics,
        events=events,
        event_times={
            name: value
            for name, value in event_times.items()
            if value is not None
        },
    )


def _iter_valid_strategy_rows(path: Path, *, source: str) -> Iterable[_StrategyOutcomeRow]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            observation = _row_observation(row)
            if not isinstance(observation, dict):
                continue
            try:
                validate_strategy_observation_dict(
                    observation,
                    allow_missing_defaults=True,
                )
            except ValueError:
                continue
            normalized = normalize_strategy_observation_dict(
                observation,
                allow_missing_defaults=True,
            )
            action = _valid_action(row.get("strategy_action"))
            if action is None:
                continue
            step = _optional_int(row.get("step"), default=line_number)
            yield _StrategyOutcomeRow(
                path=path,
                source=source,
                step=step,
                game_time=float(normalized.get("game_time", 0.0)),
                action_id=action,
                action_name=_strategy_action_name(
                    row,
                    id_key="strategy_action",
                    name_key="strategy_action_name",
                )
                or STRATEGY_ACTION_NAMES[action],
                done=bool(row.get("done", False)),
                result=_optional_str(row.get("result")),
                observation=normalized,
                observation_details=_row_observation_details(row),
                map_name=str(row.get("map_name", "")),
                difficulty=str(row.get("difficulty", "")),
                opponent_race=str(row.get("opponent_race", "")),
                opponent_ai_build=str(row.get("opponent_ai_build", "RandomBuild")),
                tactic_id=_optional_str(row.get("tactic_id")),
                before_action=_strategy_action_name(
                    row,
                    id_key="strategy_action_before_tactic_filter",
                    name_key="strategy_action_before_tactic_filter_name",
                ),
                after_action=_strategy_action_name(
                    row,
                    id_key="strategy_action_after_tactic_filter",
                    name_key="strategy_action_after_tactic_filter_name",
                ),
                execution_attempted=_optional_bool(
                    row.get("strategy_execution_attempted")
                ),
                execution_effect=_optional_str(row.get("strategy_execution_effect")),
                execution_blocker=_optional_str(row.get("strategy_execution_blocker")),
                execution_unit_type=_optional_str(
                    row.get("strategy_execution_unit_type")
                ),
                execution_target=_optional_str(row.get("strategy_execution_target")),
            )


def _summarize_file(
    path: Path,
    source: str,
    rows: list[_StrategyOutcomeRow],
    training_rows: list[_StrategyOutcomeRow],
) -> FileOutcomeSummary:
    action_counts = Counter(row.action_id for row in training_rows)
    action_first_times: dict[str, float] = {}
    execution_effects: Counter[str] = Counter()
    execution_blockers: Counter[str] = Counter()
    filter_change_rows = 0
    for row in training_rows:
        action_first_times.setdefault(row.action_name, row.game_time)
        if row.execution_effect:
            execution_effects[row.execution_effect] += 1
        if row.execution_blocker:
            execution_blockers[row.execution_blocker] += 1
        if _filter_key(row) is not None:
            filter_change_rows += 1
    first = rows[0] if rows else None
    last = rows[-1] if rows else None
    robo_payoff = _summarize_robo_payoff(training_rows)
    return FileOutcomeSummary(
        path=str(path),
        source=source,
        map_name=_first_non_empty(rows, "map_name"),
        difficulty=_first_non_empty(rows, "difficulty"),
        opponent_race=_first_non_empty(rows, "opponent_race"),
        opponent_ai_build=_first_non_empty(rows, "opponent_ai_build", default="RandomBuild"),
        result=next((row.result for row in rows if row.done and row.result), None),
        rows=len(rows),
        training_rows=len(training_rows),
        first_game_time=first.game_time if first is not None else None,
        last_game_time=last.game_time if last is not None else None,
        action_counts_by_name=_counts_by_name(action_counts),
        action_first_game_time_by_name=dict(sorted(action_first_times.items())),
        filter_change_rows=filter_change_rows,
        ready_robo_first_game_time=robo_payoff.ready_robo_first_game_time,
        observer_first_game_time=robo_payoff.observer_first_game_time,
        immortal_first_game_time=robo_payoff.immortal_first_game_time,
        base_threat_rows=sum(
            1 for row in training_rows if _value(row.observation, "base_under_threat") > 0.0
        ),
        execution_effect_counts=dict(sorted(execution_effects.items())),
        execution_blocker_counts=dict(sorted(execution_blockers.items())),
        robo_payoff=robo_payoff,
    )


@dataclass
class _SourceAccumulator:
    source: str
    files: int = 0
    rows: int = 0
    training_rows: int = 0
    result_counts: Counter[str] = None  # type: ignore[assignment]
    action_counts: Counter[int] = None  # type: ignore[assignment]
    action_first_times: dict[int, float] = None  # type: ignore[assignment]
    execution_effect_counts: Counter[str] = None  # type: ignore[assignment]
    execution_blocker_counts: Counter[str] = None  # type: ignore[assignment]
    filter_change_rows: int = 0

    def __post_init__(self) -> None:
        if self.result_counts is None:
            self.result_counts = Counter()
        if self.action_counts is None:
            self.action_counts = Counter()
        if self.action_first_times is None:
            self.action_first_times = {}
        if self.execution_effect_counts is None:
            self.execution_effect_counts = Counter()
        if self.execution_blocker_counts is None:
            self.execution_blocker_counts = Counter()

    def to_summary(self) -> SourceOutcomeSummary:
        return SourceOutcomeSummary(
            source=self.source,
            files=self.files,
            rows=self.rows,
            training_rows=self.training_rows,
            result_counts=dict(sorted(self.result_counts.items())),
            action_counts_by_name=_counts_by_name(self.action_counts),
            action_first_game_time_by_name={
                STRATEGY_ACTION_NAMES[action_id]: time
                for action_id, time in sorted(self.action_first_times.items())
                if action_id in STRATEGY_ACTION_NAMES
            },
            filter_change_rows=self.filter_change_rows,
            execution_effect_counts=dict(sorted(self.execution_effect_counts.items())),
            execution_blocker_counts=dict(sorted(self.execution_blocker_counts.items())),
        )

    def record_action_time(self, action_id: int, game_time: float) -> None:
        current = self.action_first_times.get(action_id)
        if current is None or game_time < current:
            self.action_first_times[action_id] = game_time


def _filter_summaries(
    counts: Counter[tuple[str, str, str, str]],
    early_counts: Counter[tuple[str, str, str, str]],
    window_acc: dict[tuple[str, str, str, str], dict[str, _OutcomeAccumulator]],
) -> list[FilterChangeOutcomeSummary]:
    summaries = [
        FilterChangeOutcomeSummary(
            opponent_ai_build=build,
            tactic_id=tactic_id,
            before_action=before_action,
            after_action=after_action,
            count=int(count),
            early_before_240_count=int(early_counts[key]),
            outcomes_by_window={
                window: accumulator.summary(_seconds_from_window_key(window))
                for window, accumulator in window_acc[key].items()
            },
        )
        for key, count in counts.items()
        for build, tactic_id, before_action, after_action in (key,)
    ]
    return sorted(
        summaries,
        key=lambda item: (
            -item.count,
            item.opponent_ai_build,
            item.tactic_id,
            item.before_action,
            item.after_action,
        ),
    )


def _action_summary(action_id: int, times: list[float]) -> ActionOutcomeSummary:
    name = STRATEGY_ACTION_NAMES[action_id]
    return ActionOutcomeSummary(
        action_name=name,
        count=len(times),
        first_game_time=min(times) if times else None,
        min_game_time=min(times) if times else None,
        max_game_time=max(times) if times else None,
        avg_game_time=(sum(times) / len(times)) if times else None,
        early_before_240_count=sum(
            1 for time in times if time < EARLY_GATEWAY_CUTOFF_SECONDS
        ),
    )


def _filter_key(row: _StrategyOutcomeRow) -> tuple[str, str, str, str] | None:
    if row.before_action is None or row.after_action is None:
        return None
    if row.before_action == row.after_action:
        return None
    return (
        row.opponent_ai_build or "RandomBuild",
        row.tactic_id or "<none>",
        row.before_action,
        row.after_action,
    )


def _source_for_file(path: Path, input_paths: tuple[str | Path, ...]) -> str:
    resolved_file = path.resolve()
    for raw_input in input_paths:
        input_path = Path(raw_input)
        try:
            resolved_input = input_path.resolve()
        except OSError:
            resolved_input = input_path
        if input_path.is_file() and resolved_file == resolved_input:
            return str(raw_input)
        if input_path.is_dir():
            try:
                if resolved_file.is_relative_to(resolved_input):
                    return str(raw_input)
            except ValueError:
                continue
    return str(path.parent)


def _any_future(rows: list[_StrategyOutcomeRow], field: str) -> bool:
    return any(_value(row.observation, field) > 0.0 for row in rows)


def _any_future_gt(
    rows: list[_StrategyOutcomeRow],
    field: str,
    start_observation: dict[str, float],
) -> bool:
    start_value = _value(start_observation, field)
    return any(_value(row.observation, field) > start_value for row in rows)


def _first_delay(
    start: _StrategyOutcomeRow,
    rows: list[_StrategyOutcomeRow],
    predicate: Any,
) -> float | None:
    for row in rows:
        if predicate(row):
            return float(row.game_time - start.game_time)
    return None


def _first_field_time(rows: list[_StrategyOutcomeRow], field: str) -> float | None:
    for row in rows:
        if _value(row.observation, field) > 0.0:
            return row.game_time
    return None


def _summarize_robo_payoff(rows: list[_StrategyOutcomeRow]) -> RoboPayoffSummary:
    ready_time = _first_field_time(rows, "ready_robo")
    observer_first = _first_field_time(rows, "observers")
    immortal_first = _first_field_time(rows, "immortals")
    if ready_time is None:
        return RoboPayoffSummary(
            ready_robo_first_game_time=None,
            observer_first_game_time=observer_first,
            immortal_first_game_time=immortal_first,
            observer_after_ready_game_time=None,
            immortal_after_ready_game_time=None,
            observer_after_ready_delay_seconds=None,
            immortal_after_ready_delay_seconds=None,
            observer_status="no_ready_robo",
            observer_blocker="no_ready_robo",
            immortal_status="no_ready_robo",
            immortal_blocker="no_ready_robo",
            robo_action_rows_after_ready=0,
            robo_idle_rows_after_ready=0,
            observer_candidate_rows_after_ready=0,
            immortal_candidate_rows_after_ready=0,
            immortal_affordable_candidate_rows_after_ready=0,
            immortal_mineral_blocked_candidate_rows=0,
            immortal_vespene_blocked_candidate_rows=0,
            immortal_supply_blocked_candidate_rows=0,
        )

    rows_after_ready = [row for row in rows if row.game_time >= ready_time]
    observer_after_ready = _first_field_time_after(rows_after_ready, "observers")
    immortal_after_ready = _first_field_time_after(rows_after_ready, "immortals")
    action_rows = [row for row in rows_after_ready if _is_robo_payoff_action(row)]
    idle_rows = [row for row in rows_after_ready if _row_has_idle_robo(row)]
    observer_candidates = [row for row in action_rows if _row_has_idle_robo(row)]
    immortal_candidates = observer_candidates
    immortal_affordable = [
        row
        for row in immortal_candidates
        if _can_afford_robo_unit(
            row,
            minerals=IMMORTAL_MINERALS,
            vespene=IMMORTAL_VESPENE,
            supply=IMMORTAL_SUPPLY,
        )
    ]

    observer_status, observer_blocker = _unit_payoff_status(
        produced_after_ready=observer_after_ready is not None,
        action_rows=action_rows,
        idle_rows=idle_rows,
        candidate_rows=observer_candidates,
        affordable_candidate_rows=[
            row
            for row in observer_candidates
            if _can_afford_robo_unit(
                row,
                minerals=OBSERVER_MINERALS,
                vespene=OBSERVER_VESPENE,
                supply=OBSERVER_SUPPLY,
            )
        ],
    )
    immortal_status, immortal_blocker = _unit_payoff_status(
        produced_after_ready=immortal_after_ready is not None,
        action_rows=action_rows,
        idle_rows=idle_rows,
        candidate_rows=immortal_candidates,
        affordable_candidate_rows=immortal_affordable,
    )

    return RoboPayoffSummary(
        ready_robo_first_game_time=ready_time,
        observer_first_game_time=observer_first,
        immortal_first_game_time=immortal_first,
        observer_after_ready_game_time=observer_after_ready,
        immortal_after_ready_game_time=immortal_after_ready,
        observer_after_ready_delay_seconds=_delay_from(ready_time, observer_after_ready),
        immortal_after_ready_delay_seconds=_delay_from(ready_time, immortal_after_ready),
        observer_status=observer_status,
        observer_blocker=observer_blocker,
        immortal_status=immortal_status,
        immortal_blocker=immortal_blocker,
        robo_action_rows_after_ready=len(action_rows),
        robo_idle_rows_after_ready=len(idle_rows),
        observer_candidate_rows_after_ready=len(observer_candidates),
        immortal_candidate_rows_after_ready=len(immortal_candidates),
        immortal_affordable_candidate_rows_after_ready=len(immortal_affordable),
        immortal_mineral_blocked_candidate_rows=sum(
            1
            for row in immortal_candidates
            if _value(row.observation, "minerals") < IMMORTAL_MINERALS
        ),
        immortal_vespene_blocked_candidate_rows=sum(
            1
            for row in immortal_candidates
            if _value(row.observation, "vespene") < IMMORTAL_VESPENE
        ),
        immortal_supply_blocked_candidate_rows=sum(
            1
            for row in immortal_candidates
            if _value(row.observation, "supply_left") < IMMORTAL_SUPPLY
        ),
    )


def _first_field_time_after(
    rows: list[_StrategyOutcomeRow],
    field: str,
) -> float | None:
    for row in rows:
        if _value(row.observation, field) > 0.0:
            return row.game_time
    return None


def _unit_payoff_status(
    *,
    produced_after_ready: bool,
    action_rows: list[_StrategyOutcomeRow],
    idle_rows: list[_StrategyOutcomeRow],
    candidate_rows: list[_StrategyOutcomeRow],
    affordable_candidate_rows: list[_StrategyOutcomeRow],
) -> tuple[str, str]:
    if produced_after_ready:
        return "produced_after_ready", "none"
    if not action_rows:
        return "blocked", "action_not_triggered"
    if not candidate_rows:
        if not idle_rows:
            return "blocked", "robo_not_idle"
        return "blocked", "action_not_triggered_while_idle"
    if not affordable_candidate_rows:
        return "blocked", "resource_or_supply_blocked"
    return "blocked", "not_produced_after_affordable_action"


def _is_robo_payoff_action(row: _StrategyOutcomeRow) -> bool:
    return row.action_name in ROBO_PAYOFF_ACTIONS


def _row_has_idle_robo(row: _StrategyOutcomeRow) -> bool:
    return (
        _value(row.observation, "ready_robo") > 0.0
        and _value(row.observation, "robo_idle_count") > 0.0
    )


def _can_afford_robo_unit(
    row: _StrategyOutcomeRow,
    *,
    minerals: float,
    vespene: float,
    supply: float,
) -> bool:
    return (
        _value(row.observation, "minerals") >= minerals
        and _value(row.observation, "vespene") >= vespene
        and _value(row.observation, "supply_left") >= supply
    )


def _delay_from(start_time: float, end_time: float | None) -> float | None:
    if end_time is None:
        return None
    return float(end_time - start_time)


def _delta(
    start_observation: dict[str, float],
    end_observation: dict[str, float],
    field: str,
) -> float:
    return _value(end_observation, field) - _value(start_observation, field)


def _counts_by_name(counts: Counter[int] | dict[int, int]) -> dict[str, int]:
    return {
        STRATEGY_ACTION_NAMES[action_id]: int(count)
        for action_id, count in sorted(counts.items())
        if action_id in STRATEGY_ACTION_NAMES
    }


def _strategy_action_name(row: dict[str, Any], *, id_key: str, name_key: str) -> str | None:
    name = _optional_str(row.get(name_key))
    if name is not None:
        return name
    action_id = _optional_int(row.get(id_key), default=None)
    if action_id is None:
        return None
    return STRATEGY_ACTION_NAMES.get(action_id)


def _valid_action(value: Any) -> int | None:
    action = _optional_int(value, default=None)
    if action is None or action not in STRATEGY_ACTION_NAMES:
        return None
    return action


def _row_observation(row: dict[str, Any]) -> Any:
    if "strategy_observation" in row:
        return row["strategy_observation"]
    return row.get("observation")


def _row_observation_details(row: dict[str, Any]) -> dict[str, float]:
    details = row.get("strategy_observation_details")
    if not isinstance(details, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, value in details.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


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


def _optional_int(value: Any, *, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def _value(observation: dict[str, float], field: str) -> float:
    if field not in observation and field in STRATEGY_OBSERVATION_DEFAULTS:
        return float(STRATEGY_OBSERVATION_DEFAULTS[field])
    return float(observation.get(field, 0.0))


def _result_key(value: Any) -> str:
    if value is None or value == "":
        return "NO_RESULT"
    return str(value)


def _seconds_from_window_key(key: str) -> float:
    if key.endswith("s"):
        return float(key[:-1])
    return float(key)
