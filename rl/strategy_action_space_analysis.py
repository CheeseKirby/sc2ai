"""Diagnose executable strategy action-space coverage from trajectories."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path

from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_datasets import (
    StrategyTrajectoryPathInput,
    discover_strategy_trajectory_files,
)
from rl.strategy_outcome_diagnostics import (
    _iter_valid_strategy_rows,
    _source_for_file,
)
from rl.strategy_replay_candidate import (
    candidate_executability,
    classify_replay_context,
    classify_threat_state,
)
from rl.strategy_signal_dataset import (
    StrategySignalRecord,
    build_strategy_signal_dataset,
)


ACTION_SPACE_START_METRICS: tuple[str, ...] = (
    "minerals",
    "vespene",
    "supply_left",
    "workers",
    "own_bases",
    "ready_gateways",
    "gateway_idle_count",
    "ready_robo",
    "robo_idle_count",
    "ready_static_defense",
    "pending_static_defense",
    "army_count",
    "base_under_air_threat",
    "base_under_ground_threat",
)


@dataclass(frozen=True)
class StrategyActionSpaceExample:
    """One representative row where executable strategy choices are narrow."""

    source_path: str
    source: str
    step: int
    game_time: float
    recorded_action: str
    recorded_training_use: str
    recorded_label_quality: str
    context: str
    threat_state: str
    executable_actions: list[str]
    blocked_actions: dict[str, str]
    reasons: list[str]
    start_metrics: dict[str, float]
    last_window: str
    last_window_negative_events: list[str]


@dataclass(frozen=True)
class StrategyActionSpaceAnalysis:
    """Aggregated strategy action-space coverage diagnostics."""

    inputs: list[str]
    files: int
    rows: int
    training_rows: int
    signal_records: int
    action_names: list[str]
    executable_action_count: dict[str, int]
    executable_action_sets: dict[str, int]
    only_stay_course_rows: int
    only_stay_course_ratio: float
    only_stay_course_under_threat_rows: int
    only_stay_course_under_threat_ratio: float
    only_stay_course_veto_negative_rows: int
    only_stay_course_veto_negative_ratio: float
    only_stay_course_by_training_use: dict[str, int]
    only_stay_course_by_label_quality: dict[str, int]
    only_stay_course_by_threat_state: dict[str, int]
    only_stay_course_by_recorded_action: dict[str, int]
    only_stay_course_by_file: dict[str, int]
    only_stay_course_blockers_by_action: dict[str, dict[str, int]]
    threatened_only_stay_course_blockers_by_action: dict[str, dict[str, int]]
    veto_only_stay_course_blockers_by_action: dict[str, dict[str, int]]
    threatened_only_stay_start_metric_averages: dict[str, float]
    veto_only_stay_start_metric_averages: dict[str, float]
    examples: list[StrategyActionSpaceExample]


def analyze_strategy_action_space(
    paths: StrategyTrajectoryPathInput,
    *,
    max_examples: int = 12,
) -> StrategyActionSpaceAnalysis:
    """Return executable action-space coverage diagnostics for trajectory rows."""
    if max_examples < 0:
        raise ValueError("max_examples must be >= 0")

    input_paths = _input_paths(paths)
    input_strings = [str(path) for path in input_paths]
    files = discover_strategy_trajectory_files(paths)
    signal_dataset = build_strategy_signal_dataset(
        paths,
        include_before_filter_candidates=False,
    )
    signal_by_key = {
        _record_key(record.path, record.step, record.recorded_action): record
        for record in signal_dataset.records
        if record.candidate_source == "recorded"
    }

    rows: list[_ActionSpaceRow] = []
    training_rows = 0
    action_names = _action_names()
    for path in files:
        source = _source_for_file(path, input_paths)
        for row in _iter_valid_strategy_rows(path, source=source):
            if row.done:
                continue
            training_rows += 1
            signal = signal_by_key.get(_row_key(row.path, row.step, row.action_name))
            rows.append(
                _action_space_row(
                    path=str(row.path),
                    source=row.source,
                    step=row.step,
                    game_time=row.game_time,
                    recorded_action=row.action_name,
                    threat_state=classify_threat_state(row),
                    context=(
                        signal.context
                        if signal is not None
                        else classify_replay_context(row, row.action_name)
                    ),
                    signal=signal,
                    executable_and_blockers={
                        action: candidate_executability(row, action)
                        for action in action_names
                    },
                )
            )

    only_stay_rows = [
        row for row in rows if row.executable_actions == ("STAY_COURSE",)
    ]
    threatened_only_stay_rows = [
        row for row in only_stay_rows if row.threat_state != "no_threat"
    ]
    veto_only_stay_rows = [
        row
        for row in only_stay_rows
        if row.recorded_training_use == "veto_negative"
    ]

    return StrategyActionSpaceAnalysis(
        inputs=input_strings,
        files=len(files),
        rows=len(rows),
        training_rows=training_rows,
        signal_records=len(signal_dataset.records),
        action_names=action_names,
        executable_action_count=_count(str(len(row.executable_actions)) for row in rows),
        executable_action_sets=_count(_action_set_key(row.executable_actions) for row in rows),
        only_stay_course_rows=len(only_stay_rows),
        only_stay_course_ratio=_ratio(len(only_stay_rows), len(rows)),
        only_stay_course_under_threat_rows=len(threatened_only_stay_rows),
        only_stay_course_under_threat_ratio=_ratio(
            len(threatened_only_stay_rows),
            len(rows),
        ),
        only_stay_course_veto_negative_rows=len(veto_only_stay_rows),
        only_stay_course_veto_negative_ratio=_ratio(
            len(veto_only_stay_rows),
            len(only_stay_rows),
        ),
        only_stay_course_by_training_use=_count(
            row.recorded_training_use for row in only_stay_rows
        ),
        only_stay_course_by_label_quality=_count(
            row.recorded_label_quality for row in only_stay_rows
        ),
        only_stay_course_by_threat_state=_count(
            row.threat_state for row in only_stay_rows
        ),
        only_stay_course_by_recorded_action=_count(
            row.recorded_action for row in only_stay_rows
        ),
        only_stay_course_by_file=_count(
            Path(row.path).name for row in only_stay_rows
        ),
        only_stay_course_blockers_by_action=_blockers_by_action(only_stay_rows),
        threatened_only_stay_course_blockers_by_action=_blockers_by_action(
            threatened_only_stay_rows
        ),
        veto_only_stay_course_blockers_by_action=_blockers_by_action(
            veto_only_stay_rows
        ),
        threatened_only_stay_start_metric_averages=_metric_averages(
            threatened_only_stay_rows
        ),
        veto_only_stay_start_metric_averages=_metric_averages(veto_only_stay_rows),
        examples=_examples(only_stay_rows, max_examples=max_examples),
    )


@dataclass(frozen=True)
class _ActionSpaceRow:
    path: str
    source: str
    step: int
    game_time: float
    recorded_action: str
    recorded_training_use: str
    recorded_label_quality: str
    context: str
    threat_state: str
    executable_actions: tuple[str, ...]
    blocked_actions: dict[str, str]
    reasons: tuple[str, ...]
    start_metrics: dict[str, float]
    last_window: str
    last_window_negative_events: tuple[str, ...]


def _action_space_row(
    *,
    path: str,
    source: str,
    step: int,
    game_time: float,
    recorded_action: str,
    threat_state: str,
    context: str,
    signal: StrategySignalRecord | None,
    executable_and_blockers: dict[str, tuple[bool, str | None]],
) -> _ActionSpaceRow:
    executable_actions = tuple(
        action for action, (executable, _) in executable_and_blockers.items() if executable
    )
    blocked_actions = {
        action: str(blocker or "unknown")
        for action, (executable, blocker) in executable_and_blockers.items()
        if not executable
    }
    last_window = _last_window(signal.metrics_by_window) if signal is not None else ""
    return _ActionSpaceRow(
        path=path,
        source=source,
        step=step,
        game_time=game_time,
        recorded_action=recorded_action,
        recorded_training_use=(
            signal.recommended_training_use if signal is not None else "missing_signal"
        ),
        recorded_label_quality=signal.label_quality if signal is not None else "unknown",
        context=context,
        threat_state=threat_state,
        executable_actions=executable_actions,
        blocked_actions=blocked_actions,
        reasons=tuple(signal.reasons if signal is not None else ()),
        start_metrics=(
            {
                metric: float(signal.start_metrics.get(metric, 0.0))
                for metric in ACTION_SPACE_START_METRICS
            }
            if signal is not None
            else {metric: 0.0 for metric in ACTION_SPACE_START_METRICS}
        ),
        last_window=last_window,
        last_window_negative_events=(
            tuple(signal.negative_events_by_window.get(last_window, ()))
            if signal is not None
            else ()
        ),
    )


def _examples(
    rows: list[_ActionSpaceRow],
    *,
    max_examples: int,
) -> list[StrategyActionSpaceExample]:
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row.recorded_training_use == "veto_negative" else 1,
            0 if row.threat_state != "no_threat" else 1,
            Path(row.path).name,
            row.step,
        ),
    )
    return [
        StrategyActionSpaceExample(
            source_path=row.path,
            source=row.source,
            step=row.step,
            game_time=row.game_time,
            recorded_action=row.recorded_action,
            recorded_training_use=row.recorded_training_use,
            recorded_label_quality=row.recorded_label_quality,
            context=row.context,
            threat_state=row.threat_state,
            executable_actions=list(row.executable_actions),
            blocked_actions=dict(row.blocked_actions),
            reasons=list(row.reasons),
            start_metrics=dict(row.start_metrics),
            last_window=row.last_window,
            last_window_negative_events=list(row.last_window_negative_events),
        )
        for row in ranked[:max_examples]
    ]


def _blockers_by_action(rows: list[_ActionSpaceRow]) -> dict[str, dict[str, int]]:
    by_action: dict[str, Counter[str]] = {
        action: Counter() for action in _action_names() if action != "STAY_COURSE"
    }
    for row in rows:
        for action, blocker in row.blocked_actions.items():
            if action == "STAY_COURSE":
                continue
            by_action[action][blocker] += 1
    return {
        action: _sorted_counter(counter)
        for action, counter in by_action.items()
        if counter
    }


def _metric_averages(rows: list[_ActionSpaceRow]) -> dict[str, float]:
    if not rows:
        return {}
    totals: Counter[str] = Counter()
    for row in rows:
        for metric in ACTION_SPACE_START_METRICS:
            totals[metric] += float(row.start_metrics.get(metric, 0.0))
    return {
        metric: float(totals[metric] / len(rows))
        for metric in ACTION_SPACE_START_METRICS
    }


def _action_names() -> list[str]:
    return [STRATEGY_ACTION_NAMES[action_id] for action_id in sorted(STRATEGY_ACTION_NAMES)]


def _row_key(path: str | Path, step: int, action_name: str) -> tuple[str, int, str]:
    return _record_key(path, step, action_name)


def _record_key(path: str | Path, step: int, action_name: str) -> tuple[str, int, str]:
    return (str(Path(path).resolve()), int(step), str(action_name))


def _last_window(metrics_by_window: dict[str, dict[str, float]]) -> str:
    if not metrics_by_window:
        return ""
    return sorted(metrics_by_window, key=_window_seconds)[-1]


def _window_seconds(key: str) -> float:
    if key.endswith("s"):
        return float(key[:-1])
    if not key:
        return 0.0
    return float(key)


def _action_set_key(actions: tuple[str, ...]) -> str:
    if not actions:
        return "<none>"
    return "+".join(actions)


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _count(values: IterableABC[str]) -> dict[str, int]:
    return _sorted_counter(Counter(str(value) for value in values))


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in counter.items()))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
