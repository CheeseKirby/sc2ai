"""Analysis-only emergency strategy action hypotheses."""
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
    _StrategyOutcomeRow,
    _iter_valid_strategy_rows,
    _source_for_file,
    _value,
)
from rl.strategy_replay_candidate import candidate_executability, classify_threat_state
from rl.strategy_signal_dataset import (
    StrategySignalRecord,
    build_strategy_signal_dataset,
)


EMERGENCY_DEFEND_ACTION = "EMERGENCY_DEFEND"
EMERGENCY_ACTION_NAMES: tuple[str, ...] = (EMERGENCY_DEFEND_ACTION,)
GROUND_WORKER_DEFENSE_MIN_WORKERS = 8.0
GROUND_STATIC_SUPPORT_MIN_WORKERS = 4.0
EMERGENCY_START_METRICS: tuple[str, ...] = (
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
    "zealots",
    "stalkers",
    "sentries",
    "immortals",
    "army_count",
    "base_under_air_threat",
    "base_under_ground_threat",
)


@dataclass(frozen=True)
class StrategyEmergencyActionExample:
    """One threatened only-STAY_COURSE row with emergency hypothesis results."""

    source_path: str
    source: str
    step: int
    game_time: float
    recorded_action: str
    recorded_training_use: str
    recorded_label_quality: str
    threat_state: str
    standard_executable_actions: list[str]
    standard_blocked_actions: dict[str, str]
    emergency_actions: list[str]
    emergency_blockers: dict[str, str]
    air_defense_gap_reason: str
    reasons: list[str]
    start_metrics: dict[str, float]
    last_window: str
    last_window_negative_events: list[str]


@dataclass(frozen=True)
class StrategyEmergencyActionAnalysis:
    """Dataset-level coverage estimate for analysis-only emergency actions."""

    inputs: list[str]
    files: int
    rows: int
    training_rows: int
    signal_records: int
    emergency_action_names: list[str]
    observation_detail_rows: int
    observation_detail_ratio: float
    threatened_only_stay_course_rows: int
    threatened_only_stay_course_ratio: float
    threatened_only_stay_course_detail_rows: int
    threatened_only_stay_course_detail_ratio: float
    air_threat_only_stay_course_rows: int
    air_threat_only_stay_course_detail_rows: int
    air_threat_only_stay_course_detail_ratio: float
    action_space_exhausted_rows: int
    action_space_exhausted_ratio: float
    addressable_threatened_only_stay_course_rows: int
    addressable_threatened_only_stay_course_ratio: float
    addressable_action_space_exhausted_rows: int
    addressable_action_space_exhausted_ratio: float
    emergency_action_count: dict[str, int]
    emergency_action_sets: dict[str, int]
    addressable_by_training_use: dict[str, int]
    addressable_by_threat_state: dict[str, int]
    unaddressed_by_training_use: dict[str, int]
    unaddressed_by_threat_state: dict[str, int]
    emergency_blockers_by_action: dict[str, dict[str, int]]
    unaddressed_air_defense_gap_by_reason: dict[str, int]
    addressable_start_metric_averages: dict[str, float]
    unaddressed_start_metric_averages: dict[str, float]
    examples: list[StrategyEmergencyActionExample]


def analyze_strategy_emergency_actions(
    paths: StrategyTrajectoryPathInput,
    *,
    max_examples: int = 12,
) -> StrategyEmergencyActionAnalysis:
    """Estimate how much an emergency pseudo-action could widen threat states."""
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

    rows: list[_EmergencyActionRow] = []
    training_rows = 0
    standard_action_names = _standard_action_names()
    for path in files:
        source = _source_for_file(path, input_paths)
        for row in _iter_valid_strategy_rows(path, source=source):
            if row.done:
                continue
            training_rows += 1
            standard_results = {
                action: candidate_executability(row, action)
                for action in standard_action_names
            }
            standard_executable_actions = tuple(
                action
                for action, (executable, _) in standard_results.items()
                if executable
            )
            threat_state = classify_threat_state(row)
            signal = signal_by_key.get(_record_key(row.path, row.step, row.action_name))
            rows.append(
                _emergency_action_row(
                    row=row,
                    threat_state=threat_state,
                    signal=signal,
                    standard_executable_actions=standard_executable_actions,
                    standard_blocked_actions={
                        action: str(blocker or "unknown")
                        for action, (executable, blocker) in standard_results.items()
                        if not executable
                    },
                    emergency_results={
                        action: emergency_action_executability(row, action)
                        for action in EMERGENCY_ACTION_NAMES
                    },
                )
            )

    threatened_only_stay_rows = [
        row
        for row in rows
        if row.threat_state != "no_threat"
        and row.standard_executable_actions == ("STAY_COURSE",)
    ]
    action_space_exhausted_rows = [
        row
        for row in threatened_only_stay_rows
        if row.recorded_training_use == "action_space_exhausted"
    ]
    addressable_rows = [
        row for row in threatened_only_stay_rows if row.emergency_actions
    ]
    unaddressed_rows = [
        row for row in threatened_only_stay_rows if not row.emergency_actions
    ]
    air_threat_only_stay_rows = [
        row
        for row in threatened_only_stay_rows
        if row.threat_state in {"air_threat", "air_and_ground_threat"}
    ]
    rows_with_details = [row for row in rows if row.has_observation_details]
    threatened_only_stay_rows_with_details = [
        row for row in threatened_only_stay_rows if row.has_observation_details
    ]
    air_threat_only_stay_rows_with_details = [
        row for row in air_threat_only_stay_rows if row.has_observation_details
    ]
    addressable_action_space_exhausted_rows = [
        row
        for row in action_space_exhausted_rows
        if row.emergency_actions
    ]

    return StrategyEmergencyActionAnalysis(
        inputs=input_strings,
        files=len(files),
        rows=len(rows),
        training_rows=training_rows,
        signal_records=len(signal_dataset.records),
        emergency_action_names=list(EMERGENCY_ACTION_NAMES),
        observation_detail_rows=len(rows_with_details),
        observation_detail_ratio=_ratio(len(rows_with_details), len(rows)),
        threatened_only_stay_course_rows=len(threatened_only_stay_rows),
        threatened_only_stay_course_ratio=_ratio(
            len(threatened_only_stay_rows),
            len(rows),
        ),
        threatened_only_stay_course_detail_rows=len(
            threatened_only_stay_rows_with_details
        ),
        threatened_only_stay_course_detail_ratio=_ratio(
            len(threatened_only_stay_rows_with_details),
            len(threatened_only_stay_rows),
        ),
        air_threat_only_stay_course_rows=len(air_threat_only_stay_rows),
        air_threat_only_stay_course_detail_rows=len(
            air_threat_only_stay_rows_with_details
        ),
        air_threat_only_stay_course_detail_ratio=_ratio(
            len(air_threat_only_stay_rows_with_details),
            len(air_threat_only_stay_rows),
        ),
        action_space_exhausted_rows=len(action_space_exhausted_rows),
        action_space_exhausted_ratio=_ratio(
            len(action_space_exhausted_rows),
            len(threatened_only_stay_rows),
        ),
        addressable_threatened_only_stay_course_rows=len(addressable_rows),
        addressable_threatened_only_stay_course_ratio=_ratio(
            len(addressable_rows),
            len(threatened_only_stay_rows),
        ),
        addressable_action_space_exhausted_rows=len(
            addressable_action_space_exhausted_rows
        ),
        addressable_action_space_exhausted_ratio=_ratio(
            len(addressable_action_space_exhausted_rows),
            len(action_space_exhausted_rows),
        ),
        emergency_action_count=_count(
            str(len(row.emergency_actions)) for row in threatened_only_stay_rows
        ),
        emergency_action_sets=_count(
            _action_set_key(row.emergency_actions) for row in threatened_only_stay_rows
        ),
        addressable_by_training_use=_count(
            row.recorded_training_use for row in addressable_rows
        ),
        addressable_by_threat_state=_count(row.threat_state for row in addressable_rows),
        unaddressed_by_training_use=_count(
            row.recorded_training_use for row in unaddressed_rows
        ),
        unaddressed_by_threat_state=_count(row.threat_state for row in unaddressed_rows),
        emergency_blockers_by_action=_emergency_blockers_by_action(unaddressed_rows),
        unaddressed_air_defense_gap_by_reason=_air_defense_gap_by_reason(
            unaddressed_rows
        ),
        addressable_start_metric_averages=_metric_averages(addressable_rows),
        unaddressed_start_metric_averages=_metric_averages(unaddressed_rows),
        examples=_examples(threatened_only_stay_rows, max_examples=max_examples),
    )


def emergency_action_executability(
    row: _StrategyOutcomeRow,
    emergency_action: str,
) -> tuple[bool, str | None]:
    """Return whether an analysis-only emergency action is plausible."""
    if emergency_action != EMERGENCY_DEFEND_ACTION:
        return False, "unknown_emergency_action"
    return _emergency_defend_executability(row)


@dataclass(frozen=True)
class _EmergencyActionRow:
    path: str
    source: str
    step: int
    game_time: float
    recorded_action: str
    recorded_training_use: str
    recorded_label_quality: str
    threat_state: str
    standard_executable_actions: tuple[str, ...]
    standard_blocked_actions: dict[str, str]
    emergency_actions: tuple[str, ...]
    emergency_blockers: dict[str, str]
    air_defense_gap_reason: str
    has_observation_details: bool
    reasons: tuple[str, ...]
    start_metrics: dict[str, float]
    last_window: str
    last_window_negative_events: tuple[str, ...]


def _emergency_action_row(
    *,
    row: _StrategyOutcomeRow,
    threat_state: str,
    signal: StrategySignalRecord | None,
    standard_executable_actions: tuple[str, ...],
    standard_blocked_actions: dict[str, str],
    emergency_results: dict[str, tuple[bool, str | None]],
) -> _EmergencyActionRow:
    emergency_actions = tuple(
        action for action, (executable, _) in emergency_results.items() if executable
    )
    emergency_blockers = {
        action: str(blocker or "unknown")
        for action, (executable, blocker) in emergency_results.items()
        if not executable
    }
    last_window = _last_window(signal.metrics_by_window) if signal is not None else ""
    return _EmergencyActionRow(
        path=str(row.path),
        source=row.source,
        step=row.step,
        game_time=row.game_time,
        recorded_action=row.action_name,
        recorded_training_use=(
            signal.recommended_training_use if signal is not None else "missing_signal"
        ),
        recorded_label_quality=signal.label_quality if signal is not None else "unknown",
        threat_state=threat_state,
        standard_executable_actions=standard_executable_actions,
        standard_blocked_actions=standard_blocked_actions,
        emergency_actions=emergency_actions,
        emergency_blockers=emergency_blockers,
        air_defense_gap_reason=_air_defense_gap_reason(row),
        has_observation_details=bool(row.observation_details),
        reasons=tuple(signal.reasons if signal is not None else ()),
        start_metrics={
            metric: float(row.observation.get(metric, 0.0))
            for metric in EMERGENCY_START_METRICS
        },
        last_window=last_window,
        last_window_negative_events=(
            tuple(signal.negative_events_by_window.get(last_window, ()))
            if signal is not None
            else ()
        ),
    )


def _emergency_defend_executability(
    row: _StrategyOutcomeRow,
) -> tuple[bool, str | None]:
    threat_state = classify_threat_state(row)
    if threat_state == "no_threat":
        return False, "no_active_threat"

    air_threat = _value(row.observation, "base_under_air_threat") > 0.0
    ground_threat = _value(row.observation, "base_under_ground_threat") > 0.0
    if air_threat and not _has_air_defense_assets(row):
        return False, "no_air_defense_assets"
    if ground_threat and not _has_ground_defense_assets(row):
        return False, "no_ground_defense_assets"
    if not air_threat and not ground_threat and not _has_any_defense_assets(row):
        return False, "no_defense_assets"
    return True, None


def _has_air_defense_assets(row: _StrategyOutcomeRow) -> bool:
    if row.observation_details:
        return (
            _detail_value(row, "ready_photon_cannons") > 0.0
            or _value(row.observation, "stalkers") > 0.0
            or _value(row.observation, "sentries") > 0.0
        )
    return (
        _value(row.observation, "stalkers") > 0.0
        or _value(row.observation, "sentries") > 0.0
    )


def _has_ground_defense_assets(row: _StrategyOutcomeRow) -> bool:
    return (
        _value(row.observation, "army_count") > 0.0
        or _value(row.observation, "workers") >= GROUND_WORKER_DEFENSE_MIN_WORKERS
        or (
            _value(row.observation, "ready_static_defense") > 0.0
            and _value(row.observation, "workers") >= GROUND_STATIC_SUPPORT_MIN_WORKERS
        )
    )


def _has_any_defense_assets(row: _StrategyOutcomeRow) -> bool:
    return _has_air_defense_assets(row) or _has_ground_defense_assets(row)


def _air_defense_gap_reason(row: _StrategyOutcomeRow) -> str:
    if _value(row.observation, "base_under_air_threat") <= 0.0:
        return ""
    if _has_air_defense_assets(row):
        return "air_defense_assets_present"
    if row.observation_details:
        return "no_observed_anti_air_assets"
    if (
        _value(row.observation, "ready_static_defense") > 0.0
        or _value(row.observation, "pending_static_defense") > 0.0
    ):
        return "static_defense_type_ambiguous"
    return "no_observed_anti_air_assets"


def _detail_value(row: _StrategyOutcomeRow, field: str) -> float:
    return float(row.observation_details.get(field, 0.0))


def _examples(
    rows: list[_EmergencyActionRow],
    *,
    max_examples: int,
) -> list[StrategyEmergencyActionExample]:
    ranked = sorted(
        rows,
        key=lambda row: (
            0
            if (
                row.recorded_training_use == "action_space_exhausted"
                and not row.emergency_actions
            )
            else 1,
            0 if not row.emergency_actions else 1,
            Path(row.path).name,
            row.step,
        ),
    )
    return [
        StrategyEmergencyActionExample(
            source_path=row.path,
            source=row.source,
            step=row.step,
            game_time=row.game_time,
            recorded_action=row.recorded_action,
            recorded_training_use=row.recorded_training_use,
            recorded_label_quality=row.recorded_label_quality,
            threat_state=row.threat_state,
            standard_executable_actions=list(row.standard_executable_actions),
            standard_blocked_actions=dict(row.standard_blocked_actions),
            emergency_actions=list(row.emergency_actions),
            emergency_blockers=dict(row.emergency_blockers),
            air_defense_gap_reason=row.air_defense_gap_reason,
            reasons=list(row.reasons),
            start_metrics=dict(row.start_metrics),
            last_window=row.last_window,
            last_window_negative_events=list(row.last_window_negative_events),
        )
        for row in ranked[:max_examples]
    ]


def _air_defense_gap_by_reason(
    rows: list[_EmergencyActionRow],
) -> dict[str, int]:
    return _count(
        row.air_defense_gap_reason
        for row in rows
        if row.air_defense_gap_reason
        and row.air_defense_gap_reason != "air_defense_assets_present"
    )


def _emergency_blockers_by_action(
    rows: list[_EmergencyActionRow],
) -> dict[str, dict[str, int]]:
    by_action: dict[str, Counter[str]] = {
        action: Counter() for action in EMERGENCY_ACTION_NAMES
    }
    for row in rows:
        for action, blocker in row.emergency_blockers.items():
            by_action[action][blocker] += 1
    return {
        action: _sorted_counter(counter)
        for action, counter in by_action.items()
        if counter
    }


def _metric_averages(rows: list[_EmergencyActionRow]) -> dict[str, float]:
    if not rows:
        return {}
    totals: Counter[str] = Counter()
    for row in rows:
        for metric in EMERGENCY_START_METRICS:
            totals[metric] += float(row.start_metrics.get(metric, 0.0))
    return {
        metric: float(totals[metric] / len(rows))
        for metric in EMERGENCY_START_METRICS
    }


def _standard_action_names() -> list[str]:
    return [
        STRATEGY_ACTION_NAMES[action_id]
        for action_id in sorted(STRATEGY_ACTION_NAMES)
    ]


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
