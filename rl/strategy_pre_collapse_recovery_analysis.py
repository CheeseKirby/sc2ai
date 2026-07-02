"""Trace strategy gate failures back to pre-collapse recovery windows."""
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
from rl.strategy_signal_dataset import StrategySignalRecord, build_strategy_signal_dataset


TARGET_TRAINING_USES: tuple[str, ...] = ("veto_negative", "action_space_exhausted")
RECOVERY_ACTION_NAMES: tuple[str, ...] = (
    "TECH_ROBO",
    "PRODUCE_ARMY",
    "BUILD_STATIC_DEFENSE",
)
DEFAULT_LOOKBACK_SECONDS = 240.0
DEFAULT_MAX_MISSED_PRE_COLLAPSE_RECOVERY_ROWS = 0
DEFAULT_MAX_MISSED_PRE_COLLAPSE_RECOVERY_RATE = 0.0
PRE_COLLAPSE_START_METRICS: tuple[str, ...] = (
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
    "base_under_threat",
    "base_under_air_threat",
    "base_under_ground_threat",
)


@dataclass(frozen=True)
class StrategyPreCollapseRecoveryWindow:
    """One row before a collapse target where recovery may have been available."""

    source_path: str
    source: str
    target_step: int
    target_game_time: float
    step: int
    game_time: float
    seconds_before_target: float
    recorded_action: str
    threat_state: str
    executable_recovery_actions: list[str]
    selected_recovery_action: str | None
    selected_executable_recovery_action: str | None
    recovery_blockers: dict[str, str]
    start_metrics: dict[str, float | str]


@dataclass(frozen=True)
class StrategyPreCollapseFailure:
    """One veto/action-space failure row plus its pre-collapse recovery summary."""

    source_path: str
    source: str
    step: int
    game_time: float
    recorded_action: str
    recorded_training_use: str
    recorded_label_quality: str
    threat_state: str
    avoidability: str
    executable_actions: list[str]
    non_stay_executable_actions: list[str]
    executable_recovery_actions: list[str]
    recovery_blockers: dict[str, str]
    lookback_seconds: float
    pre_collapse_rows: int
    pre_collapse_recovery_window_rows: int
    pre_collapse_selected_recovery_rows: int
    pre_collapse_selected_executable_recovery_rows: int
    pre_collapse_recovery_executable_counts_by_action: dict[str, int]
    pre_collapse_recovery_selected_counts_by_action: dict[str, int]
    pre_collapse_recovery_selected_executable_counts_by_action: dict[str, int]
    missed_pre_collapse_recovery: bool
    last_executable_recovery_time: float | None
    last_executable_recovery_actions: list[str]
    last_selected_executable_recovery_time: float | None
    last_selected_executable_recovery_action: str | None
    start_metrics: dict[str, float | str]
    recovery_windows: list[StrategyPreCollapseRecoveryWindow]


@dataclass(frozen=True)
class StrategyPreCollapseFileSummary:
    """Per-file collapse/recovery summary."""

    path: str
    source: str
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    rows: int
    training_rows: int
    target_rows: int
    avoidability_counts: dict[str, int]
    target_training_use_counts: dict[str, int]
    rows_with_pre_collapse_recovery_window: int
    rows_with_pre_collapse_selected_executable_recovery: int
    missed_pre_collapse_recovery_rows: int


@dataclass(frozen=True)
class StrategyPreCollapseRecoveryAnalysis:
    """Dataset-level analysis of recovery opportunities before gate failures."""

    inputs: list[str]
    lookback_seconds: float
    target_training_uses: list[str]
    recovery_action_names: list[str]
    recommendation: str
    blocking_reasons: list[str]
    warnings: list[str]
    max_missed_pre_collapse_recovery_rows: int
    max_missed_pre_collapse_recovery_rate: float
    files: int
    rows: int
    training_rows: int
    target_rows: int
    missed_pre_collapse_recovery_rate: float
    avoidability_counts: dict[str, int]
    target_training_use_counts: dict[str, int]
    target_label_quality_counts: dict[str, int]
    target_threat_state_counts: dict[str, int]
    target_executable_action_sets: dict[str, int]
    rows_with_pre_collapse_recovery_window: int
    rows_with_pre_collapse_selected_recovery: int
    rows_with_pre_collapse_selected_executable_recovery: int
    missed_pre_collapse_recovery_rows: int
    no_pre_collapse_recovery_window_rows: int
    pre_collapse_recovery_executable_counts_by_action: dict[str, int]
    pre_collapse_recovery_selected_counts_by_action: dict[str, int]
    pre_collapse_recovery_selected_executable_counts_by_action: dict[str, int]
    recovery_blockers_at_target_by_action: dict[str, dict[str, int]]
    file_summaries: list[StrategyPreCollapseFileSummary]
    failures: list[StrategyPreCollapseFailure]


def analyze_strategy_pre_collapse_recovery(
    paths: StrategyTrajectoryPathInput,
    *,
    lookback_seconds: float = DEFAULT_LOOKBACK_SECONDS,
    max_failures: int = 20,
    max_windows_per_failure: int = 6,
    max_missed_pre_collapse_recovery_rows: int = (
        DEFAULT_MAX_MISSED_PRE_COLLAPSE_RECOVERY_ROWS
    ),
    max_missed_pre_collapse_recovery_rate: float = (
        DEFAULT_MAX_MISSED_PRE_COLLAPSE_RECOVERY_RATE
    ),
) -> StrategyPreCollapseRecoveryAnalysis:
    """Return pre-collapse recovery diagnostics for veto/action-space failure rows."""
    if lookback_seconds <= 0.0:
        raise ValueError("lookback_seconds must be > 0")
    if max_failures < 0:
        raise ValueError("max_failures must be >= 0")
    if max_windows_per_failure < 0:
        raise ValueError("max_windows_per_failure must be >= 0")
    if max_missed_pre_collapse_recovery_rows < 0:
        raise ValueError("max_missed_pre_collapse_recovery_rows must be >= 0")
    if not 0.0 <= max_missed_pre_collapse_recovery_rate <= 1.0:
        raise ValueError("max_missed_pre_collapse_recovery_rate must be in [0.0, 1.0]")

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

    rows_total = 0
    training_rows_total = 0
    target_rows_total = 0
    avoidability_counts: Counter[str] = Counter()
    target_training_use_counts: Counter[str] = Counter()
    target_label_quality_counts: Counter[str] = Counter()
    target_threat_state_counts: Counter[str] = Counter()
    target_action_sets: Counter[str] = Counter()
    rows_with_window = 0
    rows_with_selected = 0
    rows_with_selected_executable = 0
    missed_rows = 0
    no_window_rows = 0
    recovery_executable_counts: Counter[str] = Counter()
    recovery_selected_counts: Counter[str] = Counter()
    recovery_selected_executable_counts: Counter[str] = Counter()
    blockers_at_target: dict[str, Counter[str]] = {
        action: Counter() for action in RECOVERY_ACTION_NAMES
    }
    file_summaries: list[StrategyPreCollapseFileSummary] = []
    failures: list[StrategyPreCollapseFailure] = []

    for path in files:
        source = _source_for_file(path, input_paths)
        rows = list(_iter_valid_strategy_rows(path, source=source))
        training_rows = [row for row in rows if not row.done]
        rows_total += len(rows)
        training_rows_total += len(training_rows)

        file_failures: list[StrategyPreCollapseFailure] = []
        for index, row in enumerate(rows):
            if row.done:
                continue
            signal = signal_by_key.get(_row_key(row))
            if signal is None or signal.recommended_training_use not in TARGET_TRAINING_USES:
                continue

            failure = _failure_for_row(
                rows=rows,
                target_index=index,
                signal=signal,
                lookback_seconds=float(lookback_seconds),
                max_windows=max_windows_per_failure,
            )
            file_failures.append(failure)

            target_rows_total += 1
            avoidability_counts[failure.avoidability] += 1
            target_training_use_counts[failure.recorded_training_use] += 1
            target_label_quality_counts[failure.recorded_label_quality] += 1
            target_threat_state_counts[failure.threat_state] += 1
            target_action_sets[_action_set_key(failure.executable_actions)] += 1
            if failure.pre_collapse_recovery_window_rows:
                rows_with_window += 1
            else:
                no_window_rows += 1
            if failure.pre_collapse_selected_recovery_rows:
                rows_with_selected += 1
            if failure.pre_collapse_selected_executable_recovery_rows:
                rows_with_selected_executable += 1
            if failure.missed_pre_collapse_recovery:
                missed_rows += 1
            recovery_executable_counts.update(
                failure.pre_collapse_recovery_executable_counts_by_action
            )
            recovery_selected_counts.update(
                failure.pre_collapse_recovery_selected_counts_by_action
            )
            recovery_selected_executable_counts.update(
                failure.pre_collapse_recovery_selected_executable_counts_by_action
            )
            _merge_nested_counts(blockers_at_target, failure.recovery_blockers)

        if file_failures:
            first = file_failures[0]
            file_summaries.append(
                StrategyPreCollapseFileSummary(
                    path=str(path),
                    source=source,
                    map_name=first.start_metrics.get("map_name", ""),
                    difficulty=first.start_metrics.get("difficulty", ""),
                    opponent_race=first.start_metrics.get("opponent_race", ""),
                    opponent_ai_build=first.start_metrics.get("opponent_ai_build", ""),
                    rows=len(rows),
                    training_rows=len(training_rows),
                    target_rows=len(file_failures),
                    avoidability_counts=_count(
                        failure.avoidability for failure in file_failures
                    ),
                    target_training_use_counts=_count(
                        failure.recorded_training_use for failure in file_failures
                    ),
                    rows_with_pre_collapse_recovery_window=sum(
                        1
                        for failure in file_failures
                        if failure.pre_collapse_recovery_window_rows
                    ),
                    rows_with_pre_collapse_selected_executable_recovery=sum(
                        1
                        for failure in file_failures
                        if failure.pre_collapse_selected_executable_recovery_rows
                    ),
                    missed_pre_collapse_recovery_rows=sum(
                        1
                        for failure in file_failures
                        if failure.missed_pre_collapse_recovery
                    ),
                )
            )
        failures.extend(file_failures)

    failures = sorted(
        failures,
        key=lambda failure: (
            0 if failure.missed_pre_collapse_recovery else 1,
            failure.avoidability,
            Path(failure.source_path).name,
            failure.step,
        ),
    )[:max_failures]
    missed_rate = _ratio(missed_rows, target_rows_total)
    blocking_reasons = _blocking_reasons(
        missed_pre_collapse_recovery_rows=missed_rows,
        missed_pre_collapse_recovery_rate=missed_rate,
        max_missed_pre_collapse_recovery_rows=max_missed_pre_collapse_recovery_rows,
        max_missed_pre_collapse_recovery_rate=max_missed_pre_collapse_recovery_rate,
    )
    warnings = ["no_target_rows"] if target_rows_total == 0 else []

    return StrategyPreCollapseRecoveryAnalysis(
        inputs=input_strings,
        lookback_seconds=float(lookback_seconds),
        target_training_uses=list(TARGET_TRAINING_USES),
        recovery_action_names=list(RECOVERY_ACTION_NAMES),
        recommendation="ready" if not blocking_reasons else "hold",
        blocking_reasons=blocking_reasons,
        warnings=warnings,
        max_missed_pre_collapse_recovery_rows=int(
            max_missed_pre_collapse_recovery_rows
        ),
        max_missed_pre_collapse_recovery_rate=float(
            max_missed_pre_collapse_recovery_rate
        ),
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        target_rows=target_rows_total,
        missed_pre_collapse_recovery_rate=missed_rate,
        avoidability_counts=_sorted_counter(avoidability_counts),
        target_training_use_counts=_sorted_counter(target_training_use_counts),
        target_label_quality_counts=_sorted_counter(target_label_quality_counts),
        target_threat_state_counts=_sorted_counter(target_threat_state_counts),
        target_executable_action_sets=_sorted_counter(target_action_sets),
        rows_with_pre_collapse_recovery_window=rows_with_window,
        rows_with_pre_collapse_selected_recovery=rows_with_selected,
        rows_with_pre_collapse_selected_executable_recovery=rows_with_selected_executable,
        missed_pre_collapse_recovery_rows=missed_rows,
        no_pre_collapse_recovery_window_rows=no_window_rows,
        pre_collapse_recovery_executable_counts_by_action=_action_counts(
            recovery_executable_counts
        ),
        pre_collapse_recovery_selected_counts_by_action=_action_counts(
            recovery_selected_counts
        ),
        pre_collapse_recovery_selected_executable_counts_by_action=_action_counts(
            recovery_selected_executable_counts
        ),
        recovery_blockers_at_target_by_action=_nested_counts(blockers_at_target),
        file_summaries=file_summaries,
        failures=failures,
    )


def _failure_for_row(
    *,
    rows: list[_StrategyOutcomeRow],
    target_index: int,
    signal: StrategySignalRecord,
    lookback_seconds: float,
    max_windows: int,
) -> StrategyPreCollapseFailure:
    target = rows[target_index]
    executable_actions = _executable_actions(target)
    non_stay_actions = [action for action in executable_actions if action != "STAY_COURSE"]
    executable_recovery = _executable_recovery_actions(target)
    recovery_blockers = _recovery_blockers(target)
    avoidability = _avoidability(
        executable_actions=executable_actions,
        executable_recovery_actions=executable_recovery,
    )

    pre_rows = [
        row
        for row in rows[:target_index]
        if not row.done and 0.0 < target.game_time - row.game_time <= lookback_seconds
    ]
    recovery_windows_all = [
        _window_for_row(row, target=target)
        for row in pre_rows
        if _executable_recovery_actions(row)
    ]
    selected_windows = [
        window
        for window in recovery_windows_all
        if window.selected_recovery_action is not None
    ]
    selected_executable_windows = [
        window
        for window in recovery_windows_all
        if window.selected_executable_recovery_action is not None
    ]
    recovery_executable_counts: Counter[str] = Counter()
    recovery_selected_counts: Counter[str] = Counter()
    recovery_selected_executable_counts: Counter[str] = Counter()
    for window in recovery_windows_all:
        recovery_executable_counts.update(window.executable_recovery_actions)
        if window.selected_recovery_action is not None:
            recovery_selected_counts[window.selected_recovery_action] += 1
        if window.selected_executable_recovery_action is not None:
            recovery_selected_executable_counts[
                window.selected_executable_recovery_action
            ] += 1
    last_executable = recovery_windows_all[-1] if recovery_windows_all else None
    last_selected_executable = (
        selected_executable_windows[-1] if selected_executable_windows else None
    )
    missed = bool(recovery_windows_all and not selected_executable_windows)

    return StrategyPreCollapseFailure(
        source_path=str(target.path),
        source=target.source,
        step=target.step,
        game_time=target.game_time,
        recorded_action=target.action_name,
        recorded_training_use=signal.recommended_training_use,
        recorded_label_quality=signal.label_quality,
        threat_state=classify_threat_state(target),
        avoidability=avoidability,
        executable_actions=executable_actions,
        non_stay_executable_actions=non_stay_actions,
        executable_recovery_actions=executable_recovery,
        recovery_blockers=recovery_blockers,
        lookback_seconds=lookback_seconds,
        pre_collapse_rows=len(pre_rows),
        pre_collapse_recovery_window_rows=len(recovery_windows_all),
        pre_collapse_selected_recovery_rows=len(selected_windows),
        pre_collapse_selected_executable_recovery_rows=len(selected_executable_windows),
        pre_collapse_recovery_executable_counts_by_action=_action_counts(
            recovery_executable_counts
        ),
        pre_collapse_recovery_selected_counts_by_action=_action_counts(
            recovery_selected_counts
        ),
        pre_collapse_recovery_selected_executable_counts_by_action=_action_counts(
            recovery_selected_executable_counts
        ),
        missed_pre_collapse_recovery=missed,
        last_executable_recovery_time=(
            last_executable.game_time if last_executable is not None else None
        ),
        last_executable_recovery_actions=(
            last_executable.executable_recovery_actions if last_executable else []
        ),
        last_selected_executable_recovery_time=(
            last_selected_executable.game_time
            if last_selected_executable is not None
            else None
        ),
        last_selected_executable_recovery_action=(
            last_selected_executable.selected_executable_recovery_action
            if last_selected_executable is not None
            else None
        ),
        start_metrics=_start_metrics(target),
        recovery_windows=_representative_windows(
            recovery_windows_all,
            max_windows=max_windows,
        ),
    )


def _window_for_row(
    row: _StrategyOutcomeRow,
    *,
    target: _StrategyOutcomeRow,
) -> StrategyPreCollapseRecoveryWindow:
    executable_recovery = _executable_recovery_actions(row)
    selected_recovery = row.action_name if row.action_name in RECOVERY_ACTION_NAMES else None
    selected_executable = (
        row.action_name
        if row.action_name in executable_recovery and _recovery_executability(row, row.action_name)[0]
        else None
    )
    return StrategyPreCollapseRecoveryWindow(
        source_path=str(row.path),
        source=row.source,
        target_step=target.step,
        target_game_time=target.game_time,
        step=row.step,
        game_time=row.game_time,
        seconds_before_target=target.game_time - row.game_time,
        recorded_action=row.action_name,
        threat_state=classify_threat_state(row),
        executable_recovery_actions=executable_recovery,
        selected_recovery_action=selected_recovery,
        selected_executable_recovery_action=selected_executable,
        recovery_blockers=_recovery_blockers(row),
        start_metrics=_start_metrics(row),
    )


def _representative_windows(
    windows: list[StrategyPreCollapseRecoveryWindow],
    *,
    max_windows: int,
) -> list[StrategyPreCollapseRecoveryWindow]:
    if max_windows <= 0:
        return []
    selected = [window for window in windows if window.selected_executable_recovery_action]
    missed = [window for window in windows if not window.selected_executable_recovery_action]
    ordered = [*selected[-max_windows:], *missed[-max_windows:]]
    deduped: dict[int, StrategyPreCollapseRecoveryWindow] = {
        window.step: window for window in ordered
    }
    return sorted(deduped.values(), key=lambda window: window.game_time)[-max_windows:]


def _avoidability(
    *,
    executable_actions: list[str],
    executable_recovery_actions: list[str],
) -> str:
    if executable_recovery_actions:
        return "avoidable_recovery_available"
    non_stay_actions = [action for action in executable_actions if action != "STAY_COURSE"]
    if non_stay_actions:
        return "avoidable_non_recovery_available"
    if executable_actions == ["STAY_COURSE"]:
        return "unavoidable_only_stay_course"
    return "unavoidable_no_executable_actions"


def _executable_actions(row: _StrategyOutcomeRow) -> list[str]:
    actions = [
        STRATEGY_ACTION_NAMES[action_id]
        for action_id in sorted(STRATEGY_ACTION_NAMES)
        if candidate_executability(row, STRATEGY_ACTION_NAMES[action_id])[0]
    ]
    return actions


def _executable_recovery_actions(row: _StrategyOutcomeRow) -> list[str]:
    return [
        action
        for action in RECOVERY_ACTION_NAMES
        if _recovery_executability(row, action)[0]
    ]


def _recovery_blockers(row: _StrategyOutcomeRow) -> dict[str, str]:
    blockers: dict[str, str] = {}
    for action in RECOVERY_ACTION_NAMES:
        executable, blocker = _recovery_executability(row, action)
        if not executable:
            blockers[action] = str(blocker or "unknown")
    return blockers


def _recovery_executability(
    row: _StrategyOutcomeRow,
    action: str,
) -> tuple[bool, str | None]:
    if _recorded_execution_succeeded(row, action):
        return True, None
    return candidate_executability(row, action)


def _recorded_execution_succeeded(row: _StrategyOutcomeRow, action: str) -> bool:
    if action != row.action_name or row.execution_blocker:
        return False
    return row.execution_effect not in {None, "", "noop"}


def _start_metrics(row: _StrategyOutcomeRow) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        field: _value(row.observation, field)
        for field in PRE_COLLAPSE_START_METRICS
    }
    metrics["map_name"] = row.map_name
    metrics["difficulty"] = row.difficulty
    metrics["opponent_race"] = row.opponent_race
    metrics["opponent_ai_build"] = row.opponent_ai_build
    return metrics


def _row_key(row: _StrategyOutcomeRow) -> tuple[str, int, str]:
    return _record_key(str(row.path), row.step, row.action_name)


def _record_key(path: str, step: int, action: str) -> tuple[str, int, str]:
    return (str(path), int(step), str(action))


def _action_set_key(actions: list[str]) -> str:
    return "+".join(actions) if actions else "<none>"


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _merge_nested_counts(
    target: dict[str, Counter[str]],
    source: dict[str, str],
) -> None:
    for action, blocker in source.items():
        target[action][blocker] += 1


def _action_counts(counter: Counter[str]) -> dict[str, int]:
    return {
        action: int(counter[action])
        for action in RECOVERY_ACTION_NAMES
        if counter[action]
    }


def _nested_counts(counts_by_action: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        action: _sorted_counter(counter)
        for action, counter in counts_by_action.items()
        if counter
    }


def _blocking_reasons(
    *,
    missed_pre_collapse_recovery_rows: int,
    missed_pre_collapse_recovery_rate: float,
    max_missed_pre_collapse_recovery_rows: int,
    max_missed_pre_collapse_recovery_rate: float,
) -> list[str]:
    reasons: list[str] = []
    if missed_pre_collapse_recovery_rows > max_missed_pre_collapse_recovery_rows:
        reasons.append("missed_pre_collapse_recovery_rows")
    if missed_pre_collapse_recovery_rate > max_missed_pre_collapse_recovery_rate:
        reasons.append("missed_pre_collapse_recovery_rate")
    return reasons


def _count(values) -> dict[str, int]:
    return _sorted_counter(Counter(str(value) for value in values))


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in counter.items()))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
