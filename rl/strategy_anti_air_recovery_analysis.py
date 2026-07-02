"""Diagnose anti-air recovery windows before air-threat gaps."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path

from rl.strategy_datasets import (
    StrategyTrajectoryPathInput,
    discover_strategy_trajectory_files,
)
from rl.strategy_emergency_action_analysis import (
    _air_defense_gap_reason,
    _has_air_defense_assets,
)
from rl.strategy_outcome_diagnostics import (
    _StrategyOutcomeRow,
    _iter_valid_strategy_rows,
    _source_for_file,
    _value,
)
from rl.strategy_replay_candidate import candidate_executability, classify_threat_state


RECOVERY_ACTION_NAMES: tuple[str, ...] = (
    "PRODUCE_ARMY",
    "BUILD_STATIC_DEFENSE",
    "TECH_ROBO",
)
PHOTON_CANNON_MINERALS = 150.0
ANTI_AIR_RECOVERY_METRICS: tuple[str, ...] = (
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
    "zealots",
    "stalkers",
    "sentries",
    "immortals",
    "observers",
    "army_count",
    "base_under_air_threat",
    "base_under_ground_threat",
    "base_under_threat",
)


@dataclass(frozen=True)
class StrategyAntiAirRecoveryExample:
    """One row near the first air-threat gap without anti-air assets."""

    source_path: str
    source: str
    step: int
    game_time: float
    row_role: str
    seconds_before_gap: float | None
    recorded_action: str
    threat_state: str
    anti_air_assets_present: bool
    air_defense_gap_reason: str
    executable_recovery_actions: list[str]
    missed_executable_recovery_actions: list[str]
    recovery_blockers: dict[str, str]
    start_metrics: dict[str, float]


@dataclass(frozen=True)
class StrategyAntiAirRecoveryFileSummary:
    """Per-file anti-air recovery window summary."""

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
    anti_air_asset_rows: int
    first_anti_air_asset_time: float | None
    last_anti_air_asset_time: float | None
    first_anti_air_absent_after_asset_time: float | None
    air_threat_rows: int
    air_threat_rows_with_anti_air: int
    air_threat_rows_without_anti_air: int
    first_air_threat_time: float | None
    first_air_threat_without_anti_air_time: float | None
    last_anti_air_before_gap_time: float | None
    recovery_window_rows: int
    recovery_executable_counts_by_action: dict[str, int]
    recovery_selected_counts_by_action: dict[str, int]
    recovery_selected_executable_counts_by_action: dict[str, int]
    missed_executable_recovery_counts_by_action: dict[str, int]
    blockers_by_action: dict[str, dict[str, int]]
    recovery_window_available: bool
    recovery_selected_before_gap: bool
    recovery_selected_executable_before_gap: bool
    missed_recovery_window: bool


@dataclass(frozen=True)
class StrategyAntiAirRecoveryAnalysis:
    """Dataset-level anti-air recovery window diagnostics."""

    inputs: list[str]
    files: int
    rows: int
    training_rows: int
    result_counts: dict[str, int]
    recovery_action_names: list[str]
    anti_air_asset_rows: int
    air_threat_rows: int
    air_threat_rows_with_anti_air: int
    air_threat_rows_without_anti_air: int
    anti_air_gap_files: int
    files_with_pre_gap_recovery_window: int
    files_with_pre_gap_recovery_selected: int
    files_with_pre_gap_executable_recovery_selected: int
    missed_recovery_windows: int
    recovery_executable_counts_by_action: dict[str, int]
    recovery_selected_counts_by_action: dict[str, int]
    recovery_selected_executable_counts_by_action: dict[str, int]
    missed_executable_recovery_counts_by_action: dict[str, int]
    blockers_by_action: dict[str, dict[str, int]]
    file_summaries: list[StrategyAntiAirRecoveryFileSummary]
    examples: list[StrategyAntiAirRecoveryExample]


def analyze_strategy_anti_air_recovery(
    paths: StrategyTrajectoryPathInput,
    *,
    max_examples: int = 12,
) -> StrategyAntiAirRecoveryAnalysis:
    """Return anti-air recovery diagnostics for strategy trajectories."""
    if max_examples < 0:
        raise ValueError("max_examples must be >= 0")

    input_paths = _input_paths(paths)
    input_strings = [str(path) for path in input_paths]
    files = discover_strategy_trajectory_files(paths)

    rows_total = 0
    training_rows_total = 0
    result_counts: Counter[str] = Counter()
    anti_air_asset_rows = 0
    air_threat_rows = 0
    air_threat_rows_with_anti_air = 0
    air_threat_rows_without_anti_air = 0
    anti_air_gap_files = 0
    files_with_window = 0
    files_with_selected = 0
    files_with_selected_executable = 0
    missed_recovery_windows = 0
    executable_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    selected_executable_counts: Counter[str] = Counter()
    missed_counts: Counter[str] = Counter()
    blockers: dict[str, Counter[str]] = {
        action: Counter() for action in RECOVERY_ACTION_NAMES
    }
    file_summaries: list[StrategyAntiAirRecoveryFileSummary] = []
    examples: list[StrategyAntiAirRecoveryExample] = []

    for path in files:
        source = _source_for_file(path, input_paths)
        rows = list(_iter_valid_strategy_rows(path, source=source))
        training_rows = [row for row in rows if not row.done]
        result_counts.update(_result_key(row.result) for row in rows if row.done)
        summary, file_examples = _summarize_file(path, source, rows, training_rows)
        file_summaries.append(summary)

        rows_total += summary.rows
        training_rows_total += summary.training_rows
        anti_air_asset_rows += summary.anti_air_asset_rows
        air_threat_rows += summary.air_threat_rows
        air_threat_rows_with_anti_air += summary.air_threat_rows_with_anti_air
        air_threat_rows_without_anti_air += summary.air_threat_rows_without_anti_air
        if summary.first_air_threat_without_anti_air_time is not None:
            anti_air_gap_files += 1
        if summary.recovery_window_available:
            files_with_window += 1
        if summary.recovery_selected_before_gap:
            files_with_selected += 1
        if summary.recovery_selected_executable_before_gap:
            files_with_selected_executable += 1
        if summary.missed_recovery_window:
            missed_recovery_windows += 1
        executable_counts.update(summary.recovery_executable_counts_by_action)
        selected_counts.update(summary.recovery_selected_counts_by_action)
        selected_executable_counts.update(
            summary.recovery_selected_executable_counts_by_action
        )
        missed_counts.update(summary.missed_executable_recovery_counts_by_action)
        _merge_nested_counts(blockers, summary.blockers_by_action)
        examples.extend(file_examples)

    examples = sorted(
        examples,
        key=lambda example: (
            0 if example.missed_executable_recovery_actions else 1,
            0 if example.row_role == "pre_gap" else 1,
            example.seconds_before_gap
            if example.seconds_before_gap is not None
            else float("inf"),
            Path(example.source_path).name,
            example.step,
        ),
    )[:max_examples]

    return StrategyAntiAirRecoveryAnalysis(
        inputs=input_strings,
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        result_counts=_sorted_counter(result_counts),
        recovery_action_names=list(RECOVERY_ACTION_NAMES),
        anti_air_asset_rows=anti_air_asset_rows,
        air_threat_rows=air_threat_rows,
        air_threat_rows_with_anti_air=air_threat_rows_with_anti_air,
        air_threat_rows_without_anti_air=air_threat_rows_without_anti_air,
        anti_air_gap_files=anti_air_gap_files,
        files_with_pre_gap_recovery_window=files_with_window,
        files_with_pre_gap_recovery_selected=files_with_selected,
        files_with_pre_gap_executable_recovery_selected=files_with_selected_executable,
        missed_recovery_windows=missed_recovery_windows,
        recovery_executable_counts_by_action=_action_counts(executable_counts),
        recovery_selected_counts_by_action=_action_counts(selected_counts),
        recovery_selected_executable_counts_by_action=_action_counts(
            selected_executable_counts
        ),
        missed_executable_recovery_counts_by_action=_action_counts(missed_counts),
        blockers_by_action=_nested_counts(blockers),
        file_summaries=file_summaries,
        examples=examples,
    )


def _summarize_file(
    path: Path,
    source: str,
    rows: list[_StrategyOutcomeRow],
    training_rows: list[_StrategyOutcomeRow],
) -> tuple[StrategyAntiAirRecoveryFileSummary, list[StrategyAntiAirRecoveryExample]]:
    first = rows[0] if rows else None
    last = rows[-1] if rows else None
    anti_air_rows = [row for row in training_rows if _has_air_defense_assets(row)]
    air_threat_rows = [row for row in training_rows if _has_air_threat(row)]
    air_with_assets = [
        row for row in air_threat_rows if _has_air_defense_assets(row)
    ]
    air_without_assets = [
        row for row in air_threat_rows if not _has_air_defense_assets(row)
    ]
    first_gap = air_without_assets[0] if air_without_assets else None
    first_gap_time = first_gap.game_time if first_gap is not None else None
    pre_gap_rows = (
        [row for row in training_rows if row.game_time < first_gap_time]
        if first_gap_time is not None
        else []
    )

    executable_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    selected_executable_counts: Counter[str] = Counter()
    missed_counts: Counter[str] = Counter()
    blockers: dict[str, Counter[str]] = {
        action: Counter() for action in RECOVERY_ACTION_NAMES
    }

    for row in pre_gap_rows:
        executable_actions = _executable_recovery_actions(row)
        if row.action_name in RECOVERY_ACTION_NAMES:
            selected_counts[row.action_name] += 1
        for action in RECOVERY_ACTION_NAMES:
            if action in executable_actions:
                executable_counts[action] += 1
                if row.action_name == action:
                    selected_executable_counts[action] += 1
                else:
                    missed_counts[action] += 1
                continue
            _, blocker = anti_air_recovery_executability(row, action)
            blockers[action][str(blocker or "unknown")] += 1

    recovery_window_available = bool(sum(executable_counts.values()))
    selected_before_gap = bool(sum(selected_counts.values()))
    selected_executable_before_gap = bool(sum(selected_executable_counts.values()))
    missed_recovery_window = bool(
        first_gap is not None
        and recovery_window_available
        and not selected_executable_before_gap
    )
    file_examples = _examples_for_file(
        path=path,
        source=source,
        first_gap=first_gap,
        pre_gap_rows=pre_gap_rows,
    )

    return (
        StrategyAntiAirRecoveryFileSummary(
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
            first_game_time=first.game_time if first is not None else None,
            last_game_time=last.game_time if last is not None else None,
            anti_air_asset_rows=len(anti_air_rows),
            first_anti_air_asset_time=_first_time(anti_air_rows),
            last_anti_air_asset_time=_last_time(anti_air_rows),
            first_anti_air_absent_after_asset_time=_first_absent_after_asset_time(
                training_rows
            ),
            air_threat_rows=len(air_threat_rows),
            air_threat_rows_with_anti_air=len(air_with_assets),
            air_threat_rows_without_anti_air=len(air_without_assets),
            first_air_threat_time=_first_time(air_threat_rows),
            first_air_threat_without_anti_air_time=first_gap_time,
            last_anti_air_before_gap_time=_last_anti_air_before_gap_time(
                anti_air_rows,
                first_gap_time,
            ),
            recovery_window_rows=len(pre_gap_rows),
            recovery_executable_counts_by_action=_action_counts(executable_counts),
            recovery_selected_counts_by_action=_action_counts(selected_counts),
            recovery_selected_executable_counts_by_action=_action_counts(
                selected_executable_counts
            ),
            missed_executable_recovery_counts_by_action=_action_counts(missed_counts),
            blockers_by_action=_nested_counts(blockers),
            recovery_window_available=recovery_window_available,
            recovery_selected_before_gap=selected_before_gap,
            recovery_selected_executable_before_gap=selected_executable_before_gap,
            missed_recovery_window=missed_recovery_window,
        ),
        file_examples,
    )


def _examples_for_file(
    *,
    path: Path,
    source: str,
    first_gap: _StrategyOutcomeRow | None,
    pre_gap_rows: list[_StrategyOutcomeRow],
) -> list[StrategyAntiAirRecoveryExample]:
    if first_gap is None:
        return []
    candidates = sorted(
        pre_gap_rows,
        key=lambda row: (
            0 if _missed_executable_recovery_actions(row) else 1,
            first_gap.game_time - row.game_time,
            -row.game_time,
        ),
    )[:3]
    if not candidates:
        candidates = [first_gap]
    examples = [
        _example_for_row(
            path=path,
            source=source,
            row=row,
            first_gap_time=first_gap.game_time,
            row_role="pre_gap" if row is not first_gap else "gap",
        )
        for row in candidates
    ]
    return examples


def _example_for_row(
    *,
    path: Path,
    source: str,
    row: _StrategyOutcomeRow,
    first_gap_time: float,
    row_role: str,
) -> StrategyAntiAirRecoveryExample:
    executable_actions = _executable_recovery_actions(row)
    return StrategyAntiAirRecoveryExample(
        source_path=str(path),
        source=source,
        step=row.step,
        game_time=row.game_time,
        row_role=row_role,
        seconds_before_gap=(
            float(first_gap_time - row.game_time) if row.game_time <= first_gap_time else None
        ),
        recorded_action=row.action_name,
        threat_state=classify_threat_state(row),
        anti_air_assets_present=_has_air_defense_assets(row),
        air_defense_gap_reason=_air_defense_gap_reason(row),
        executable_recovery_actions=list(executable_actions),
        missed_executable_recovery_actions=[
            action for action in executable_actions if row.action_name != action
        ],
        recovery_blockers=_recovery_blockers(row),
        start_metrics=_start_metrics(row),
    )


def _executable_recovery_actions(row: _StrategyOutcomeRow) -> tuple[str, ...]:
    return tuple(
        action
        for action in RECOVERY_ACTION_NAMES
        if anti_air_recovery_executability(row, action)[0]
    )


def _missed_executable_recovery_actions(row: _StrategyOutcomeRow) -> tuple[str, ...]:
    return tuple(
        action
        for action in _executable_recovery_actions(row)
        if row.action_name != action
    )


def _recovery_blockers(row: _StrategyOutcomeRow) -> dict[str, str]:
    blocked: dict[str, str] = {}
    for action in RECOVERY_ACTION_NAMES:
        executable, blocker = anti_air_recovery_executability(row, action)
        if not executable:
            blocked[action] = str(blocker or "unknown")
    return blocked


def anti_air_recovery_executability(
    row: _StrategyOutcomeRow,
    action: str,
) -> tuple[bool, str | None]:
    """Return whether an action plausibly rebuilds anti-air capability."""
    if _recorded_recovery_execution_succeeded(row, action):
        return True, None
    executable, blocker = candidate_executability(row, action)
    if not executable:
        return executable, blocker
    if action != "BUILD_STATIC_DEFENSE":
        return executable, blocker
    if _value(row.observation, "ready_forge") <= 0.0:
        return False, "missing_ready_forge_for_photon_cannon"
    if _value(row.observation, "minerals") < PHOTON_CANNON_MINERALS:
        return False, "cannot_afford_photon_cannon"
    return True, None


def _recorded_recovery_execution_succeeded(
    row: _StrategyOutcomeRow,
    action: str,
) -> bool:
    if action != row.action_name or row.execution_blocker:
        return False
    if row.execution_effect in {None, "", "noop"}:
        return False
    if action == "BUILD_STATIC_DEFENSE":
        return row.execution_unit_type == "PHOTONCANNON"
    return action in {"PRODUCE_ARMY", "TECH_ROBO"}


def _has_air_threat(row: _StrategyOutcomeRow) -> bool:
    return _value(row.observation, "base_under_air_threat") > 0.0


def _first_absent_after_asset_time(
    rows: list[_StrategyOutcomeRow],
) -> float | None:
    seen_asset = False
    for row in rows:
        if _has_air_defense_assets(row):
            seen_asset = True
            continue
        if seen_asset:
            return row.game_time
    return None


def _last_anti_air_before_gap_time(
    anti_air_rows: list[_StrategyOutcomeRow],
    first_gap_time: float | None,
) -> float | None:
    if first_gap_time is None:
        return None
    before_gap = [row.game_time for row in anti_air_rows if row.game_time < first_gap_time]
    if not before_gap:
        return None
    return max(before_gap)


def _start_metrics(row: _StrategyOutcomeRow) -> dict[str, float]:
    metrics = {
        field: _value(row.observation, field)
        for field in ANTI_AIR_RECOVERY_METRICS
    }
    metrics["ready_photon_cannons"] = _detail_value(row, "ready_photon_cannons")
    metrics["pending_photon_cannons"] = _detail_value(row, "pending_photon_cannons")
    metrics["ready_shield_batteries"] = _detail_value(row, "ready_shield_batteries")
    metrics["pending_shield_batteries"] = _detail_value(
        row,
        "pending_shield_batteries",
    )
    return metrics


def _detail_value(row: _StrategyOutcomeRow, field: str) -> float:
    return float(row.observation_details.get(field, 0.0))


def _first_time(rows: list[_StrategyOutcomeRow]) -> float | None:
    return rows[0].game_time if rows else None


def _last_time(rows: list[_StrategyOutcomeRow]) -> float | None:
    return rows[-1].game_time if rows else None


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


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _merge_nested_counts(
    target: dict[str, Counter[str]],
    source: dict[str, dict[str, int]],
) -> None:
    for action, counts in source.items():
        target.setdefault(action, Counter()).update(counts)


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


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in counter.items()))
