"""Diagnostics for tactic metadata in strategy trajectory JSONL files."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_datasets import StrategyTrajectoryPathInput, discover_strategy_trajectory_files


@dataclass(frozen=True)
class TacticFilterChangeSummary:
    """Count for one tactic filter before/after action change."""

    opponent_ai_build: str
    tactic_id: str
    before_action: str
    after_action: str
    count: int


@dataclass(frozen=True)
class TacticTimelineSegment:
    """One consecutive tactic run in a strategy trajectory file."""

    tactic_id: str
    tactic_phase: str
    tactic_source: str
    start_step: int
    end_step: int
    start_game_time: float
    end_game_time: float
    count: int


@dataclass(frozen=True)
class TacticFilterTimelineEvent:
    """One strategy row with tactic filter before/after metadata."""

    line_number: int
    step: int
    game_time: float
    opponent_ai_build: str
    tactic_id: str
    tactic_phase: str
    tactic_source: str
    original_action: str
    selected_action: str
    changed: bool
    minerals: float
    vespene: float
    supply_left: float
    pending_gateways: float
    ready_gateways: float
    pending_robo: float
    ready_robo: float
    pending_static_defense: float
    ready_static_defense: float
    base_under_threat: float
    gateway_idle_count: float
    robo_idle_count: float


@dataclass(frozen=True)
class FileTacticSummary:
    """Per-file tactic metadata summary."""

    path: str
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    result: str | None
    rows: int
    training_rows: int
    rows_with_tactic_metadata: int
    rows_with_filter_metadata: int
    filter_change_rows: int
    training_rows_with_tactic_metadata: int
    training_rows_with_filter_metadata: int
    training_filter_change_rows: int
    tactic_counts: dict[str, int]
    tactic_phase_counts: dict[str, int]
    tactic_source_counts: dict[str, int]
    filter_changes: list[TacticFilterChangeSummary]
    timeline: list[TacticTimelineSegment]
    filter_timeline: list[TacticFilterTimelineEvent]


@dataclass(frozen=True)
class TacticDiagnostics:
    """Dataset-level tactic metadata diagnostics."""

    inputs: list[str]
    files: int
    rows: int
    training_rows: int
    rows_with_tactic_metadata: int
    rows_with_filter_metadata: int
    filter_change_rows: int
    training_rows_with_tactic_metadata: int
    training_rows_with_filter_metadata: int
    training_filter_change_rows: int
    opponent_ai_build_counts: dict[str, int]
    tactic_counts: dict[str, int]
    tactic_phase_counts: dict[str, int]
    tactic_source_counts: dict[str, int]
    filter_changes: list[TacticFilterChangeSummary]
    result_counts: dict[str, int]
    file_summaries: list[FileTacticSummary]


@dataclass(frozen=True)
class _TacticRow:
    path: Path
    line_number: int
    step: int
    game_time: float
    done: bool
    result: str | None
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    tactic_id: str | None
    tactic_phase: str | None
    tactic_source: str | None
    before_action: str | None
    after_action: str | None
    selected_action: str | None
    minerals: float
    vespene: float
    supply_left: float
    pending_gateways: float
    ready_gateways: float
    pending_robo: float
    ready_robo: float
    pending_static_defense: float
    ready_static_defense: float
    base_under_threat: float
    gateway_idle_count: float
    robo_idle_count: float


def diagnose_tactics(paths: StrategyTrajectoryPathInput) -> TacticDiagnostics:
    """Return tactic metadata diagnostics for strategy trajectory JSONL files."""
    input_paths = [str(path) for path in _input_paths(paths)]
    files = discover_strategy_trajectory_files(paths)
    rows_total = 0
    training_rows_total = 0
    rows_with_tactic_metadata = 0
    rows_with_filter_metadata = 0
    filter_change_rows = 0
    training_rows_with_tactic_metadata = 0
    training_rows_with_filter_metadata = 0
    training_filter_change_rows = 0
    opponent_ai_build_counts: Counter[str] = Counter()
    tactic_counts: Counter[str] = Counter()
    tactic_phase_counts: Counter[str] = Counter()
    tactic_source_counts: Counter[str] = Counter()
    filter_change_counts: Counter[tuple[str, str, str, str]] = Counter()
    result_counts: Counter[str] = Counter()
    file_summaries: list[FileTacticSummary] = []

    for path in files:
        file_rows = list(_iter_tactic_rows(path))
        rows_total += len(file_rows)
        training_rows = [row for row in file_rows if not row.done]
        training_rows_total += len(training_rows)
        file_rows_with_tactic = 0
        file_rows_with_filter = 0
        file_filter_change_rows = 0
        for row in file_rows:
            if row.done:
                result_counts[_result_key(row.result)] += 1
            if row.tactic_id:
                rows_with_tactic_metadata += 1
                file_rows_with_tactic += 1
            if row.before_action is not None and row.after_action is not None:
                rows_with_filter_metadata += 1
                file_rows_with_filter += 1
                if row.before_action != row.after_action:
                    filter_change_rows += 1
                    file_filter_change_rows += 1

        file_tactic_counts: Counter[str] = Counter()
        file_phase_counts: Counter[str] = Counter()
        file_source_counts: Counter[str] = Counter()
        file_filter_counts: Counter[tuple[str, str, str, str]] = Counter()
        file_training_rows_with_tactic = 0
        file_training_rows_with_filter = 0
        file_training_filter_change_rows = 0

        for row in training_rows:
            build = _metadata_key(row.opponent_ai_build, default="RandomBuild")
            tactic = _metadata_key(row.tactic_id)
            phase = _metadata_key(row.tactic_phase)
            source = _metadata_key(row.tactic_source)
            opponent_ai_build_counts[build] += 1

            if row.tactic_id:
                training_rows_with_tactic_metadata += 1
                file_training_rows_with_tactic += 1
                tactic_counts[tactic] += 1
                file_tactic_counts[tactic] += 1
            if row.tactic_phase:
                tactic_phase_counts[phase] += 1
                file_phase_counts[phase] += 1
            if row.tactic_source:
                tactic_source_counts[source] += 1
                file_source_counts[source] += 1

            if row.before_action is not None and row.after_action is not None:
                training_rows_with_filter_metadata += 1
                file_training_rows_with_filter += 1
                if row.before_action != row.after_action:
                    key = (build, tactic, row.before_action, row.after_action)
                    filter_change_counts[key] += 1
                    file_filter_counts[key] += 1
                    training_filter_change_rows += 1
                    file_training_filter_change_rows += 1

        file_summaries.append(
            FileTacticSummary(
                path=str(path),
                map_name=_first_non_empty(file_rows, "map_name"),
                difficulty=_first_non_empty(file_rows, "difficulty"),
                opponent_race=_first_non_empty(file_rows, "opponent_race"),
                opponent_ai_build=_first_non_empty(
                    file_rows,
                    "opponent_ai_build",
                    default="RandomBuild",
                ),
                result=next(
                    (row.result for row in file_rows if row.done and row.result),
                    None,
                ),
                rows=len(file_rows),
                training_rows=len(training_rows),
                rows_with_tactic_metadata=file_rows_with_tactic,
                rows_with_filter_metadata=file_rows_with_filter,
                filter_change_rows=file_filter_change_rows,
                training_rows_with_tactic_metadata=file_training_rows_with_tactic,
                training_rows_with_filter_metadata=file_training_rows_with_filter,
                training_filter_change_rows=file_training_filter_change_rows,
                tactic_counts=_sorted_counts(file_tactic_counts),
                tactic_phase_counts=_sorted_counts(file_phase_counts),
                tactic_source_counts=_sorted_counts(file_source_counts),
                filter_changes=_filter_change_summaries(file_filter_counts),
                timeline=_timeline_segments(training_rows),
                filter_timeline=_filter_timeline_events(training_rows),
            )
        )

    return TacticDiagnostics(
        inputs=input_paths,
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        rows_with_tactic_metadata=rows_with_tactic_metadata,
        rows_with_filter_metadata=rows_with_filter_metadata,
        filter_change_rows=filter_change_rows,
        training_rows_with_tactic_metadata=training_rows_with_tactic_metadata,
        training_rows_with_filter_metadata=training_rows_with_filter_metadata,
        training_filter_change_rows=training_filter_change_rows,
        opponent_ai_build_counts=_sorted_counts(opponent_ai_build_counts),
        tactic_counts=_sorted_counts(tactic_counts),
        tactic_phase_counts=_sorted_counts(tactic_phase_counts),
        tactic_source_counts=_sorted_counts(tactic_source_counts),
        filter_changes=_filter_change_summaries(filter_change_counts),
        result_counts=dict(sorted(result_counts.items())),
        file_summaries=file_summaries,
    )


def _iter_tactic_rows(path: Path) -> Iterable[_TacticRow]:
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
            observation = row.get("strategy_observation")
            if not isinstance(observation, dict):
                observation = {}
            yield _TacticRow(
                path=path,
                line_number=line_number,
                step=_optional_int(row.get("step"), default=line_number),
                game_time=_optional_float(observation.get("game_time"), default=0.0),
                done=bool(row.get("done", False)),
                result=_optional_str(row.get("result")),
                map_name=str(row.get("map_name", "")),
                difficulty=str(row.get("difficulty", "")),
                opponent_race=str(row.get("opponent_race", "")),
                opponent_ai_build=str(row.get("opponent_ai_build", "RandomBuild")),
                tactic_id=_optional_str(row.get("tactic_id")),
                tactic_phase=_optional_str(row.get("tactic_phase")),
                tactic_source=_optional_str(row.get("tactic_source")),
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
                selected_action=_strategy_action_name(
                    row,
                    id_key="strategy_action",
                    name_key="strategy_action_name",
                ),
                minerals=_optional_float(observation.get("minerals"), default=0.0),
                vespene=_optional_float(observation.get("vespene"), default=0.0),
                supply_left=_optional_float(observation.get("supply_left"), default=0.0),
                pending_gateways=_optional_float(
                    observation.get("pending_gateways"),
                    default=0.0,
                ),
                ready_gateways=_optional_float(
                    observation.get("ready_gateways"),
                    default=0.0,
                ),
                pending_robo=_optional_float(observation.get("pending_robo"), default=0.0),
                ready_robo=_optional_float(observation.get("ready_robo"), default=0.0),
                pending_static_defense=_optional_float(
                    observation.get("pending_static_defense"),
                    default=0.0,
                ),
                ready_static_defense=_optional_float(
                    observation.get("ready_static_defense"),
                    default=0.0,
                ),
                base_under_threat=_optional_float(
                    observation.get("base_under_threat"),
                    default=0.0,
                ),
                gateway_idle_count=_optional_float(
                    observation.get("gateway_idle_count"),
                    default=0.0,
                ),
                robo_idle_count=_optional_float(
                    observation.get("robo_idle_count"),
                    default=0.0,
                ),
            )


def _timeline_segments(rows: list[_TacticRow]) -> list[TacticTimelineSegment]:
    if not rows:
        return []
    segments: list[TacticTimelineSegment] = []
    start = rows[0]
    previous = rows[0]
    count = 1
    for row in rows[1:]:
        if _same_tactic_segment(row, previous):
            previous = row
            count += 1
            continue
        segments.append(_make_segment(start, previous, count))
        start = row
        previous = row
        count = 1
    segments.append(_make_segment(start, previous, count))
    return segments


def _same_tactic_segment(left: _TacticRow, right: _TacticRow) -> bool:
    return (
        _metadata_key(left.tactic_id) == _metadata_key(right.tactic_id)
        and _metadata_key(left.tactic_phase) == _metadata_key(right.tactic_phase)
        and _metadata_key(left.tactic_source) == _metadata_key(right.tactic_source)
    )


def _make_segment(
    start: _TacticRow,
    end: _TacticRow,
    count: int,
) -> TacticTimelineSegment:
    return TacticTimelineSegment(
        tactic_id=_metadata_key(start.tactic_id),
        tactic_phase=_metadata_key(start.tactic_phase),
        tactic_source=_metadata_key(start.tactic_source),
        start_step=start.step,
        end_step=end.step,
        start_game_time=start.game_time,
        end_game_time=end.game_time,
        count=count,
    )


def _filter_timeline_events(rows: list[_TacticRow]) -> list[TacticFilterTimelineEvent]:
    events: list[TacticFilterTimelineEvent] = []
    for row in rows:
        if row.before_action is None or row.after_action is None:
            continue
        events.append(
            TacticFilterTimelineEvent(
                line_number=row.line_number,
                step=row.step,
                game_time=row.game_time,
                opponent_ai_build=_metadata_key(
                    row.opponent_ai_build,
                    default="RandomBuild",
                ),
                tactic_id=_metadata_key(row.tactic_id),
                tactic_phase=_metadata_key(row.tactic_phase),
                tactic_source=_metadata_key(row.tactic_source),
                original_action=row.before_action,
                selected_action=row.after_action,
                changed=row.before_action != row.after_action,
                minerals=row.minerals,
                vespene=row.vespene,
                supply_left=row.supply_left,
                pending_gateways=row.pending_gateways,
                ready_gateways=row.ready_gateways,
                pending_robo=row.pending_robo,
                ready_robo=row.ready_robo,
                pending_static_defense=row.pending_static_defense,
                ready_static_defense=row.ready_static_defense,
                base_under_threat=row.base_under_threat,
                gateway_idle_count=row.gateway_idle_count,
                robo_idle_count=row.robo_idle_count,
            )
        )
    return events


def _filter_change_summaries(
    counts: Counter[tuple[str, str, str, str]],
) -> list[TacticFilterChangeSummary]:
    summaries = [
        TacticFilterChangeSummary(
            opponent_ai_build=build,
            tactic_id=tactic_id,
            before_action=before_action,
            after_action=after_action,
            count=int(count),
        )
        for (build, tactic_id, before_action, after_action), count in counts.items()
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


def _strategy_action_name(row: dict[str, Any], *, id_key: str, name_key: str) -> str | None:
    name = _optional_str(row.get(name_key))
    if name is not None:
        return name
    action_id = _optional_int(row.get(id_key), default=None)
    if action_id is None:
        return None
    return STRATEGY_ACTION_NAMES.get(action_id)


def _metadata_key(value: str | None, *, default: str = "<none>") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _sorted_counts(counts: Counter[str]) -> dict[str, int]:
    return dict(sorted((str(key), int(count)) for key, count in counts.items()))


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _first_non_empty(
    rows: list[_TacticRow],
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


def _optional_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _result_key(value: Any) -> str:
    if value is None or value == "":
        return "NO_RESULT"
    return str(value)
