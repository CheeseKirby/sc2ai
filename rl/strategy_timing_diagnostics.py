"""Timing-oriented diagnostics for strategy trajectory JSONL files."""
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
    infer_strategy_observation_schema_version,
    normalize_strategy_observation_dict,
    validate_strategy_observation_dict,
)


PENDING_REPEAT_FIELDS: dict[int, tuple[str, ...]] = {
    1: ("pending_bases",),
    2: ("pending_gateways",),
    3: ("pending_robo",),
    4: (
        "pending_forge",
        "ground_weapon_upgrade_pending",
        "ground_armor_upgrade_pending",
    ),
    5: ("pending_static_defense",),
}


@dataclass(frozen=True)
class ActionTimingSummary:
    """Timing stats for one strategy action."""

    count: int
    first_game_time: float | None
    min_game_time: float | None
    max_game_time: float | None
    avg_game_time: float | None


@dataclass(frozen=True)
class TimelineSegment:
    """One consecutive action run in a strategy trajectory file."""

    action_name: str
    start_step: int
    end_step: int
    start_game_time: float
    end_game_time: float
    count: int


@dataclass(frozen=True)
class SignalLatencySummary:
    """Latency from a tactical signal to first later TECH_ROBO."""

    signal_field: str
    files_with_signal: int
    files_with_tech_after_signal: int
    files_with_tech_before_signal: int
    files_without_tech: int
    files_without_tech_after_signal: int
    avg_delay: float
    min_delay: float
    max_delay: float
    delays: list[float]
    avg_early_lead: float
    min_early_lead: float
    max_early_lead: float
    early_leads: list[float]
    early_file_paths: list[str]
    no_tech_file_paths: list[str]
    missing_file_paths: list[str]


@dataclass(frozen=True)
class FileStrategyTimingSummary:
    """Per-file strategy timing summary."""

    path: str
    map_name: str
    difficulty: str
    opponent_race: str
    result: str | None
    rows: int
    training_rows: int
    first_step: int | None
    last_step: int | None
    first_game_time: float | None
    last_game_time: float | None
    action_counts_by_name: dict[str, int]
    action_first_game_time_by_name: dict[str, float]
    threat_action_counts_by_name: dict[str, int]
    pending_repeat_counts_by_name: dict[str, int]
    signal_first_game_time: dict[str, float]
    tech_robo_first_game_time: float | None
    timeline: list[TimelineSegment]


@dataclass(frozen=True)
class StrategyTimingDiagnostics:
    """Dataset-level timing diagnostics for strategy trajectories."""

    inputs: list[str]
    files: int
    rows: int
    training_rows: int
    result_counts: dict[str, int]
    action_timing_by_name: dict[str, ActionTimingSummary]
    threat_action_counts_by_name: dict[str, int]
    pending_repeat_counts_by_name: dict[str, int]
    tech_robo_latency: dict[str, SignalLatencySummary]
    hard_defeat_file_paths: list[str]
    file_summaries: list[FileStrategyTimingSummary]


@dataclass(frozen=True)
class _StrategyTimingRow:
    path: Path
    step: int
    game_time: float
    action_id: int
    action_name: str
    done: bool
    result: str | None
    observation: dict[str, float]
    map_name: str
    difficulty: str
    opponent_race: str


def diagnose_strategy_timing(
    paths: StrategyTrajectoryPathInput,
) -> StrategyTimingDiagnostics:
    """Return timing diagnostics for strategy trajectory JSONL files."""
    input_paths = [str(path) for path in _input_paths(paths)]
    files = discover_strategy_trajectory_files(paths)
    file_summaries: list[FileStrategyTimingSummary] = []
    rows_total = 0
    training_rows_total = 0
    result_counts: Counter[str] = Counter()
    action_times: dict[int, list[float]] = defaultdict(list)
    threat_action_counts: Counter[int] = Counter()
    pending_repeat_counts: Counter[int] = Counter()

    for path in files:
        file_rows = list(_iter_valid_strategy_rows(path))
        rows_total += len(file_rows)
        training_rows = [row for row in file_rows if not row.done]
        training_rows_total += len(training_rows)
        for row in file_rows:
            if row.done:
                result_counts[_result_key(row.result)] += 1
        for row in training_rows:
            action_times[row.action_id].append(row.game_time)
            if _value(row.observation, "base_under_threat") > 0.0:
                threat_action_counts[row.action_id] += 1
            if _is_pending_repeat(row.action_id, row.observation):
                pending_repeat_counts[row.action_id] += 1

        file_summaries.append(_summarize_file(path, file_rows, training_rows))

    return StrategyTimingDiagnostics(
        inputs=input_paths,
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        result_counts=dict(sorted(result_counts.items())),
        action_timing_by_name={
            STRATEGY_ACTION_NAMES[action_id]: _action_timing(times)
            for action_id, times in sorted(action_times.items())
        },
        threat_action_counts_by_name=_counts_by_name(threat_action_counts),
        pending_repeat_counts_by_name=_counts_by_name(pending_repeat_counts),
        tech_robo_latency={
            "armored_signal": _latency_summary(
                file_summaries,
                signal_key="enemy_armored_units_known",
            ),
            "cloaked_signal": _latency_summary(
                file_summaries,
                signal_key="enemy_cloaked_units_seen",
            ),
        },
        hard_defeat_file_paths=[
            summary.path
            for summary in file_summaries
            if summary.difficulty == "Hard" and _is_defeat_result(summary.result)
        ],
        file_summaries=file_summaries,
    )


def _iter_valid_strategy_rows(path: Path) -> Iterable[_StrategyTimingRow]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
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
            step = _optional_int(row.get("step"))
            if step is None:
                step = line_number
            yield _StrategyTimingRow(
                path=path,
                step=step,
                game_time=float(normalized.get("game_time", 0.0)),
                action_id=action,
                action_name=STRATEGY_ACTION_NAMES[action],
                done=bool(row.get("done", False)),
                result=row.get("result"),
                observation=normalized,
                map_name=str(row.get("map_name", "")),
                difficulty=str(row.get("difficulty", "")),
                opponent_race=str(row.get("opponent_race", "")),
            )


def _summarize_file(
    path: Path,
    rows: list[_StrategyTimingRow],
    training_rows: list[_StrategyTimingRow],
) -> FileStrategyTimingSummary:
    result = next((row.result for row in rows if row.done and row.result), None)
    action_counts = Counter(row.action_id for row in training_rows)
    threat_action_counts = Counter(
        row.action_id
        for row in training_rows
        if _value(row.observation, "base_under_threat") > 0.0
    )
    pending_repeat_counts = Counter(
        row.action_id
        for row in training_rows
        if _is_pending_repeat(row.action_id, row.observation)
    )
    action_first_times: dict[str, float] = {}
    for row in training_rows:
        action_first_times.setdefault(row.action_name, row.game_time)

    signal_first = {
        "enemy_armored_units_known": _first_signal_time(
            training_rows,
            "enemy_armored_units_known",
        ),
        "enemy_cloaked_units_seen": _first_signal_time(
            training_rows,
            "enemy_cloaked_units_seen",
        ),
    }
    signal_first = {
        key: value
        for key, value in signal_first.items()
        if value is not None
    }
    tech_first = action_first_times.get("TECH_ROBO")
    first = rows[0] if rows else None
    last = rows[-1] if rows else None
    return FileStrategyTimingSummary(
        path=str(path),
        map_name=_first_non_empty(rows, "map_name"),
        difficulty=_first_non_empty(rows, "difficulty"),
        opponent_race=_first_non_empty(rows, "opponent_race"),
        result=result,
        rows=len(rows),
        training_rows=len(training_rows),
        first_step=first.step if first is not None else None,
        last_step=last.step if last is not None else None,
        first_game_time=first.game_time if first is not None else None,
        last_game_time=last.game_time if last is not None else None,
        action_counts_by_name=_counts_by_name(action_counts),
        action_first_game_time_by_name=dict(sorted(action_first_times.items())),
        threat_action_counts_by_name=_counts_by_name(threat_action_counts),
        pending_repeat_counts_by_name=_counts_by_name(pending_repeat_counts),
        signal_first_game_time=signal_first,
        tech_robo_first_game_time=tech_first,
        timeline=_timeline_segments(rows),
    )


def _timeline_segments(rows: list[_StrategyTimingRow]) -> list[TimelineSegment]:
    if not rows:
        return []
    segments: list[TimelineSegment] = []
    start = rows[0]
    previous = rows[0]
    count = 1
    for row in rows[1:]:
        if row.action_id == previous.action_id:
            previous = row
            count += 1
            continue
        segments.append(
            TimelineSegment(
                action_name=start.action_name,
                start_step=start.step,
                end_step=previous.step,
                start_game_time=start.game_time,
                end_game_time=previous.game_time,
                count=count,
            )
        )
        start = row
        previous = row
        count = 1
    segments.append(
        TimelineSegment(
            action_name=start.action_name,
            start_step=start.step,
            end_step=previous.step,
            start_game_time=start.game_time,
            end_game_time=previous.game_time,
            count=count,
        )
    )
    return segments


def _latency_summary(
    file_summaries: list[FileStrategyTimingSummary],
    *,
    signal_key: str,
) -> SignalLatencySummary:
    delays: list[float] = []
    early_leads: list[float] = []
    early_file_paths: list[str] = []
    no_tech_file_paths: list[str] = []
    missing: list[str] = []
    files_with_signal = 0
    for summary in file_summaries:
        signal_time = summary.signal_first_game_time.get(signal_key)
        if signal_time is None:
            continue
        files_with_signal += 1
        tech_time = summary.tech_robo_first_game_time
        if tech_time is not None and tech_time >= signal_time:
            delays.append(float(tech_time - signal_time))
        elif tech_time is not None:
            early_leads.append(float(signal_time - tech_time))
            early_file_paths.append(summary.path)
            missing.append(summary.path)
        else:
            no_tech_file_paths.append(summary.path)
            missing.append(summary.path)
    return SignalLatencySummary(
        signal_field=signal_key,
        files_with_signal=files_with_signal,
        files_with_tech_after_signal=len(delays),
        files_with_tech_before_signal=len(early_leads),
        files_without_tech=len(no_tech_file_paths),
        files_without_tech_after_signal=len(missing),
        avg_delay=(sum(delays) / len(delays)) if delays else 0.0,
        min_delay=min(delays) if delays else 0.0,
        max_delay=max(delays) if delays else 0.0,
        delays=delays,
        avg_early_lead=(sum(early_leads) / len(early_leads)) if early_leads else 0.0,
        min_early_lead=min(early_leads) if early_leads else 0.0,
        max_early_lead=max(early_leads) if early_leads else 0.0,
        early_leads=early_leads,
        early_file_paths=early_file_paths,
        no_tech_file_paths=no_tech_file_paths,
        missing_file_paths=missing,
    )


def _first_signal_time(
    rows: list[_StrategyTimingRow],
    field: str,
) -> float | None:
    for row in rows:
        if _value(row.observation, field) > 0.0:
            return row.game_time
    return None


def _is_pending_repeat(action_id: int, observation: dict[str, float]) -> bool:
    return any(
        _value(observation, field) > 0.0
        for field in PENDING_REPEAT_FIELDS.get(action_id, ())
    )


def _action_timing(times: list[float]) -> ActionTimingSummary:
    return ActionTimingSummary(
        count=len(times),
        first_game_time=times[0] if times else None,
        min_game_time=min(times) if times else None,
        max_game_time=max(times) if times else None,
        avg_game_time=(sum(times) / len(times)) if times else None,
    )


def _counts_by_name(counts: Counter[int] | dict[int, int]) -> dict[str, int]:
    return {
        STRATEGY_ACTION_NAMES[action_id]: int(count)
        for action_id, count in sorted(counts.items())
        if action_id in STRATEGY_ACTION_NAMES
    }


def _valid_action(value: Any) -> int | None:
    try:
        action = int(value)
    except (TypeError, ValueError):
        return None
    if action not in STRATEGY_ACTION_NAMES:
        return None
    return action


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_observation(row: dict[str, Any]) -> Any:
    if "strategy_observation" in row:
        return row["strategy_observation"]
    return row.get("observation")


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _first_non_empty(rows: list[_StrategyTimingRow], attr: str) -> str:
    for row in rows:
        value = getattr(row, attr)
        if value:
            return str(value)
    return ""


def _value(observation: dict[str, float], field: str) -> float:
    if field not in observation and field in STRATEGY_OBSERVATION_DEFAULTS:
        return float(STRATEGY_OBSERVATION_DEFAULTS[field])
    return float(observation.get(field, 0.0))


def _result_key(value: Any) -> str:
    if value is None or value == "":
        return "NO_RESULT"
    return str(value)


def _is_defeat_result(value: Any) -> bool:
    return _result_key(value).split(".")[-1] == "Defeat"
