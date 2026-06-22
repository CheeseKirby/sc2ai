"""Diagnostics for trajectory datasets used by imitation and RL workflows."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.actions import ACTION_NAMES
from rl.datasets import TrajectoryPathInput, discover_trajectory_files
from rl.observations import (
    OBSERVATION_DEFAULTS,
    OBSERVATION_FIELDS,
    infer_observation_schema_version,
    normalize_observation_dict,
    validate_observation_dict,
)


OBSERVATION_STAT_FIELDS: tuple[str, ...] = (
    "army_count",
    "attack_army_peak",
    "army_lost_from_peak",
    "army_lost_from_peak_ratio",
    "army_count_delta",
    "army_idle_count",
    "army_busy_count",
    "base_under_threat",
    "enemy_to_home_distance",
)


@dataclass(frozen=True)
class FileTrajectoryDiagnostics:
    """Per-file trajectory health summary."""

    path: str
    rows: int
    training_rows: int
    terminal_rows: int
    first_step: int | None
    last_step: int | None
    result_counts: dict[str, int]
    action_counts_by_name: dict[str, int]
    missing_terminal: bool


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    """Dataset-level trajectory health summary."""

    inputs: list[str]
    files: int
    rows: int
    training_rows: int
    terminal_rows: int
    empty_files: int
    files_missing_terminal: int
    rows_missing_done: int
    terminal_rows_missing_result: int
    invalid_json_rows: int
    invalid_action_rows: int
    invalid_observation_rows: int
    observation_dim: int
    observation_schema_counts: dict[str, int]
    rows_defaulted_observation_fields: int
    observation_feature_stats: dict[str, dict[str, float]]
    action_names: dict[int, str]
    action_counts: dict[int, int]
    action_counts_by_name: dict[str, int]
    missing_action_names: list[str]
    min_action_count: int
    low_count_action_names: list[str]
    action_coverage: float
    result_counts: dict[str, int]
    rows_per_file: dict[str, float]
    training_rows_per_file: dict[str, float]
    warnings: list[str]
    file_summaries: list[FileTrajectoryDiagnostics]


def diagnose_trajectories(
    paths: TrajectoryPathInput,
    *,
    require_terminal: bool = True,
    min_action_count: int = 10,
) -> TrajectoryDiagnostics:
    """Return health and coverage diagnostics for trajectory JSONL files."""
    if min_action_count < 0:
        raise ValueError("min_action_count must be >= 0")
    input_paths = [str(path) for path in _input_paths(paths)]
    files = discover_trajectory_files(paths)

    total_rows = 0
    training_rows = 0
    terminal_rows = 0
    empty_files = 0
    files_missing_terminal = 0
    rows_missing_done = 0
    terminal_rows_missing_result = 0
    invalid_json_rows = 0
    invalid_action_rows = 0
    invalid_observation_rows = 0
    rows_defaulted_observation_fields = 0
    action_counts: Counter[int] = Counter()
    result_counts: Counter[str] = Counter()
    observation_schema_counts: Counter[str] = Counter()
    observation_feature_values: dict[str, list[float]] = {
        field: [] for field in OBSERVATION_STAT_FIELDS
    }
    file_summaries: list[FileTrajectoryDiagnostics] = []

    for path in files:
        file_rows = 0
        file_training_rows = 0
        file_terminal_rows = 0
        first_step: int | None = None
        last_step: int | None = None
        file_result_counts: Counter[str] = Counter()
        file_action_counts: Counter[int] = Counter()

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json_rows += 1
                    continue

                file_rows += 1
                total_rows += 1

                if "done" not in row:
                    rows_missing_done += 1
                done = bool(row.get("done", False))
                if done:
                    file_terminal_rows += 1
                    terminal_rows += 1
                    result = _result_key(row.get("result"))
                    file_result_counts[result] += 1
                    result_counts[result] += 1
                    if result == "NO_RESULT":
                        terminal_rows_missing_result += 1
                else:
                    file_training_rows += 1
                    training_rows += 1

                step = _optional_int(row.get("step"))
                if step is not None:
                    first_step = step if first_step is None else min(first_step, step)
                    last_step = step if last_step is None else max(last_step, step)

                action = _valid_action(row.get("action"))
                if action is None:
                    invalid_action_rows += 1
                elif not done:
                    action_counts[action] += 1
                    file_action_counts[action] += 1

                observation = row.get("observation")
                if not isinstance(observation, dict):
                    invalid_observation_rows += 1
                else:
                    schema_version = infer_observation_schema_version(observation)
                    try:
                        validate_observation_dict(
                            observation,
                            allow_missing_defaults=True,
                        )
                    except ValueError:
                        invalid_observation_rows += 1
                    else:
                        normalized_observation = normalize_observation_dict(
                            observation,
                            allow_missing_defaults=True,
                        )
                        observation_schema_counts[
                            str(schema_version) if schema_version is not None else "unknown"
                        ] += 1
                        if any(
                            field not in observation
                            for field in OBSERVATION_DEFAULTS
                        ):
                            rows_defaulted_observation_fields += 1
                        for field in OBSERVATION_STAT_FIELDS:
                            observation_feature_values[field].append(
                                float(normalized_observation[field])
                            )

        if file_rows == 0:
            empty_files += 1
        missing_terminal = require_terminal and file_terminal_rows == 0
        if missing_terminal:
            files_missing_terminal += 1

        file_summaries.append(
            FileTrajectoryDiagnostics(
                path=str(path),
                rows=file_rows,
                training_rows=file_training_rows,
                terminal_rows=file_terminal_rows,
                first_step=first_step,
                last_step=last_step,
                result_counts=dict(sorted(file_result_counts.items())),
                action_counts_by_name=_counts_by_name(file_action_counts),
                missing_terminal=missing_terminal,
            )
        )

    missing_action_names = [
        ACTION_NAMES[action_id]
        for action_id in sorted(ACTION_NAMES)
        if action_id not in action_counts
    ]
    low_count_action_names = [
        ACTION_NAMES[action_id]
        for action_id, count in sorted(action_counts.items())
        if 0 < count < min_action_count
    ]

    return TrajectoryDiagnostics(
        inputs=input_paths,
        files=len(files),
        rows=total_rows,
        training_rows=training_rows,
        terminal_rows=terminal_rows,
        empty_files=empty_files,
        files_missing_terminal=files_missing_terminal,
        rows_missing_done=rows_missing_done,
        terminal_rows_missing_result=terminal_rows_missing_result,
        invalid_json_rows=invalid_json_rows,
        invalid_action_rows=invalid_action_rows,
        invalid_observation_rows=invalid_observation_rows,
        observation_dim=len(OBSERVATION_FIELDS),
        observation_schema_counts=dict(sorted(observation_schema_counts.items())),
        rows_defaulted_observation_fields=rows_defaulted_observation_fields,
        observation_feature_stats=_observation_feature_stats(
            observation_feature_values
        ),
        action_names=ACTION_NAMES,
        action_counts=dict(sorted(action_counts.items())),
        action_counts_by_name=_counts_by_name(action_counts),
        missing_action_names=missing_action_names,
        min_action_count=min_action_count,
        low_count_action_names=low_count_action_names,
        action_coverage=(
            len(action_counts) / len(ACTION_NAMES) if ACTION_NAMES else 0.0
        ),
        result_counts=dict(sorted(result_counts.items())),
        rows_per_file=_series_stats(summary.rows for summary in file_summaries),
        training_rows_per_file=_series_stats(
            summary.training_rows for summary in file_summaries
        ),
        warnings=_build_warnings(
            files=len(files),
            training_rows=training_rows,
            empty_files=empty_files,
            files_missing_terminal=files_missing_terminal,
            rows_missing_done=rows_missing_done,
            terminal_rows_missing_result=terminal_rows_missing_result,
            invalid_json_rows=invalid_json_rows,
            invalid_action_rows=invalid_action_rows,
            invalid_observation_rows=invalid_observation_rows,
            rows_defaulted_observation_fields=rows_defaulted_observation_fields,
            missing_action_names=missing_action_names,
            min_action_count=min_action_count,
            low_count_action_names=low_count_action_names,
        ),
        file_summaries=file_summaries,
    )


def _input_paths(paths: TrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    return tuple(paths)


def _valid_action(value: Any) -> int | None:
    try:
        action = int(value)
    except (TypeError, ValueError):
        return None
    if action not in ACTION_NAMES:
        return None
    return action


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _result_key(value: Any) -> str:
    if value is None or value == "":
        return "NO_RESULT"
    return str(value)


def _counts_by_name(counts: Counter[int] | dict[int, int]) -> dict[str, int]:
    return {
        ACTION_NAMES[action_id]: int(count)
        for action_id, count in sorted(counts.items())
        if action_id in ACTION_NAMES
    }


def _series_stats(values: Any) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": min(numbers),
        "max": max(numbers),
        "avg": sum(numbers) / len(numbers),
    }


def _observation_feature_stats(
    values_by_field: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    return {
        field: _series_stats(values)
        for field, values in values_by_field.items()
        if values
    }


def _build_warnings(
    *,
    files: int,
    training_rows: int,
    empty_files: int,
    files_missing_terminal: int,
    rows_missing_done: int,
    terminal_rows_missing_result: int,
    invalid_json_rows: int,
    invalid_action_rows: int,
    invalid_observation_rows: int,
    rows_defaulted_observation_fields: int,
    missing_action_names: list[str],
    min_action_count: int,
    low_count_action_names: list[str],
) -> list[str]:
    warnings: list[str] = []
    if files == 0:
        warnings.append("no trajectory JSONL files discovered")
    if training_rows == 0:
        warnings.append("no non-terminal training rows discovered")
    if empty_files:
        warnings.append(f"{empty_files} trajectory file(s) are empty")
    if files_missing_terminal:
        warnings.append(
            f"{files_missing_terminal} trajectory file(s) have no terminal row"
        )
    if rows_missing_done:
        warnings.append(f"{rows_missing_done} row(s) omit the done field")
    if terminal_rows_missing_result:
        warnings.append(
            f"{terminal_rows_missing_result} terminal row(s) omit result"
        )
    if invalid_json_rows:
        warnings.append(f"{invalid_json_rows} invalid JSON row(s)")
    if invalid_action_rows:
        warnings.append(f"{invalid_action_rows} row(s) contain invalid actions")
    if invalid_observation_rows:
        warnings.append(
            f"{invalid_observation_rows} row(s) contain invalid observations"
        )
    if rows_defaulted_observation_fields:
        warnings.append(
            f"{rows_defaulted_observation_fields} row(s) used current-schema default fields"
        )
    if missing_action_names:
        warnings.append(
            "missing action coverage: " + ", ".join(missing_action_names)
        )
    if low_count_action_names:
        warnings.append(
            f"low action counts (<{min_action_count}): "
            + ", ".join(low_count_action_names)
        )
    return warnings
