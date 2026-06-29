"""Offline agreement diagnostics for strategy teacher and checkpoint policies."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from bot.managers.coverage_strategy_policy import CoverageStrategyPolicy
from rl.normalization import ObservationNormalizer
from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_checkpoints import load_strategy_policy_checkpoint
from rl.strategy_datasets import (
    StrategyTrajectoryPathInput,
    discover_strategy_trajectory_files,
)
from rl.strategy_observations import (
    STRATEGY_OBSERVATION_DEFAULTS,
    STRATEGY_OBSERVATION_FIELDS,
    STRATEGY_OBSERVATION_SCHEMA_VERSION,
    infer_strategy_observation_schema_version,
    normalize_strategy_observation_dict,
    strategy_observation_dict_to_vector,
    validate_strategy_observation_dict,
)


@dataclass(frozen=True)
class StrategyAgreementBucketSummary:
    """Agreement summary for one filtered row bucket."""

    rows: int
    stored_vs_teacher_matches: int
    checkpoint_vs_teacher_matches: int
    checkpoint_vs_stored_matches: int
    stored_vs_teacher_accuracy: float
    checkpoint_vs_teacher_accuracy: float
    checkpoint_vs_stored_accuracy: float
    stored_action_counts_by_name: dict[str, int]
    teacher_action_counts_by_name: dict[str, int]
    checkpoint_action_counts_by_name: dict[str, int]
    mismatch_counts_by_teacher_name: dict[str, int]
    confusion_matrix_teacher_to_checkpoint_by_name: dict[str, dict[str, int]]


@dataclass(frozen=True)
class FileStrategyAgreementSummary:
    """Per-file agreement summary."""

    path: str
    difficulty: str
    opponent_race: str
    rows: int
    first_game_time: float | None
    last_game_time: float | None
    stored_vs_teacher_accuracy: float
    checkpoint_vs_teacher_accuracy: float
    checkpoint_vs_stored_accuracy: float
    stored_action_counts_by_name: dict[str, int]
    teacher_action_counts_by_name: dict[str, int]
    checkpoint_action_counts_by_name: dict[str, int]
    mismatch_counts_by_teacher_name: dict[str, int]
    confusion_matrix_teacher_to_checkpoint_by_name: dict[str, dict[str, int]]


@dataclass(frozen=True)
class StrategyAgreementDiagnostics:
    """Offline policy agreement diagnostics for strategy trajectories."""

    inputs: list[str]
    checkpoint_path: str
    files: int
    rows: int
    observation_schema_counts: dict[str, int]
    rows_defaulted_observation_fields: int
    stored_vs_teacher_matches: int
    checkpoint_vs_teacher_matches: int
    checkpoint_vs_stored_matches: int
    stored_vs_teacher_accuracy: float
    checkpoint_vs_teacher_accuracy: float
    checkpoint_vs_stored_accuracy: float
    stored_action_counts_by_name: dict[str, int]
    teacher_action_counts_by_name: dict[str, int]
    checkpoint_action_counts_by_name: dict[str, int]
    mismatch_counts_by_teacher_name: dict[str, int]
    confusion_matrix_teacher_to_checkpoint_by_name: dict[str, dict[str, int]]
    time_buckets: dict[str, StrategyAgreementBucketSummary]
    state_buckets: dict[str, StrategyAgreementBucketSummary]
    file_summaries: list[FileStrategyAgreementSummary]


@dataclass(frozen=True)
class _AgreementRow:
    path: Path
    step: int
    game_time: float
    stored_action_id: int
    teacher_action_id: int
    checkpoint_action_id: int
    observation: dict[str, float]
    observation_schema_version: str
    defaulted_observation_fields: tuple[str, ...]
    difficulty: str
    opponent_race: str


def diagnose_strategy_agreement(
    paths: StrategyTrajectoryPathInput,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    allow_observation_defaults: bool = True,
) -> StrategyAgreementDiagnostics:
    """Compare stored labels, current coverage teacher, and checkpoint predictions."""
    input_paths = [str(path) for path in _input_paths(paths)]
    files = discover_strategy_trajectory_files(paths)
    loaded = load_strategy_policy_checkpoint(checkpoint_path, map_location=device)
    model = loaded.model.to(device)
    model.eval()
    normalizer = (
        ObservationNormalizer.from_dict(
            loaded.metadata.normalizer,
            expected_fields=STRATEGY_OBSERVATION_FIELDS,
            expected_schema_version=STRATEGY_OBSERVATION_SCHEMA_VERSION,
        )
        if loaded.metadata.normalizer is not None
        else None
    )
    teacher = CoverageStrategyPolicy()
    rows: list[_AgreementRow] = []
    for path in files:
        rows.extend(
            _iter_agreement_rows(
                path,
                teacher=teacher,
                model=model,
                normalizer=normalizer,
                device=torch.device(device),
                allow_observation_defaults=allow_observation_defaults,
            )
        )

    full_summary = _summarize_bucket(rows)
    return StrategyAgreementDiagnostics(
        inputs=input_paths,
        checkpoint_path=str(checkpoint_path),
        files=len(files),
        rows=len(rows),
        observation_schema_counts=dict(
            sorted(Counter(row.observation_schema_version for row in rows).items())
        ),
        rows_defaulted_observation_fields=sum(
            1 for row in rows if row.defaulted_observation_fields
        ),
        stored_vs_teacher_matches=full_summary.stored_vs_teacher_matches,
        checkpoint_vs_teacher_matches=full_summary.checkpoint_vs_teacher_matches,
        checkpoint_vs_stored_matches=full_summary.checkpoint_vs_stored_matches,
        stored_vs_teacher_accuracy=full_summary.stored_vs_teacher_accuracy,
        checkpoint_vs_teacher_accuracy=full_summary.checkpoint_vs_teacher_accuracy,
        checkpoint_vs_stored_accuracy=full_summary.checkpoint_vs_stored_accuracy,
        stored_action_counts_by_name=full_summary.stored_action_counts_by_name,
        teacher_action_counts_by_name=full_summary.teacher_action_counts_by_name,
        checkpoint_action_counts_by_name=full_summary.checkpoint_action_counts_by_name,
        mismatch_counts_by_teacher_name=full_summary.mismatch_counts_by_teacher_name,
        confusion_matrix_teacher_to_checkpoint_by_name=(
            full_summary.confusion_matrix_teacher_to_checkpoint_by_name
        ),
        time_buckets=_time_bucket_summaries(rows),
        state_buckets=_state_bucket_summaries(rows),
        file_summaries=_file_summaries(rows),
    )


def _iter_agreement_rows(
    path: Path,
    *,
    teacher: CoverageStrategyPolicy,
    model: torch.nn.Module,
    normalizer: ObservationNormalizer | None,
    device: torch.device,
    allow_observation_defaults: bool,
) -> Iterable[_AgreementRow]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("done", False)):
                continue
            raw_observation = row.get("strategy_observation", row.get("observation"))
            if not isinstance(raw_observation, dict):
                continue
            try:
                validate_strategy_observation_dict(
                    raw_observation,
                    allow_missing_defaults=allow_observation_defaults,
                )
            except ValueError:
                continue
            observation = normalize_strategy_observation_dict(
                raw_observation,
                allow_missing_defaults=allow_observation_defaults,
            )
            stored_action = _valid_action(row.get("strategy_action"))
            if stored_action is None:
                continue
            teacher_action = int(teacher.decide_from_observation(observation))
            checkpoint_action = _predict_checkpoint_action(
                model,
                observation,
                normalizer=normalizer,
                device=device,
            )
            step = _optional_int(row.get("step"))
            yield _AgreementRow(
                path=path,
                step=step if step is not None else line_number,
                game_time=float(observation.get("game_time", 0.0)),
                stored_action_id=stored_action,
                teacher_action_id=teacher_action,
                checkpoint_action_id=checkpoint_action,
                observation=observation,
                observation_schema_version=str(
                    infer_strategy_observation_schema_version(raw_observation)
                ),
                defaulted_observation_fields=tuple(
                    field
                    for field in STRATEGY_OBSERVATION_DEFAULTS
                    if field not in raw_observation
                ),
                difficulty=str(row.get("difficulty", "")),
                opponent_race=str(row.get("opponent_race", "")),
            )


@torch.no_grad()
def _predict_checkpoint_action(
    model: torch.nn.Module,
    observation: dict[str, float],
    *,
    normalizer: ObservationNormalizer | None,
    device: torch.device,
) -> int:
    vector = strategy_observation_dict_to_vector(observation)
    if normalizer is not None:
        vector = normalizer.transform(vector)
    tensor = torch.from_numpy(np.asarray(vector, dtype=np.float32)).to(device)
    return int(model.predict_action(tensor))


def _summarize_bucket(
    rows: Iterable[_AgreementRow],
) -> StrategyAgreementBucketSummary:
    row_list = list(rows)
    stored_counts: Counter[int] = Counter()
    teacher_counts: Counter[int] = Counter()
    checkpoint_counts: Counter[int] = Counter()
    mismatch_by_teacher: Counter[int] = Counter()
    confusion: dict[int, Counter[int]] = defaultdict(Counter)
    stored_teacher_matches = 0
    checkpoint_teacher_matches = 0
    checkpoint_stored_matches = 0

    for row in row_list:
        stored_counts[row.stored_action_id] += 1
        teacher_counts[row.teacher_action_id] += 1
        checkpoint_counts[row.checkpoint_action_id] += 1
        confusion[row.teacher_action_id][row.checkpoint_action_id] += 1
        if row.stored_action_id == row.teacher_action_id:
            stored_teacher_matches += 1
        if row.checkpoint_action_id == row.teacher_action_id:
            checkpoint_teacher_matches += 1
        else:
            mismatch_by_teacher[row.teacher_action_id] += 1
        if row.checkpoint_action_id == row.stored_action_id:
            checkpoint_stored_matches += 1

    total = len(row_list)
    return StrategyAgreementBucketSummary(
        rows=total,
        stored_vs_teacher_matches=stored_teacher_matches,
        checkpoint_vs_teacher_matches=checkpoint_teacher_matches,
        checkpoint_vs_stored_matches=checkpoint_stored_matches,
        stored_vs_teacher_accuracy=_accuracy(stored_teacher_matches, total),
        checkpoint_vs_teacher_accuracy=_accuracy(checkpoint_teacher_matches, total),
        checkpoint_vs_stored_accuracy=_accuracy(checkpoint_stored_matches, total),
        stored_action_counts_by_name=_counts_by_name(stored_counts),
        teacher_action_counts_by_name=_counts_by_name(teacher_counts),
        checkpoint_action_counts_by_name=_counts_by_name(checkpoint_counts),
        mismatch_counts_by_teacher_name=_counts_by_name(mismatch_by_teacher),
        confusion_matrix_teacher_to_checkpoint_by_name=_confusion_by_name(confusion),
    )


def _time_bucket_summaries(
    rows: list[_AgreementRow],
) -> dict[str, StrategyAgreementBucketSummary]:
    buckets: dict[str, list[_AgreementRow]] = defaultdict(list)
    for row in rows:
        buckets[_time_bucket(row.game_time)].append(row)
    return {
        bucket: _summarize_bucket(bucket_rows)
        for bucket, bucket_rows in sorted(buckets.items(), key=lambda item: item[0])
    }


def _state_bucket_summaries(
    rows: list[_AgreementRow],
) -> dict[str, StrategyAgreementBucketSummary]:
    buckets: dict[str, list[_AgreementRow]] = defaultdict(list)
    for row in rows:
        for bucket in _state_buckets(row):
            buckets[bucket].append(row)
    return {
        bucket: _summarize_bucket(bucket_rows)
        for bucket, bucket_rows in sorted(buckets.items())
    }


def _file_summaries(rows: list[_AgreementRow]) -> list[FileStrategyAgreementSummary]:
    by_path: dict[Path, list[_AgreementRow]] = defaultdict(list)
    for row in rows:
        by_path[row.path].append(row)
    summaries: list[FileStrategyAgreementSummary] = []
    for path, file_rows in sorted(by_path.items(), key=lambda item: str(item[0])):
        summary = _summarize_bucket(file_rows)
        first = file_rows[0] if file_rows else None
        last = file_rows[-1] if file_rows else None
        summaries.append(
            FileStrategyAgreementSummary(
                path=str(path),
                difficulty=_first_non_empty(file_rows, "difficulty"),
                opponent_race=_first_non_empty(file_rows, "opponent_race"),
                rows=len(file_rows),
                first_game_time=first.game_time if first is not None else None,
                last_game_time=last.game_time if last is not None else None,
                stored_vs_teacher_accuracy=summary.stored_vs_teacher_accuracy,
                checkpoint_vs_teacher_accuracy=summary.checkpoint_vs_teacher_accuracy,
                checkpoint_vs_stored_accuracy=summary.checkpoint_vs_stored_accuracy,
                stored_action_counts_by_name=summary.stored_action_counts_by_name,
                teacher_action_counts_by_name=summary.teacher_action_counts_by_name,
                checkpoint_action_counts_by_name=summary.checkpoint_action_counts_by_name,
                mismatch_counts_by_teacher_name=summary.mismatch_counts_by_teacher_name,
                confusion_matrix_teacher_to_checkpoint_by_name=(
                    summary.confusion_matrix_teacher_to_checkpoint_by_name
                ),
            )
        )
    return summaries


def _state_buckets(row: _AgreementRow) -> list[str]:
    observation = row.observation
    buckets: list[str] = []
    if row.difficulty:
        buckets.append(f"difficulty:{row.difficulty}")
    if row.opponent_race:
        buckets.append(f"opponent:{row.opponent_race}")
    if _value(observation, "base_under_threat") > 0.0:
        buckets.append("base_under_threat")
    if _value(observation, "enemy_armored_units_known") > 0.0:
        buckets.append("armored_signal")
    if _value(observation, "enemy_cloaked_units_seen") > 0.0:
        buckets.append("cloaked_signal")
    if _value(observation, "worker_saturation_ratio") < 0.75:
        buckets.append("low_worker_saturation")
    if _gateway_scaling_needed(observation):
        buckets.append("gateway_scaling_needed")
    if _tech_robo_needed(observation):
        buckets.append("tech_robo_needed")
    if _value(observation, "pending_bases") > 0.0:
        buckets.append("pending_bases")
    if _value(observation, "pending_gateways") > 0.0:
        buckets.append("pending_gateways")
    if _value(observation, "pending_robo") > 0.0:
        buckets.append("pending_robo")
    if (
        _value(observation, "pending_forge") > 0.0
        or _value(observation, "ground_weapon_upgrade_pending") > 0.0
        or _value(observation, "ground_armor_upgrade_pending") > 0.0
    ):
        buckets.append("pending_forge_or_upgrade")
    if _value(observation, "pending_static_defense") > 0.0:
        buckets.append("pending_static_defense")
    return buckets


def _gateway_scaling_needed(observation: dict[str, float]) -> bool:
    bases = max(_value(observation, "own_bases"), 1.0)
    gateways = _value(observation, "ready_gateways") + _value(
        observation,
        "pending_gateways",
    )
    return gateways < bases * 4.0 and _value(observation, "minerals") >= 150.0


def _tech_robo_needed(observation: dict[str, float]) -> bool:
    robo = _value(observation, "ready_robo") + _value(observation, "pending_robo")
    return (
        _value(observation, "has_cybernetics_core") > 0.0
        and robo <= 0.0
        and _value(observation, "vespene") >= 100.0
        and (
            _value(observation, "enemy_armored_units_known") > 0.0
            or _value(observation, "enemy_cloaked_units_seen") > 0.0
            or _value(observation, "game_time") >= 360.0
            or _value(observation, "army_count") >= 10.0
        )
    )


def _time_bucket(game_time: float) -> str:
    if game_time < 180.0:
        return "0-180"
    if game_time < 360.0:
        return "180-360"
    if game_time < 540.0:
        return "360-540"
    if game_time < 720.0:
        return "540-720"
    return "720+"


def _counts_by_name(counts: Counter[int] | dict[int, int]) -> dict[str, int]:
    return {
        STRATEGY_ACTION_NAMES[action_id]: int(count)
        for action_id, count in sorted(counts.items())
        if action_id in STRATEGY_ACTION_NAMES
    }


def _confusion_by_name(
    confusion: dict[int, Counter[int]],
) -> dict[str, dict[str, int]]:
    return {
        STRATEGY_ACTION_NAMES[teacher_action]: _counts_by_name(checkpoint_counts)
        for teacher_action, checkpoint_counts in sorted(confusion.items())
        if teacher_action in STRATEGY_ACTION_NAMES
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


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _value(observation: dict[str, float], field: str) -> float:
    if field not in observation and field in STRATEGY_OBSERVATION_DEFAULTS:
        return float(STRATEGY_OBSERVATION_DEFAULTS[field])
    return float(observation.get(field, 0.0))


def _accuracy(matches: int, total: int) -> float:
    return float(matches) / float(total) if total else 0.0


def _first_non_empty(rows: list[_AgreementRow], attr: str) -> str:
    for row in rows:
        value = getattr(row, attr)
        if value:
            return str(value)
    return ""
