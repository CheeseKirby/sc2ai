"""Raw strategy trajectory gate for observation detail integrity."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.strategy_datasets import (
    StrategyTrajectoryPathInput,
    discover_strategy_trajectory_files,
)
from rl.strategy_observations import STRATEGY_OBSERVATION_DETAIL_FIELDS


@dataclass(frozen=True)
class StrategyTrajectoryDetailGateConfig:
    """Fail-closed thresholds for raw trajectory detail integrity."""

    min_rows: int = 1
    min_observation_detail_ratio: float = 1.0
    min_observation_detail_complete_ratio: float = 1.0
    max_ready_static_defense_mismatch_rows: int = 0
    max_pending_static_defense_mismatch_rows: int = 0


@dataclass(frozen=True)
class StrategyTrajectoryDetailGateResult:
    """Readiness decision for raw strategy trajectory details."""

    recommendation: str
    ready: bool
    inputs: list[str]
    files: int
    rows: int
    blocking_reasons: list[str]
    observation_detail_rows: int
    observation_detail_ratio: float
    observation_detail_complete_rows: int
    observation_detail_complete_ratio: float
    missing_detail_field_counts: dict[str, int]
    ready_static_defense_mismatch_rows: int
    pending_static_defense_mismatch_rows: int
    config: StrategyTrajectoryDetailGateConfig


def evaluate_strategy_trajectory_detail_gate(
    paths: StrategyTrajectoryPathInput,
    *,
    config: StrategyTrajectoryDetailGateConfig | None = None,
) -> StrategyTrajectoryDetailGateResult:
    """Evaluate whether raw strategy trajectories carry complete detail fields."""
    gate_config = config or StrategyTrajectoryDetailGateConfig()
    inputs = [str(path) for path in _input_paths(paths)]
    files = discover_strategy_trajectory_files(paths)

    rows = 0
    detail_rows = 0
    complete_detail_rows = 0
    missing_fields: Counter[str] = Counter()
    ready_mismatches = 0
    pending_mismatches = 0

    for path in files:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if bool(row.get("done", False)):
                    continue
                rows += 1
                details = row.get("strategy_observation_details")
                if not isinstance(details, dict):
                    continue
                detail_rows += 1
                row_missing = [
                    field
                    for field in STRATEGY_OBSERVATION_DETAIL_FIELDS
                    if field not in details
                ]
                missing_fields.update(row_missing)
                if not row_missing:
                    complete_detail_rows += 1
                ready_mismatches += int(_ready_static_defense_mismatch(row, details))
                pending_mismatches += int(_pending_static_defense_mismatch(row, details))

    detail_ratio = _ratio(detail_rows, rows)
    complete_ratio = _ratio(complete_detail_rows, rows)
    blocking_reasons = _blocking_reasons(
        rows=rows,
        observation_detail_ratio=detail_ratio,
        observation_detail_complete_ratio=complete_ratio,
        ready_static_defense_mismatch_rows=ready_mismatches,
        pending_static_defense_mismatch_rows=pending_mismatches,
        config=gate_config,
    )
    ready = not blocking_reasons
    return StrategyTrajectoryDetailGateResult(
        recommendation="ready" if ready else "hold",
        ready=ready,
        inputs=inputs,
        files=len(files),
        rows=rows,
        blocking_reasons=blocking_reasons,
        observation_detail_rows=detail_rows,
        observation_detail_ratio=detail_ratio,
        observation_detail_complete_rows=complete_detail_rows,
        observation_detail_complete_ratio=complete_ratio,
        missing_detail_field_counts=dict(sorted(missing_fields.items())),
        ready_static_defense_mismatch_rows=ready_mismatches,
        pending_static_defense_mismatch_rows=pending_mismatches,
        config=gate_config,
    )


def _blocking_reasons(
    *,
    rows: int,
    observation_detail_ratio: float,
    observation_detail_complete_ratio: float,
    ready_static_defense_mismatch_rows: int,
    pending_static_defense_mismatch_rows: int,
    config: StrategyTrajectoryDetailGateConfig,
) -> list[str]:
    reasons: list[str] = []
    if rows < config.min_rows:
        reasons.append("insufficient_rows")
    if observation_detail_ratio < config.min_observation_detail_ratio:
        reasons.append("observation_detail_coverage_low")
    if (
        observation_detail_complete_ratio
        < config.min_observation_detail_complete_ratio
    ):
        reasons.append("observation_detail_complete_coverage_low")
    if (
        ready_static_defense_mismatch_rows
        > config.max_ready_static_defense_mismatch_rows
    ):
        reasons.append("ready_static_defense_detail_mismatch")
    if (
        pending_static_defense_mismatch_rows
        > config.max_pending_static_defense_mismatch_rows
    ):
        reasons.append("pending_static_defense_detail_mismatch")
    return reasons


def _ready_static_defense_mismatch(row: dict[str, Any], details: dict[str, Any]) -> bool:
    observation = row.get("strategy_observation") or {}
    if not isinstance(observation, dict) or "ready_static_defense" not in observation:
        return False
    expected = _float(observation, "ready_static_defense")
    actual = _float(details, "ready_photon_cannons") + _float(
        details,
        "ready_shield_batteries",
    )
    return not _same_count(expected, actual)


def _pending_static_defense_mismatch(
    row: dict[str, Any],
    details: dict[str, Any],
) -> bool:
    observation = row.get("strategy_observation") or {}
    if not isinstance(observation, dict) or "pending_static_defense" not in observation:
        return False
    expected = _float(observation, "pending_static_defense")
    actual = _float(details, "pending_photon_cannons") + _float(
        details,
        "pending_shield_batteries",
    )
    return not _same_count(expected, actual)


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _float(payload: dict[str, Any], key: str) -> float:
    return float(payload.get(key, 0.0) or 0.0)


def _same_count(left: float, right: float) -> bool:
    return abs(left - right) < 1e-6


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
