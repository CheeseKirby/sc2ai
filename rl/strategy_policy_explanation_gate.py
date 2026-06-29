"""Raw strategy trajectory gate for policy explanation metadata."""
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


@dataclass(frozen=True)
class StrategyPolicyExplanationGateConfig:
    """Fail-closed thresholds for strategy policy explanation coverage."""

    min_rows: int = 1
    min_policy_source_ratio: float = 1.0
    min_policy_reason_ratio: float = 1.0


@dataclass(frozen=True)
class StrategyPolicyExplanationGateResult:
    """Readiness decision for raw strategy policy explanations."""

    recommendation: str
    ready: bool
    inputs: list[str]
    files: int
    rows: int
    blocking_reasons: list[str]
    policy_source_rows: int
    policy_source_ratio: float
    policy_reason_rows: int
    policy_reason_ratio: float
    missing_policy_source_rows: int
    missing_policy_reason_rows: int
    policy_source_counts: dict[str, int]
    policy_reason_counts: dict[str, int]
    config: StrategyPolicyExplanationGateConfig


def evaluate_strategy_policy_explanation_gate(
    paths: StrategyTrajectoryPathInput,
    *,
    config: StrategyPolicyExplanationGateConfig | None = None,
) -> StrategyPolicyExplanationGateResult:
    """Evaluate whether raw strategy trajectories carry policy explanations."""
    gate_config = config or StrategyPolicyExplanationGateConfig()
    inputs = [str(path) for path in _input_paths(paths)]
    files = discover_strategy_trajectory_files(paths)

    rows = 0
    source_rows = 0
    reason_rows = 0
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for path in files:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if bool(row.get("done", False)):
                    continue
                rows += 1
                source = _nonblank_string(row.get("strategy_policy_source"))
                reason = _nonblank_string(row.get("strategy_policy_reason"))
                if source is not None:
                    source_rows += 1
                    source_counts[source] += 1
                if reason is not None:
                    reason_rows += 1
                    reason_counts[reason] += 1

    source_ratio = _ratio(source_rows, rows)
    reason_ratio = _ratio(reason_rows, rows)
    blocking_reasons = _blocking_reasons(
        rows=rows,
        policy_source_ratio=source_ratio,
        policy_reason_ratio=reason_ratio,
        config=gate_config,
    )
    ready = not blocking_reasons
    return StrategyPolicyExplanationGateResult(
        recommendation="ready" if ready else "hold",
        ready=ready,
        inputs=inputs,
        files=len(files),
        rows=rows,
        blocking_reasons=blocking_reasons,
        policy_source_rows=source_rows,
        policy_source_ratio=source_ratio,
        policy_reason_rows=reason_rows,
        policy_reason_ratio=reason_ratio,
        missing_policy_source_rows=rows - source_rows,
        missing_policy_reason_rows=rows - reason_rows,
        policy_source_counts=dict(sorted(source_counts.items())),
        policy_reason_counts=dict(sorted(reason_counts.items())),
        config=gate_config,
    )


def _blocking_reasons(
    *,
    rows: int,
    policy_source_ratio: float,
    policy_reason_ratio: float,
    config: StrategyPolicyExplanationGateConfig,
) -> list[str]:
    reasons: list[str] = []
    if rows < config.min_rows:
        reasons.append("insufficient_rows")
    if policy_source_ratio < config.min_policy_source_ratio:
        reasons.append("policy_source_coverage_low")
    if policy_reason_ratio < config.min_policy_reason_ratio:
        reasons.append("policy_reason_coverage_low")
    return reasons


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _nonblank_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
