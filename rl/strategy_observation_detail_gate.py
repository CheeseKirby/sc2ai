"""Readiness gate for strategy observation detail coverage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.experiments import read_json


@dataclass(frozen=True)
class StrategyObservationDetailGateConfig:
    """Fail-closed thresholds for observation detail readiness."""

    min_rows: int = 1
    min_observation_detail_ratio: float = 1.0
    min_threatened_only_stay_course_detail_ratio: float = 1.0
    min_air_threat_only_stay_course_detail_ratio: float = 1.0
    max_static_defense_type_ambiguous_rows: int = 0


@dataclass(frozen=True)
class StrategyObservationDetailGateResult:
    """Readiness decision for one emergency-action analysis artifact."""

    recommendation: str
    ready: bool
    analysis_path: str
    analysis_inputs: list[str]
    blocking_reasons: list[str]
    rows: int
    observation_detail_rows: int
    observation_detail_ratio: float
    threatened_only_stay_course_rows: int
    threatened_only_stay_course_detail_rows: int
    threatened_only_stay_course_detail_ratio: float
    air_threat_only_stay_course_rows: int
    air_threat_only_stay_course_detail_rows: int
    air_threat_only_stay_course_detail_ratio: float
    static_defense_type_ambiguous_rows: int
    config: StrategyObservationDetailGateConfig


def evaluate_strategy_observation_detail_gate(
    analysis_path: str | Path,
    *,
    config: StrategyObservationDetailGateConfig | None = None,
) -> StrategyObservationDetailGateResult:
    """Evaluate whether emergency analysis data has enough detail coverage."""
    gate_config = config or StrategyObservationDetailGateConfig()
    path = Path(analysis_path)
    analysis = read_json(path)
    return evaluate_strategy_observation_detail_payload(
        analysis,
        analysis_path=str(path),
        config=gate_config,
    )


def evaluate_strategy_observation_detail_payload(
    analysis: dict[str, Any],
    *,
    analysis_path: str,
    config: StrategyObservationDetailGateConfig,
) -> StrategyObservationDetailGateResult:
    """Evaluate one loaded emergency-action analysis payload."""
    analysis_inputs = _analysis_inputs(analysis)
    rows = _int(analysis, "rows")
    observation_detail_rows = _int(analysis, "observation_detail_rows")
    observation_detail_ratio = _float(analysis, "observation_detail_ratio")
    threatened_rows = _int(analysis, "threatened_only_stay_course_rows")
    threatened_detail_rows = _int(
        analysis,
        "threatened_only_stay_course_detail_rows",
    )
    threatened_detail_ratio = _float(
        analysis,
        "threatened_only_stay_course_detail_ratio",
    )
    air_threat_rows = _int(analysis, "air_threat_only_stay_course_rows")
    air_threat_detail_rows = _int(
        analysis,
        "air_threat_only_stay_course_detail_rows",
    )
    air_threat_detail_ratio = _float(
        analysis,
        "air_threat_only_stay_course_detail_ratio",
    )
    ambiguous_rows = _static_defense_type_ambiguous_rows(analysis)
    blocking_reasons = _blocking_reasons(
        rows=rows,
        observation_detail_ratio=observation_detail_ratio,
        threatened_rows=threatened_rows,
        threatened_detail_ratio=threatened_detail_ratio,
        air_threat_rows=air_threat_rows,
        air_threat_detail_ratio=air_threat_detail_ratio,
        static_defense_type_ambiguous_rows=ambiguous_rows,
        config=config,
    )
    ready = not blocking_reasons
    return StrategyObservationDetailGateResult(
        recommendation="ready" if ready else "hold",
        ready=ready,
        analysis_path=analysis_path,
        analysis_inputs=analysis_inputs,
        blocking_reasons=blocking_reasons,
        rows=rows,
        observation_detail_rows=observation_detail_rows,
        observation_detail_ratio=observation_detail_ratio,
        threatened_only_stay_course_rows=threatened_rows,
        threatened_only_stay_course_detail_rows=threatened_detail_rows,
        threatened_only_stay_course_detail_ratio=threatened_detail_ratio,
        air_threat_only_stay_course_rows=air_threat_rows,
        air_threat_only_stay_course_detail_rows=air_threat_detail_rows,
        air_threat_only_stay_course_detail_ratio=air_threat_detail_ratio,
        static_defense_type_ambiguous_rows=ambiguous_rows,
        config=config,
    )


def _blocking_reasons(
    *,
    rows: int,
    observation_detail_ratio: float,
    threatened_rows: int,
    threatened_detail_ratio: float,
    air_threat_rows: int,
    air_threat_detail_ratio: float,
    static_defense_type_ambiguous_rows: int,
    config: StrategyObservationDetailGateConfig,
) -> list[str]:
    reasons: list[str] = []
    if rows < config.min_rows:
        reasons.append("insufficient_rows")
    if observation_detail_ratio < config.min_observation_detail_ratio:
        reasons.append("observation_detail_coverage_low")
    if (
        threatened_rows > 0
        and threatened_detail_ratio
        < config.min_threatened_only_stay_course_detail_ratio
    ):
        reasons.append("threatened_only_stay_detail_coverage_low")
    if (
        air_threat_rows > 0
        and air_threat_detail_ratio
        < config.min_air_threat_only_stay_course_detail_ratio
    ):
        reasons.append("air_threat_only_stay_detail_coverage_low")
    if (
        static_defense_type_ambiguous_rows
        > config.max_static_defense_type_ambiguous_rows
    ):
        reasons.append("static_defense_type_ambiguous_rows_high")
    return reasons


def _static_defense_type_ambiguous_rows(analysis: dict[str, Any]) -> int:
    gap_counts = analysis.get("unaddressed_air_defense_gap_by_reason") or {}
    if not isinstance(gap_counts, dict):
        return 0
    return int(gap_counts.get("static_defense_type_ambiguous", 0) or 0)


def _analysis_inputs(analysis: dict[str, Any]) -> list[str]:
    inputs = analysis.get("inputs") or []
    if not isinstance(inputs, list):
        return []
    return [str(value) for value in inputs]


def _int(payload: dict[str, Any], key: str) -> int:
    return int(payload.get(key, 0) or 0)


def _float(payload: dict[str, Any], key: str) -> float:
    return float(payload.get(key, 0.0) or 0.0)
