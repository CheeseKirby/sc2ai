"""Power-build strategy/tactic failure diagnostics for strategy trajectories."""
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
    normalize_strategy_observation_dict,
    validate_strategy_observation_dict,
)


ARMY_COUNT_THRESHOLDS: tuple[int, ...] = (8, 12, 16, 20)
MINERAL_BANK_THRESHOLD = 500.0
VESPENE_BANK_THRESHOLD = 300.0
IDLE_GATEWAY_MINERALS_THRESHOLD = 150.0
LOW_WORKER_SATURATION_THRESHOLD = 0.8
STATIC_DEFENSE_MINERALS = 100.0
INITIAL_ROBO_MINERALS = 150.0
INITIAL_ROBO_VESPENE = 100.0
STATIC_DEFENSE_CONTEXT_ORDER: tuple[str, ...] = (
    "no_static_affordable",
    "no_static_mineral_short",
    "pending_static_waiting",
    "pending_static_with_ready",
    "ready_static_low_minerals",
    "ready_static_affordable",
    "other",
)
ROBO_BANKING_CONTEXT_ORDER: tuple[str, ...] = (
    "first_robo_affordable",
    "first_robo_mineral_short",
    "first_robo_vespene_short",
    "first_robo_resource_short",
    "pending_robo_cap",
    "ready_robo_already_exists",
    "base_under_threat",
    "no_cybernetics_core",
    "other",
)


@dataclass(frozen=True)
class PowerActionTimingSummary:
    """Timing stats for one strategy action."""

    count: int
    first_game_time: float | None
    min_game_time: float | None
    max_game_time: float | None
    avg_game_time: float | None


@dataclass(frozen=True)
class PowerActionTimelineSegment:
    """One consecutive strategy-action run in a trajectory file."""

    action_name: str
    start_step: int
    end_step: int
    start_game_time: float
    end_game_time: float
    count: int


@dataclass(frozen=True)
class PowerTacticTimelineSegment:
    """One consecutive tactic run in a trajectory file."""

    tactic_id: str
    tactic_phase: str
    tactic_source: str
    start_step: int
    end_step: int
    start_game_time: float
    end_game_time: float
    count: int


@dataclass(frozen=True)
class PowerFilterChangeSummary:
    """Count for one tactic-filter before/after action change."""

    opponent_ai_build: str
    tactic_id: str
    before_action: str
    after_action: str
    count: int


@dataclass(frozen=True)
class PowerRoboBankingFilterContextSummary:
    """Context counts for TECH_ROBO -> STAY_COURSE tactic-filter changes."""

    opponent_ai_build: str
    tactic_id: str
    context: str
    count: int


@dataclass(frozen=True)
class PowerStaticDefenseFilterContextSummary:
    """Context counts for active-threat BUILD_STATIC_DEFENSE filter changes."""

    opponent_ai_build: str
    tactic_id: str
    after_action: str
    context: str
    count: int


@dataclass(frozen=True)
class PowerSignalTimingSummary:
    """First timings and maxima for Power-specific tech and defense signals."""

    first_tech_robo_action_time: float | None
    first_pending_robo_time: float | None
    first_ready_robo_time: float | None
    first_observer_time: float | None
    first_immortal_time: float | None
    first_pending_forge_time: float | None
    first_ready_forge_time: float | None
    first_ground_upgrade_pending_time: float | None
    first_ground_upgrade_complete_time: float | None
    first_pending_static_defense_time: float | None
    first_ready_static_defense_time: float | None
    first_produce_army_action_time: float | None
    first_base_under_threat_time: float | None
    army_count_first_reached: dict[str, float]
    max_army_count: float
    avg_army_count: float | None
    max_pending_robo: float
    max_ready_robo: float
    max_observers: float
    max_immortals: float
    max_pending_forge: float
    max_ready_forge: float
    max_pending_static_defense: float
    max_ready_static_defense: float


@dataclass(frozen=True)
class PowerEconomySummary:
    """Resource-bank and worker-saturation stats."""

    avg_minerals: float | None
    max_minerals: float
    avg_vespene: float | None
    max_vespene: float
    min_worker_saturation_ratio: float | None
    avg_worker_saturation_ratio: float | None
    max_worker_saturation_ratio: float | None
    mineral_bank_rows_ge_500: int
    vespene_bank_rows_ge_300: int
    dual_bank_rows_ge_500_300: int
    low_worker_saturation_rows_lt_0_8: int


@dataclass(frozen=True)
class PowerGatewaySummary:
    """Gateway capacity and idle-production stats."""

    avg_ready_gateways: float | None
    max_ready_gateways: float
    max_pending_gateways: float
    avg_gateway_idle_count: float | None
    max_gateway_idle_count: float
    gateway_idle_rows: int
    idle_gateway_bank_rows_ge_150_minerals: int
    avg_robo_idle_count: float | None
    max_robo_idle_count: float


@dataclass(frozen=True)
class PowerThreatSummary:
    """Threat-state action stats."""

    threat_rows: int
    air_threat_rows: int
    ground_threat_rows: int
    first_base_under_threat_time: float | None
    threat_action_counts_by_name: dict[str, int]
    produce_army_under_threat_rows: int
    static_defense_under_threat_rows: int
    tech_robo_under_threat_rows: int


@dataclass(frozen=True)
class FilePowerTacticSummary:
    """Per-file Power tactic diagnostic summary."""

    path: str
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    result: str | None
    rows: int
    training_rows: int
    first_step: int | None
    last_step: int | None
    first_game_time: float | None
    last_game_time: float | None
    action_counts_by_name: dict[str, int]
    action_first_game_time_by_name: dict[str, float]
    action_timing_by_name: dict[str, PowerActionTimingSummary]
    before_filter_action_counts_by_name: dict[str, int]
    after_filter_action_counts_by_name: dict[str, int]
    filter_action_delta_by_name: dict[str, int]
    rows_with_tactic_metadata: int
    rows_with_filter_metadata: int
    filter_change_rows: int
    training_rows_with_tactic_metadata: int
    training_rows_with_filter_metadata: int
    training_filter_change_rows: int
    tactic_counts: dict[str, int]
    tactic_phase_counts: dict[str, int]
    tactic_source_counts: dict[str, int]
    filter_changes: list[PowerFilterChangeSummary]
    robo_banking_filter_contexts: list[PowerRoboBankingFilterContextSummary]
    static_defense_filter_contexts: list[PowerStaticDefenseFilterContextSummary]
    signals: PowerSignalTimingSummary
    economy: PowerEconomySummary
    gateways: PowerGatewaySummary
    threat: PowerThreatSummary
    action_timeline: list[PowerActionTimelineSegment]
    tactic_timeline: list[PowerTacticTimelineSegment]


@dataclass(frozen=True)
class PowerTacticDiagnostics:
    """Dataset-level Power tactic diagnostics."""

    inputs: list[str]
    files: int
    rows: int
    training_rows: int
    result_counts: dict[str, int]
    opponent_ai_build_counts: dict[str, int]
    action_counts_by_name: dict[str, int]
    action_timing_by_name: dict[str, PowerActionTimingSummary]
    threat_action_counts_by_name: dict[str, int]
    before_filter_action_counts_by_name: dict[str, int]
    after_filter_action_counts_by_name: dict[str, int]
    filter_action_delta_by_name: dict[str, int]
    rows_with_tactic_metadata: int
    rows_with_filter_metadata: int
    filter_change_rows: int
    training_rows_with_tactic_metadata: int
    training_rows_with_filter_metadata: int
    training_filter_change_rows: int
    tactic_counts: dict[str, int]
    tactic_phase_counts: dict[str, int]
    tactic_source_counts: dict[str, int]
    filter_changes: list[PowerFilterChangeSummary]
    robo_banking_filter_contexts: list[PowerRoboBankingFilterContextSummary]
    static_defense_filter_contexts: list[PowerStaticDefenseFilterContextSummary]
    file_summaries: list[FilePowerTacticSummary]


@dataclass(frozen=True)
class _PowerRow:
    path: Path
    line_number: int
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
    opponent_ai_build: str
    tactic_id: str | None
    tactic_phase: str | None
    tactic_source: str | None
    before_action: str | None
    after_action: str | None


def diagnose_power_tactics(paths: StrategyTrajectoryPathInput) -> PowerTacticDiagnostics:
    """Return Power-specific strategy/tactic diagnostics for JSONL trajectories."""
    input_paths = [str(path) for path in _input_paths(paths)]
    files = discover_strategy_trajectory_files(paths)
    rows_total = 0
    training_rows_total = 0
    result_counts: Counter[str] = Counter()
    opponent_ai_build_counts: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    action_times: dict[int, list[float]] = defaultdict(list)
    threat_action_counts: Counter[int] = Counter()
    before_filter_counts: Counter[str] = Counter()
    after_filter_counts: Counter[str] = Counter()
    tactic_counts: Counter[str] = Counter()
    tactic_phase_counts: Counter[str] = Counter()
    tactic_source_counts: Counter[str] = Counter()
    filter_change_counts: Counter[tuple[str, str, str, str]] = Counter()
    robo_banking_context_counts: Counter[tuple[str, str, str]] = Counter()
    static_defense_context_counts: Counter[tuple[str, str, str, str]] = Counter()
    rows_with_tactic_metadata = 0
    rows_with_filter_metadata = 0
    filter_change_rows = 0
    training_rows_with_tactic_metadata = 0
    training_rows_with_filter_metadata = 0
    training_filter_change_rows = 0
    file_summaries: list[FilePowerTacticSummary] = []

    for path in files:
        file_rows = list(_iter_valid_power_rows(path))
        rows_total += len(file_rows)
        training_rows = [row for row in file_rows if not row.done]
        training_rows_total += len(training_rows)

        for row in file_rows:
            if row.done:
                result_counts[_result_key(row.result)] += 1
            if row.tactic_id:
                rows_with_tactic_metadata += 1
            if _has_filter_metadata(row):
                rows_with_filter_metadata += 1
                if row.before_action != row.after_action:
                    filter_change_rows += 1

        for row in training_rows:
            build = _metadata_key(row.opponent_ai_build, default="RandomBuild")
            opponent_ai_build_counts[build] += 1
            action_counts[row.action_id] += 1
            action_times[row.action_id].append(row.game_time)
            if _value(row, "base_under_threat") > 0.0:
                threat_action_counts[row.action_id] += 1

            if row.tactic_id:
                training_rows_with_tactic_metadata += 1
                tactic_counts[_metadata_key(row.tactic_id)] += 1
            if row.tactic_phase:
                tactic_phase_counts[_metadata_key(row.tactic_phase)] += 1
            if row.tactic_source:
                tactic_source_counts[_metadata_key(row.tactic_source)] += 1

            if _has_filter_metadata(row):
                training_rows_with_filter_metadata += 1
                before_filter_counts[str(row.before_action)] += 1
                after_filter_counts[str(row.after_action)] += 1
                if row.before_action != row.after_action:
                    training_filter_change_rows += 1
                    key = (
                        build,
                        _metadata_key(row.tactic_id),
                        str(row.before_action),
                        str(row.after_action),
                    )
                    filter_change_counts[key] += 1
                robo_banking_context = _robo_banking_filter_context(row)
                if robo_banking_context is not None:
                    robo_banking_context_counts[
                        (
                            build,
                            _metadata_key(row.tactic_id),
                            robo_banking_context,
                        )
                    ] += 1
                static_defense_context = _static_defense_filter_context(row)
                if static_defense_context is not None:
                    static_defense_context_counts[
                        (
                            build,
                            _metadata_key(row.tactic_id),
                            str(row.after_action),
                            static_defense_context,
                        )
                    ] += 1

        file_summaries.append(_summarize_file(path, file_rows, training_rows))

    return PowerTacticDiagnostics(
        inputs=input_paths,
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        result_counts=dict(sorted(result_counts.items())),
        opponent_ai_build_counts=_sorted_counts(opponent_ai_build_counts),
        action_counts_by_name=_counts_by_action_name(action_counts),
        action_timing_by_name=_timing_by_action_name(action_times),
        threat_action_counts_by_name=_counts_by_action_name(threat_action_counts),
        before_filter_action_counts_by_name=_sorted_counts(before_filter_counts),
        after_filter_action_counts_by_name=_sorted_counts(after_filter_counts),
        filter_action_delta_by_name=_action_delta(before_filter_counts, after_filter_counts),
        rows_with_tactic_metadata=rows_with_tactic_metadata,
        rows_with_filter_metadata=rows_with_filter_metadata,
        filter_change_rows=filter_change_rows,
        training_rows_with_tactic_metadata=training_rows_with_tactic_metadata,
        training_rows_with_filter_metadata=training_rows_with_filter_metadata,
        training_filter_change_rows=training_filter_change_rows,
        tactic_counts=_sorted_counts(tactic_counts),
        tactic_phase_counts=_sorted_counts(tactic_phase_counts),
        tactic_source_counts=_sorted_counts(tactic_source_counts),
        filter_changes=_filter_change_summaries(filter_change_counts),
        robo_banking_filter_contexts=_robo_banking_context_summaries(
            robo_banking_context_counts,
        ),
        static_defense_filter_contexts=_static_defense_context_summaries(
            static_defense_context_counts,
        ),
        file_summaries=file_summaries,
    )


def _summarize_file(
    path: Path,
    rows: list[_PowerRow],
    training_rows: list[_PowerRow],
) -> FilePowerTacticSummary:
    action_counts = Counter(row.action_id for row in training_rows)
    action_times: dict[int, list[float]] = defaultdict(list)
    action_first_times: dict[str, float] = {}
    before_filter_counts: Counter[str] = Counter()
    after_filter_counts: Counter[str] = Counter()
    tactic_counts: Counter[str] = Counter()
    tactic_phase_counts: Counter[str] = Counter()
    tactic_source_counts: Counter[str] = Counter()
    filter_change_counts: Counter[tuple[str, str, str, str]] = Counter()
    robo_banking_context_counts: Counter[tuple[str, str, str]] = Counter()
    static_defense_context_counts: Counter[tuple[str, str, str, str]] = Counter()
    file_rows_with_tactic = 0
    file_rows_with_filter = 0
    file_filter_change_rows = 0

    for row in rows:
        if row.tactic_id:
            file_rows_with_tactic += 1
        if _has_filter_metadata(row):
            file_rows_with_filter += 1
            if row.before_action != row.after_action:
                file_filter_change_rows += 1

    training_rows_with_tactic = 0
    training_rows_with_filter = 0
    training_filter_change_rows = 0
    for row in training_rows:
        action_times[row.action_id].append(row.game_time)
        action_first_times.setdefault(row.action_name, row.game_time)
        if row.tactic_id:
            training_rows_with_tactic += 1
            tactic_counts[_metadata_key(row.tactic_id)] += 1
        if row.tactic_phase:
            tactic_phase_counts[_metadata_key(row.tactic_phase)] += 1
        if row.tactic_source:
            tactic_source_counts[_metadata_key(row.tactic_source)] += 1
        if _has_filter_metadata(row):
            training_rows_with_filter += 1
            before_filter_counts[str(row.before_action)] += 1
            after_filter_counts[str(row.after_action)] += 1
            if row.before_action != row.after_action:
                training_filter_change_rows += 1
                key = (
                    _metadata_key(row.opponent_ai_build, default="RandomBuild"),
                    _metadata_key(row.tactic_id),
                    str(row.before_action),
                    str(row.after_action),
                )
                filter_change_counts[key] += 1
            robo_banking_context = _robo_banking_filter_context(row)
            if robo_banking_context is not None:
                robo_banking_context_counts[
                    (
                        _metadata_key(row.opponent_ai_build, default="RandomBuild"),
                        _metadata_key(row.tactic_id),
                        robo_banking_context,
                    )
                ] += 1
            static_defense_context = _static_defense_filter_context(row)
            if static_defense_context is not None:
                static_defense_context_counts[
                    (
                        _metadata_key(row.opponent_ai_build, default="RandomBuild"),
                        _metadata_key(row.tactic_id),
                        str(row.after_action),
                        static_defense_context,
                    )
                ] += 1

    first = rows[0] if rows else None
    last = rows[-1] if rows else None
    return FilePowerTacticSummary(
        path=str(path),
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
        first_step=first.step if first is not None else None,
        last_step=last.step if last is not None else None,
        first_game_time=first.game_time if first is not None else None,
        last_game_time=last.game_time if last is not None else None,
        action_counts_by_name=_counts_by_action_name(action_counts),
        action_first_game_time_by_name=dict(sorted(action_first_times.items())),
        action_timing_by_name=_timing_by_action_name(action_times),
        before_filter_action_counts_by_name=_sorted_counts(before_filter_counts),
        after_filter_action_counts_by_name=_sorted_counts(after_filter_counts),
        filter_action_delta_by_name=_action_delta(before_filter_counts, after_filter_counts),
        rows_with_tactic_metadata=file_rows_with_tactic,
        rows_with_filter_metadata=file_rows_with_filter,
        filter_change_rows=file_filter_change_rows,
        training_rows_with_tactic_metadata=training_rows_with_tactic,
        training_rows_with_filter_metadata=training_rows_with_filter,
        training_filter_change_rows=training_filter_change_rows,
        tactic_counts=_sorted_counts(tactic_counts),
        tactic_phase_counts=_sorted_counts(tactic_phase_counts),
        tactic_source_counts=_sorted_counts(tactic_source_counts),
        filter_changes=_filter_change_summaries(filter_change_counts),
        robo_banking_filter_contexts=_robo_banking_context_summaries(
            robo_banking_context_counts,
        ),
        static_defense_filter_contexts=_static_defense_context_summaries(
            static_defense_context_counts,
        ),
        signals=_signal_summary(training_rows),
        economy=_economy_summary(training_rows),
        gateways=_gateway_summary(training_rows),
        threat=_threat_summary(training_rows),
        action_timeline=_action_timeline_segments(training_rows),
        tactic_timeline=_tactic_timeline_segments(training_rows),
    )


def _iter_valid_power_rows(path: Path) -> Iterable[_PowerRow]:
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
            step = _optional_int(row.get("step"), default=line_number)
            yield _PowerRow(
                path=path,
                line_number=line_number,
                step=int(step),
                game_time=float(normalized.get("game_time", 0.0)),
                action_id=action,
                action_name=STRATEGY_ACTION_NAMES[action],
                done=bool(row.get("done", False)),
                result=_optional_str(row.get("result")),
                observation=normalized,
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
            )


def _signal_summary(rows: list[_PowerRow]) -> PowerSignalTimingSummary:
    return PowerSignalTimingSummary(
        first_tech_robo_action_time=_first_time(
            rows,
            lambda row: row.action_name == "TECH_ROBO",
        ),
        first_pending_robo_time=_first_time(rows, lambda row: _value(row, "pending_robo") > 0.0),
        first_ready_robo_time=_first_time(rows, lambda row: _value(row, "ready_robo") > 0.0),
        first_observer_time=_first_time(rows, lambda row: _value(row, "observers") > 0.0),
        first_immortal_time=_first_time(rows, lambda row: _value(row, "immortals") > 0.0),
        first_pending_forge_time=_first_time(rows, lambda row: _value(row, "pending_forge") > 0.0),
        first_ready_forge_time=_first_time(rows, lambda row: _value(row, "ready_forge") > 0.0),
        first_ground_upgrade_pending_time=_first_time(
            rows,
            lambda row: _value(row, "ground_weapon_upgrade_pending") > 0.0
            or _value(row, "ground_armor_upgrade_pending") > 0.0,
        ),
        first_ground_upgrade_complete_time=_first_time(
            rows,
            lambda row: _value(row, "ground_weapon_level") > 0.0
            or _value(row, "ground_armor_level") > 0.0,
        ),
        first_pending_static_defense_time=_first_time(
            rows,
            lambda row: _value(row, "pending_static_defense") > 0.0,
        ),
        first_ready_static_defense_time=_first_time(
            rows,
            lambda row: _value(row, "ready_static_defense") > 0.0,
        ),
        first_produce_army_action_time=_first_time(
            rows,
            lambda row: row.action_name == "PRODUCE_ARMY",
        ),
        first_base_under_threat_time=_first_time(
            rows,
            lambda row: _value(row, "base_under_threat") > 0.0,
        ),
        army_count_first_reached=_army_threshold_times(rows),
        max_army_count=_max_value(rows, "army_count"),
        avg_army_count=_avg_value(rows, "army_count"),
        max_pending_robo=_max_value(rows, "pending_robo"),
        max_ready_robo=_max_value(rows, "ready_robo"),
        max_observers=_max_value(rows, "observers"),
        max_immortals=_max_value(rows, "immortals"),
        max_pending_forge=_max_value(rows, "pending_forge"),
        max_ready_forge=_max_value(rows, "ready_forge"),
        max_pending_static_defense=_max_value(rows, "pending_static_defense"),
        max_ready_static_defense=_max_value(rows, "ready_static_defense"),
    )


def _economy_summary(rows: list[_PowerRow]) -> PowerEconomySummary:
    return PowerEconomySummary(
        avg_minerals=_avg_value(rows, "minerals"),
        max_minerals=_max_value(rows, "minerals"),
        avg_vespene=_avg_value(rows, "vespene"),
        max_vespene=_max_value(rows, "vespene"),
        min_worker_saturation_ratio=_min_value(rows, "worker_saturation_ratio"),
        avg_worker_saturation_ratio=_avg_value(rows, "worker_saturation_ratio"),
        max_worker_saturation_ratio=_max_value_or_none(rows, "worker_saturation_ratio"),
        mineral_bank_rows_ge_500=sum(
            1 for row in rows if _value(row, "minerals") >= MINERAL_BANK_THRESHOLD
        ),
        vespene_bank_rows_ge_300=sum(
            1 for row in rows if _value(row, "vespene") >= VESPENE_BANK_THRESHOLD
        ),
        dual_bank_rows_ge_500_300=sum(
            1
            for row in rows
            if _value(row, "minerals") >= MINERAL_BANK_THRESHOLD
            and _value(row, "vespene") >= VESPENE_BANK_THRESHOLD
        ),
        low_worker_saturation_rows_lt_0_8=sum(
            1
            for row in rows
            if _value(row, "worker_saturation_ratio") < LOW_WORKER_SATURATION_THRESHOLD
        ),
    )


def _gateway_summary(rows: list[_PowerRow]) -> PowerGatewaySummary:
    return PowerGatewaySummary(
        avg_ready_gateways=_avg_value(rows, "ready_gateways"),
        max_ready_gateways=_max_value(rows, "ready_gateways"),
        max_pending_gateways=_max_value(rows, "pending_gateways"),
        avg_gateway_idle_count=_avg_value(rows, "gateway_idle_count"),
        max_gateway_idle_count=_max_value(rows, "gateway_idle_count"),
        gateway_idle_rows=sum(1 for row in rows if _value(row, "gateway_idle_count") > 0.0),
        idle_gateway_bank_rows_ge_150_minerals=sum(
            1
            for row in rows
            if _value(row, "gateway_idle_count") > 0.0
            and _value(row, "minerals") >= IDLE_GATEWAY_MINERALS_THRESHOLD
        ),
        avg_robo_idle_count=_avg_value(rows, "robo_idle_count"),
        max_robo_idle_count=_max_value(rows, "robo_idle_count"),
    )


def _threat_summary(rows: list[_PowerRow]) -> PowerThreatSummary:
    threat_rows = [row for row in rows if _value(row, "base_under_threat") > 0.0]
    threat_action_counts = Counter(row.action_id for row in threat_rows)
    return PowerThreatSummary(
        threat_rows=len(threat_rows),
        air_threat_rows=sum(1 for row in rows if _value(row, "base_under_air_threat") > 0.0),
        ground_threat_rows=sum(
            1 for row in rows if _value(row, "base_under_ground_threat") > 0.0
        ),
        first_base_under_threat_time=_first_time(
            rows,
            lambda row: _value(row, "base_under_threat") > 0.0,
        ),
        threat_action_counts_by_name=_counts_by_action_name(threat_action_counts),
        produce_army_under_threat_rows=sum(
            1 for row in threat_rows if row.action_name == "PRODUCE_ARMY"
        ),
        static_defense_under_threat_rows=sum(
            1 for row in threat_rows if row.action_name == "BUILD_STATIC_DEFENSE"
        ),
        tech_robo_under_threat_rows=sum(
            1 for row in threat_rows if row.action_name == "TECH_ROBO"
        ),
    )


def _action_timeline_segments(rows: list[_PowerRow]) -> list[PowerActionTimelineSegment]:
    if not rows:
        return []
    segments: list[PowerActionTimelineSegment] = []
    start = rows[0]
    previous = rows[0]
    count = 1
    for row in rows[1:]:
        if row.action_id == previous.action_id:
            previous = row
            count += 1
            continue
        segments.append(
            PowerActionTimelineSegment(
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
        PowerActionTimelineSegment(
            action_name=start.action_name,
            start_step=start.step,
            end_step=previous.step,
            start_game_time=start.game_time,
            end_game_time=previous.game_time,
            count=count,
        )
    )
    return segments


def _tactic_timeline_segments(rows: list[_PowerRow]) -> list[PowerTacticTimelineSegment]:
    if not rows:
        return []
    segments: list[PowerTacticTimelineSegment] = []
    start = rows[0]
    previous = rows[0]
    count = 1
    for row in rows[1:]:
        if _same_tactic_segment(row, previous):
            previous = row
            count += 1
            continue
        segments.append(_make_tactic_segment(start, previous, count))
        start = row
        previous = row
        count = 1
    segments.append(_make_tactic_segment(start, previous, count))
    return segments


def _same_tactic_segment(left: _PowerRow, right: _PowerRow) -> bool:
    return (
        _metadata_key(left.tactic_id) == _metadata_key(right.tactic_id)
        and _metadata_key(left.tactic_phase) == _metadata_key(right.tactic_phase)
        and _metadata_key(left.tactic_source) == _metadata_key(right.tactic_source)
    )


def _make_tactic_segment(
    start: _PowerRow,
    end: _PowerRow,
    count: int,
) -> PowerTacticTimelineSegment:
    return PowerTacticTimelineSegment(
        tactic_id=_metadata_key(start.tactic_id),
        tactic_phase=_metadata_key(start.tactic_phase),
        tactic_source=_metadata_key(start.tactic_source),
        start_step=start.step,
        end_step=end.step,
        start_game_time=start.game_time,
        end_game_time=end.game_time,
        count=count,
    )


def _filter_change_summaries(
    counts: Counter[tuple[str, str, str, str]],
) -> list[PowerFilterChangeSummary]:
    summaries = [
        PowerFilterChangeSummary(
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


def _robo_banking_context_summaries(
    counts: Counter[tuple[str, str, str]],
) -> list[PowerRoboBankingFilterContextSummary]:
    summaries = [
        PowerRoboBankingFilterContextSummary(
            opponent_ai_build=build,
            tactic_id=tactic_id,
            context=context,
            count=int(count),
        )
        for (build, tactic_id, context), count in counts.items()
    ]
    return sorted(
        summaries,
        key=lambda item: (
            -item.count,
            item.opponent_ai_build,
            _context_sort_index(item.context),
            item.tactic_id,
            item.context,
        ),
    )


def _static_defense_context_summaries(
    counts: Counter[tuple[str, str, str, str]],
) -> list[PowerStaticDefenseFilterContextSummary]:
    summaries = [
        PowerStaticDefenseFilterContextSummary(
            opponent_ai_build=build,
            tactic_id=tactic_id,
            after_action=after_action,
            context=context,
            count=int(count),
        )
        for (build, tactic_id, after_action, context), count in counts.items()
    ]
    return sorted(
        summaries,
        key=lambda item: (
            -item.count,
            item.opponent_ai_build,
            item.tactic_id,
            _static_context_sort_index(item.context),
            item.after_action,
            item.context,
        ),
    )


def _robo_banking_filter_context(row: _PowerRow) -> str | None:
    if row.before_action != "TECH_ROBO" or row.after_action != "STAY_COURSE":
        return None
    if _value(row, "ready_robo") > 0.0:
        return "ready_robo_already_exists"
    if _value(row, "pending_robo") > 0.0:
        return "pending_robo_cap"
    if _value(row, "base_under_threat") > 0.0:
        return "base_under_threat"
    if _value(row, "has_cybernetics_core") <= 0.0:
        return "no_cybernetics_core"

    has_minerals = _value(row, "minerals") >= INITIAL_ROBO_MINERALS
    has_vespene = _value(row, "vespene") >= INITIAL_ROBO_VESPENE
    if has_minerals and has_vespene:
        return "first_robo_affordable"
    if not has_minerals and has_vespene:
        return "first_robo_mineral_short"
    if has_minerals and not has_vespene:
        return "first_robo_vespene_short"
    if not has_minerals and not has_vespene:
        return "first_robo_resource_short"
    return "other"


def _static_defense_filter_context(row: _PowerRow) -> str | None:
    if row.before_action != "BUILD_STATIC_DEFENSE":
        return None
    if row.after_action == "BUILD_STATIC_DEFENSE":
        return None
    if _value(row, "base_under_threat") <= 0.0:
        return None
    ready_static = _value(row, "ready_static_defense") > 0.0
    if _value(row, "pending_static_defense") > 0.0:
        if ready_static:
            return "pending_static_with_ready"
        return "pending_static_waiting"
    has_minerals = _value(row, "minerals") >= STATIC_DEFENSE_MINERALS
    if ready_static and has_minerals:
        return "ready_static_affordable"
    if ready_static:
        return "ready_static_low_minerals"
    if has_minerals:
        return "no_static_affordable"
    return "no_static_mineral_short"


def _context_sort_index(context: str) -> int:
    try:
        return ROBO_BANKING_CONTEXT_ORDER.index(context)
    except ValueError:
        return len(ROBO_BANKING_CONTEXT_ORDER)


def _static_context_sort_index(context: str) -> int:
    try:
        return STATIC_DEFENSE_CONTEXT_ORDER.index(context)
    except ValueError:
        return len(STATIC_DEFENSE_CONTEXT_ORDER)


def _timing_by_action_name(
    action_times: dict[int, list[float]],
) -> dict[str, PowerActionTimingSummary]:
    return {
        STRATEGY_ACTION_NAMES[action_id]: _action_timing(times)
        for action_id, times in sorted(action_times.items())
        if action_id in STRATEGY_ACTION_NAMES
    }


def _action_timing(times: list[float]) -> PowerActionTimingSummary:
    return PowerActionTimingSummary(
        count=len(times),
        first_game_time=times[0] if times else None,
        min_game_time=min(times) if times else None,
        max_game_time=max(times) if times else None,
        avg_game_time=(sum(times) / len(times)) if times else None,
    )


def _army_threshold_times(rows: list[_PowerRow]) -> dict[str, float]:
    reached: dict[str, float] = {}
    for threshold in ARMY_COUNT_THRESHOLDS:
        for row in rows:
            if _value(row, "army_count") >= float(threshold):
                reached[f">={threshold}"] = row.game_time
                break
    return reached


def _first_time(rows: list[_PowerRow], predicate: Any) -> float | None:
    for row in rows:
        if predicate(row):
            return row.game_time
    return None


def _min_value(rows: list[_PowerRow], field: str) -> float | None:
    if not rows:
        return None
    return min(_value(row, field) for row in rows)


def _max_value(rows: list[_PowerRow], field: str) -> float:
    if not rows:
        return 0.0
    return max(_value(row, field) for row in rows)


def _max_value_or_none(rows: list[_PowerRow], field: str) -> float | None:
    if not rows:
        return None
    return _max_value(rows, field)


def _avg_value(rows: list[_PowerRow], field: str) -> float | None:
    if not rows:
        return None
    return sum(_value(row, field) for row in rows) / len(rows)


def _value(row: _PowerRow, field: str) -> float:
    if field not in row.observation and field in STRATEGY_OBSERVATION_DEFAULTS:
        return float(STRATEGY_OBSERVATION_DEFAULTS[field])
    return float(row.observation.get(field, 0.0))


def _has_filter_metadata(row: _PowerRow) -> bool:
    return row.before_action is not None and row.after_action is not None


def _strategy_action_name(row: dict[str, Any], *, id_key: str, name_key: str) -> str | None:
    name = _optional_str(row.get(name_key))
    if name is not None:
        return name
    action_id = _optional_int(row.get(id_key), default=None)
    if action_id is None:
        return None
    return STRATEGY_ACTION_NAMES.get(action_id)


def _valid_action(value: Any) -> int | None:
    try:
        action = int(value)
    except (TypeError, ValueError):
        return None
    if action not in STRATEGY_ACTION_NAMES:
        return None
    return action


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


def _first_non_empty(
    rows: list[_PowerRow],
    attr: str,
    *,
    default: str = "",
) -> str:
    for row in rows:
        value = getattr(row, attr)
        if value:
            return str(value)
    return default


def _counts_by_action_name(counts: Counter[int] | dict[int, int]) -> dict[str, int]:
    return {
        STRATEGY_ACTION_NAMES[action_id]: int(count)
        for action_id, count in sorted(counts.items())
        if action_id in STRATEGY_ACTION_NAMES
    }


def _sorted_counts(counts: Counter[str]) -> dict[str, int]:
    return dict(sorted((str(key), int(count)) for key, count in counts.items()))


def _action_delta(
    before_counts: Counter[str],
    after_counts: Counter[str],
) -> dict[str, int]:
    actions = sorted(set(before_counts) | set(after_counts))
    return {
        action: int(after_counts.get(action, 0) - before_counts.get(action, 0))
        for action in actions
        if after_counts.get(action, 0) != before_counts.get(action, 0)
    }


def _metadata_key(value: str | None, *, default: str = "<none>") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _optional_int(value: Any, *, default: int | None) -> int | None:
    try:
        return int(value)
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
