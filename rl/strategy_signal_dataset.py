"""Build row-level training signals from strategy trajectories."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_datasets import StrategyTrajectoryPathInput, discover_strategy_trajectory_files
from rl.strategy_outcome_diagnostics import (
    DEFAULT_LOOKAHEAD_SECONDS,
    _StrategyOutcomeRow,
    _iter_valid_strategy_rows,
    _source_for_file,
    _window_outcome,
    window_key,
)
from rl.strategy_replay_candidate import (
    candidate_executability,
    classify_replay_context,
    classify_threat_state,
)


DEFAULT_SIGNAL_WINDOWS: tuple[float, ...] = DEFAULT_LOOKAHEAD_SECONDS
PAYOFF_EVENTS_BY_ACTION: dict[str, tuple[str, ...]] = {
    "EXPAND": ("pending_nexus_seen", "base_count_increased"),
    "ADD_GATEWAYS": ("pending_gateway_seen", "ready_gateway_increased"),
    "TECH_ROBO": (
        "pending_robo_seen",
        "ready_robo_seen",
        "observer_increased",
        "immortal_increased",
        "robo_unit_produced",
    ),
    "FORGE_UPGRADES": (
        "forge_pending_seen",
        "forge_ready_seen",
        "upgrade_pending_seen",
        "upgrade_level_increased",
    ),
    "BUILD_STATIC_DEFENSE": ("static_defense_increased", "threat_cleared"),
    "PRODUCE_ARMY": (
        "army_count_increased",
        "observer_increased",
        "immortal_increased",
        "robo_unit_produced",
    ),
    "BOOST_WORKERS": ("worker_count_increased",),
}
SELECTED_METRICS: tuple[str, ...] = (
    "ready_gateway_delta",
    "pending_gateway_delta",
    "ready_robo_delta",
    "observer_delta",
    "immortal_delta",
    "army_count_delta",
    "static_defense_delta",
    "pending_static_defense_delta",
    "base_under_threat_after",
    "worker_delta",
    "minerals_after",
    "vespene_after",
    "gateway_idle_after",
    "robo_idle_after",
)
START_METRICS: tuple[str, ...] = (
    "minerals",
    "vespene",
    "supply_left",
    "own_bases",
    "pending_bases",
    "ready_gateways",
    "pending_gateways",
    "ready_robo",
    "pending_robo",
    "ready_forge",
    "ready_static_defense",
    "pending_static_defense",
    "has_cybernetics_core",
    "army_count",
    "workers",
    "gateway_idle_count",
    "robo_idle_count",
    "base_under_air_threat",
    "base_under_ground_threat",
    "base_under_threat",
)


@dataclass(frozen=True)
class StrategySignalRecord:
    """One candidate action quality label derived from a trajectory row."""

    source: str
    path: str
    step: int
    game_time: float
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    tactic_id: str | None
    candidate_source: str
    recorded_action: str
    candidate_action: str
    context: str
    threat_state: str
    immediate_executable: bool
    candidate_blocker: str | None
    start_metrics: dict[str, float]
    execution_attempted: bool | None
    execution_effect: str | None
    execution_blocker: str | None
    payoff_observed: bool
    payoff_events_by_window: dict[str, list[str]]
    negative_events_by_window: dict[str, list[str]]
    metrics_by_window: dict[str, dict[str, float]]
    label_quality: str
    recommended_training_use: str
    reasons: list[str]


@dataclass(frozen=True)
class StrategySignalDataset:
    """A compact signal dataset plus summary counts for reports."""

    inputs: list[str]
    lookahead_seconds: list[float]
    files: int
    rows: int
    training_rows: int
    records: list[StrategySignalRecord]
    records_by_training_use: dict[str, int]
    records_by_label_quality: dict[str, int]
    records_by_candidate_source: dict[str, int]
    records_by_candidate_action: dict[str, int]


def build_strategy_signal_dataset(
    paths: StrategyTrajectoryPathInput,
    *,
    lookahead_seconds: Iterable[float] = DEFAULT_SIGNAL_WINDOWS,
    include_before_filter_candidates: bool = True,
) -> StrategySignalDataset:
    """Return row-level action quality signals from strategy trajectory files."""
    input_paths = _input_paths(paths)
    input_strings = [str(path) for path in input_paths]
    windows = tuple(float(value) for value in lookahead_seconds)
    files = discover_strategy_trajectory_files(paths)
    records: list[StrategySignalRecord] = []
    rows_total = 0
    training_rows_total = 0

    for path in files:
        source = _source_for_file(path, input_paths)
        rows = list(_iter_valid_strategy_rows(path, source=source))
        rows_total += len(rows)
        for index, row in enumerate(rows):
            if row.done:
                continue
            training_rows_total += 1
            outcomes = {
                window_key(seconds): _window_outcome(rows, index, seconds)
                for seconds in windows
            }
            records.append(
                _build_record(
                    row=row,
                    candidate_source="recorded",
                    candidate_action=row.action_name,
                    outcomes=outcomes,
                    is_counterfactual=False,
                )
            )
            if include_before_filter_candidates and row.before_action:
                if row.before_action != row.action_name:
                    records.append(
                        _build_record(
                            row=row,
                            candidate_source="before_filter",
                            candidate_action=row.before_action,
                            outcomes=outcomes,
                            is_counterfactual=True,
                        )
                    )

    return StrategySignalDataset(
        inputs=input_strings,
        lookahead_seconds=list(windows),
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        records=records,
        records_by_training_use=_count(
            record.recommended_training_use for record in records
        ),
        records_by_label_quality=_count(record.label_quality for record in records),
        records_by_candidate_source=_count(record.candidate_source for record in records),
        records_by_candidate_action=_count(record.candidate_action for record in records),
    )


def _build_record(
    *,
    row: _StrategyOutcomeRow,
    candidate_source: str,
    candidate_action: str,
    outcomes: dict[str, Any],
    is_counterfactual: bool,
) -> StrategySignalRecord:
    executable, blocker = candidate_executability(row, candidate_action)
    if _recorded_execution_succeeded(
        row=row,
        candidate_action=candidate_action,
        is_counterfactual=is_counterfactual,
    ):
        executable = True
        blocker = None
    payoff_events_by_window = {
        key: _payoff_events(candidate_action, outcome.events)
        for key, outcome in outcomes.items()
    }
    negative_events_by_window = {
        key: _negative_events(outcome.events, outcome.metrics)
        for key, outcome in outcomes.items()
    }
    metrics_by_window = {
        key: _selected_metrics(outcome.metrics)
        for key, outcome in outcomes.items()
    }
    payoff_observed = any(payoff_events_by_window.values())
    label_quality, training_use, reasons = _classify_signal(
        row=row,
        candidate_action=candidate_action,
        immediate_executable=executable,
        candidate_blocker=blocker,
        payoff_observed=payoff_observed,
        negative_events_by_window=negative_events_by_window,
        metrics_by_window=metrics_by_window,
        is_counterfactual=is_counterfactual,
    )
    return StrategySignalRecord(
        source=row.source,
        path=str(row.path),
        step=row.step,
        game_time=row.game_time,
        map_name=row.map_name,
        difficulty=row.difficulty,
        opponent_race=row.opponent_race,
        opponent_ai_build=row.opponent_ai_build,
        tactic_id=row.tactic_id,
        candidate_source=candidate_source,
        recorded_action=row.action_name,
        candidate_action=candidate_action,
        context=classify_replay_context(row, candidate_action),
        threat_state=classify_threat_state(row),
        immediate_executable=executable,
        candidate_blocker=blocker,
        start_metrics=_start_metrics(row.observation),
        execution_attempted=row.execution_attempted,
        execution_effect=row.execution_effect,
        execution_blocker=row.execution_blocker,
        payoff_observed=payoff_observed,
        payoff_events_by_window=payoff_events_by_window,
        negative_events_by_window=negative_events_by_window,
        metrics_by_window=metrics_by_window,
        label_quality=label_quality,
        recommended_training_use=training_use,
        reasons=reasons,
    )


def _recorded_execution_succeeded(
    *,
    row: _StrategyOutcomeRow,
    candidate_action: str,
    is_counterfactual: bool,
) -> bool:
    if is_counterfactual or candidate_action != row.action_name:
        return False
    if row.execution_blocker:
        return False
    return row.execution_effect not in {None, "", "noop"}


def _classify_signal(
    *,
    row: _StrategyOutcomeRow,
    candidate_action: str,
    immediate_executable: bool,
    candidate_blocker: str | None,
    payoff_observed: bool,
    negative_events_by_window: dict[str, list[str]],
    metrics_by_window: dict[str, dict[str, float]],
    is_counterfactual: bool,
) -> tuple[str, str, list[str]]:
    reasons: list[str] = []
    if not immediate_executable:
        reasons.append(f"candidate_not_executable:{candidate_blocker}")
        return "bad", "drop_non_executable", reasons
    if is_counterfactual:
        reasons.append("counterfactual_not_observed")
        return "unknown", "needs_fresh_ab", reasons
    if row.execution_blocker:
        reasons.append(f"execution_blocker:{row.execution_blocker}")
        return "bad", "drop_non_executable", reasons

    last_window = _last_window(metrics_by_window)
    terminal_negatives = negative_events_by_window.get(last_window, [])
    if _is_action_space_exhausted(row, candidate_action):
        reasons.append("action_space_exhausted")
        reasons.extend(terminal_negatives)
        label_quality = "bad" if terminal_negatives else "unknown"
        return label_quality, "action_space_exhausted", sorted(dict.fromkeys(reasons))

    if _is_veto_negative(
        candidate_action,
        terminal_negatives,
        metrics_by_window.get(last_window, {}),
    ):
        reasons.extend(terminal_negatives or ["negative_outcome"])
        return "bad", "veto_negative", sorted(dict.fromkeys(reasons))

    if payoff_observed:
        reasons.append("payoff_observed")
        return "good", "accept_positive", reasons

    if candidate_action == "STAY_COURSE":
        reasons.append("noop_or_wait")
        return "unknown", "drop_ambiguous", reasons

    reasons.append("no_payoff_observed")
    return "weak", "weak_context", reasons


def _is_action_space_exhausted(
    row: _StrategyOutcomeRow,
    candidate_action: str,
) -> bool:
    if candidate_action != "STAY_COURSE":
        return False
    if classify_threat_state(row) == "no_threat":
        return False
    executable_actions = [
        action
        for action in _strategy_action_names()
        if candidate_executability(row, action)[0]
    ]
    return executable_actions == ["STAY_COURSE"]


def _is_veto_negative(
    candidate_action: str,
    negative_events: list[str],
    metrics: dict[str, float],
) -> bool:
    if "threat_persisted" in negative_events and metrics.get("army_count_delta", 0.0) < 0.0:
        return True
    if candidate_action == "PRODUCE_ARMY" and metrics.get("army_count_delta", 0.0) < 0.0:
        return True
    if candidate_action == "ADD_GATEWAYS" and metrics.get("ready_gateway_delta", 0.0) < 0.0:
        return True
    if candidate_action == "BUILD_STATIC_DEFENSE" and "threat_persisted" in negative_events:
        return True
    if candidate_action == "STAY_COURSE" and "threat_persisted" in negative_events:
        return True
    return False


def _payoff_events(candidate_action: str, events: dict[str, bool]) -> list[str]:
    names = PAYOFF_EVENTS_BY_ACTION.get(candidate_action, ())
    return [name for name in names if events.get(name, False)]


def _negative_events(events: dict[str, bool], metrics: dict[str, float]) -> list[str]:
    names: list[str] = []
    if events.get("threat_persisted", False):
        names.append("threat_persisted")
    if metrics.get("army_count_delta", 0.0) < 0.0:
        names.append("army_count_decreased")
    if metrics.get("ready_gateway_delta", 0.0) < 0.0:
        names.append("ready_gateways_decreased")
    if metrics.get("static_defense_delta", 0.0) < 0.0:
        names.append("static_defense_decreased")
    return names


def _selected_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        name: float(metrics[name])
        for name in SELECTED_METRICS
        if name in metrics
    }


def _start_metrics(observation: dict[str, float]) -> dict[str, float]:
    return {
        name: float(observation.get(name, 0.0))
        for name in START_METRICS
    }


def _last_window(metrics_by_window: dict[str, dict[str, float]]) -> str:
    if not metrics_by_window:
        return ""
    return sorted(
        metrics_by_window,
        key=lambda key: float(key[:-1]) if key.endswith("s") else float(key),
    )[-1]


def _strategy_action_names() -> list[str]:
    return [
        STRATEGY_ACTION_NAMES[action_id]
        for action_id in sorted(STRATEGY_ACTION_NAMES)
    ]


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _count(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in Counter(values).items()))
