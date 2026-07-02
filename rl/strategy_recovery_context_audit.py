"""Recovery accept-positive context-slice audit for strategy checkpoints."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path

import torch

from rl.strategy_checkpoint_signal_audit import (
    ACTION_CRITIC_FALLBACK_POLICIES,
    PREDICTION_MODES,
    StrategyCheckpointSignalDecision,
    audit_strategy_checkpoint_signals,
)
from rl.strategy_datasets import (
    StrategyTrajectoryPathInput,
    discover_strategy_trajectory_files,
)
from rl.strategy_filtered_datasets import (
    RECOVERY_ACCEPT_POSITIVE_CONTEXT_FILTERS,
    RECOVERY_ACTION_NAMES,
    recovery_accept_positive_context_matches_observation,
)
from rl.strategy_outcome_diagnostics import _iter_valid_strategy_rows, _source_for_file
from rl.strategy_replay_candidate import classify_replay_context, classify_threat_state
from rl.strategy_signal_critic import (
    DEFAULT_CRITIC_MAX_BAD_RATE,
    DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    DEFAULT_CRITIC_MIN_SAMPLES,
)
from rl.strategy_signal_dataset import StrategySignalRecord, build_strategy_signal_dataset


DEFAULT_MAX_CONTEXT_MISSED_ROWS = 0
DEFAULT_MAX_CONTEXT_MISSED_RATE = 0.0
DEFAULT_MAX_CONTEXT_CROSS_ACTION_ROWS = 0
DEFAULT_MAX_CONTEXT_CROSS_ACTION_RATE = 0.0


@dataclass(frozen=True)
class StrategyRecoveryContextActionSummary:
    """Checkpoint behavior for one recorded recovery action."""

    recorded_action: str
    rows: int
    matches: int
    match_rate: float
    context_matched_rows: int
    context_matched_matches: int
    context_matched_match_rate: float
    context_skipped_rows: int
    context_skipped_matches: int
    context_skipped_match_rate: float
    cross_action_confusion_rows: int
    context_matched_cross_action_confusion_rows: int
    predicted_counts_by_action: dict[str, int]
    context_matched_predicted_counts_by_action: dict[str, int]
    context_skipped_predicted_counts_by_action: dict[str, int]


@dataclass(frozen=True)
class StrategyRecoveryContextDecision:
    """One recovery accept-positive row and checkpoint prediction."""

    path: str
    source: str
    step: int
    game_time: float
    recorded_action: str
    raw_predicted_action: str
    predicted_action: str
    prediction_matches_recorded: bool
    prediction_was_masked: bool
    context_filter: str
    context_matches_recorded_action: bool
    recorded_action_replay_context: str
    predicted_action_replay_context: str
    threat_state: str
    is_cross_action_confusion: bool
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    start_metrics: dict[str, float]


@dataclass(frozen=True)
class StrategyRecoveryContextAudit:
    """Dataset-level checkpoint audit for recovery-positive context slices."""

    inputs: list[str]
    checkpoint_path: str
    prediction_mode: str
    context_filter: str
    action_critic_checkpoint_path: str | None
    action_critic_threshold: float | None
    action_critic_fallback_policy: str | None
    recommendation: str
    blocking_reasons: list[str]
    warnings: list[str]
    max_context_missed_rows: int
    max_context_missed_rate: float
    max_context_cross_action_rows: int
    max_context_cross_action_rate: float
    files: int
    rows: int
    accept_positive_recovery_rows: int
    accept_positive_recovery_matches: int
    accept_positive_recovery_match_rate: float
    context_matched_accept_positive_recovery_rows: int
    context_matched_accept_positive_recovery_matches: int
    context_matched_accept_positive_recovery_match_rate: float
    context_missed_accept_positive_recovery_rows: int
    context_missed_accept_positive_recovery_rate: float
    context_skipped_accept_positive_recovery_rows: int
    cross_action_confusion_rows: int
    cross_action_confusion_rate: float
    context_matched_cross_action_confusion_rows: int
    context_matched_cross_action_confusion_rate: float
    recorded_counts_by_action: dict[str, int]
    predicted_counts_by_action: dict[str, int]
    context_matched_recorded_counts_by_action: dict[str, int]
    context_matched_predicted_counts_by_action: dict[str, int]
    confusion_counts_by_recorded_then_predicted: dict[str, dict[str, int]]
    context_matched_confusion_counts_by_recorded_then_predicted: dict[
        str,
        dict[str, int],
    ]
    action_summaries: list[StrategyRecoveryContextActionSummary]
    decisions: list[StrategyRecoveryContextDecision]


def audit_strategy_recovery_context(
    paths: StrategyTrajectoryPathInput,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    prediction_mode: str = "executable-mask",
    context_filter: str = "pre-collapse-recovery",
    max_decisions: int = 40,
    critic_min_samples: int = DEFAULT_CRITIC_MIN_SAMPLES,
    critic_max_bad_rate: float = DEFAULT_CRITIC_MAX_BAD_RATE,
    critic_max_veto_negative_rate: float = DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    action_critic_checkpoint_path: str | Path | None = None,
    action_critic_threshold: float = 0.5,
    action_critic_fallback_policy: str = "lowest-risk",
    max_context_missed_rows: int = DEFAULT_MAX_CONTEXT_MISSED_ROWS,
    max_context_missed_rate: float = DEFAULT_MAX_CONTEXT_MISSED_RATE,
    max_context_cross_action_rows: int = DEFAULT_MAX_CONTEXT_CROSS_ACTION_ROWS,
    max_context_cross_action_rate: float = DEFAULT_MAX_CONTEXT_CROSS_ACTION_RATE,
) -> StrategyRecoveryContextAudit:
    """Audit checkpoint confusion on observed recovery accept-positive rows."""
    _validate_args(
        prediction_mode=prediction_mode,
        context_filter=context_filter,
        max_decisions=max_decisions,
        action_critic_fallback_policy=action_critic_fallback_policy,
        max_context_missed_rows=max_context_missed_rows,
        max_context_missed_rate=max_context_missed_rate,
        max_context_cross_action_rows=max_context_cross_action_rows,
        max_context_cross_action_rate=max_context_cross_action_rate,
    )
    input_paths = _input_paths(paths)
    input_strings = [str(path) for path in input_paths]
    files = discover_strategy_trajectory_files(paths)
    signal_dataset = build_strategy_signal_dataset(
        paths,
        include_before_filter_candidates=False,
    )
    signal_by_key = {
        _record_key(record.path, record.step, record.recorded_action): record
        for record in signal_dataset.records
        if record.candidate_source == "recorded"
    }
    signal_audit = audit_strategy_checkpoint_signals(
        paths,
        checkpoint_path,
        device=device,
        prediction_mode=prediction_mode,
        critic_min_samples=critic_min_samples,
        critic_max_bad_rate=critic_max_bad_rate,
        critic_max_veto_negative_rate=critic_max_veto_negative_rate,
        action_critic_checkpoint_path=action_critic_checkpoint_path,
        action_critic_threshold=action_critic_threshold,
        action_critic_fallback_policy=action_critic_fallback_policy,
    )
    decision_by_key = {
        _record_key(decision.path, decision.step, decision.recorded_action): decision
        for decision in signal_audit.decisions
    }

    rows_total = 0
    missing_signal_rows = 0
    missing_decision_rows = 0
    decisions: list[StrategyRecoveryContextDecision] = []
    for path in files:
        source = _source_for_file(path, input_paths)
        for row in _iter_valid_strategy_rows(path, source=source):
            rows_total += 1
            if row.done:
                continue
            key = _row_key(row)
            signal = signal_by_key.get(key)
            if signal is None:
                missing_signal_rows += 1
                continue
            if not _is_accept_positive_recovery(signal):
                continue
            checkpoint_decision = decision_by_key.get(key)
            if checkpoint_decision is None:
                missing_decision_rows += 1
                continue
            decisions.append(
                _context_decision(
                    row=row,
                    signal=signal,
                    decision=checkpoint_decision,
                    context_filter=context_filter,
                )
            )

    warnings: list[str] = list(signal_audit.warnings)
    if missing_signal_rows:
        warnings.append(f"missing_signal_rows:{missing_signal_rows}")
    if missing_decision_rows:
        warnings.append(f"missing_checkpoint_decision_rows:{missing_decision_rows}")
    if not decisions:
        warnings.append("no_accept_positive_recovery_rows")

    return _summarize(
        input_strings=input_strings,
        checkpoint_path=checkpoint_path,
        prediction_mode=prediction_mode,
        context_filter=context_filter,
        action_critic_checkpoint_path=action_critic_checkpoint_path,
        action_critic_threshold=(
            float(action_critic_threshold)
            if prediction_mode == "action-critic-mask"
            else None
        ),
        action_critic_fallback_policy=(
            action_critic_fallback_policy
            if prediction_mode == "action-critic-mask"
            else None
        ),
        files=len(files),
        rows=rows_total,
        decisions=decisions,
        warnings=warnings,
        max_decisions=max_decisions,
        max_context_missed_rows=max_context_missed_rows,
        max_context_missed_rate=max_context_missed_rate,
        max_context_cross_action_rows=max_context_cross_action_rows,
        max_context_cross_action_rate=max_context_cross_action_rate,
    )


def _context_decision(
    *,
    row,
    signal: StrategySignalRecord,
    decision: StrategyCheckpointSignalDecision,
    context_filter: str,
) -> StrategyRecoveryContextDecision:
    context_matches = recovery_accept_positive_context_matches_observation(
        row.observation,
        row.action_name,
        context_filter,
    )
    predicted_recovery_other_action = (
        decision.predicted_action in RECOVERY_ACTION_NAMES
        and decision.predicted_action != row.action_name
    )
    return StrategyRecoveryContextDecision(
        path=str(row.path),
        source=row.source,
        step=row.step,
        game_time=row.game_time,
        recorded_action=row.action_name,
        raw_predicted_action=decision.raw_predicted_action,
        predicted_action=decision.predicted_action,
        prediction_matches_recorded=decision.prediction_matches_recorded,
        prediction_was_masked=decision.prediction_was_masked,
        context_filter=context_filter,
        context_matches_recorded_action=context_matches,
        recorded_action_replay_context=classify_replay_context(row, row.action_name),
        predicted_action_replay_context=classify_replay_context(
            row,
            decision.predicted_action,
        ),
        threat_state=classify_threat_state(row),
        is_cross_action_confusion=predicted_recovery_other_action,
        map_name=row.map_name,
        difficulty=row.difficulty,
        opponent_race=row.opponent_race,
        opponent_ai_build=row.opponent_ai_build,
        start_metrics={
            "game_time": float(row.observation.get("game_time", 0.0)),
            "minerals": float(row.observation.get("minerals", 0.0)),
            "vespene": float(row.observation.get("vespene", 0.0)),
            "ready_robo": float(row.observation.get("ready_robo", 0.0)),
            "pending_robo": float(row.observation.get("pending_robo", 0.0)),
            "ready_static_defense": float(
                row.observation.get("ready_static_defense", 0.0)
            ),
            "pending_static_defense": float(
                row.observation.get("pending_static_defense", 0.0)
            ),
            "base_under_threat": float(row.observation.get("base_under_threat", 0.0)),
            "base_under_air_threat": float(
                row.observation.get("base_under_air_threat", 0.0)
            ),
            "base_under_ground_threat": float(
                row.observation.get("base_under_ground_threat", 0.0)
            ),
        },
    )


def _summarize(
    *,
    input_strings: list[str],
    checkpoint_path: str | Path,
    prediction_mode: str,
    context_filter: str,
    action_critic_checkpoint_path: str | Path | None,
    action_critic_threshold: float | None,
    action_critic_fallback_policy: str | None,
    files: int,
    rows: int,
    decisions: list[StrategyRecoveryContextDecision],
    warnings: list[str],
    max_decisions: int,
    max_context_missed_rows: int,
    max_context_missed_rate: float,
    max_context_cross_action_rows: int,
    max_context_cross_action_rate: float,
) -> StrategyRecoveryContextAudit:
    total = len(decisions)
    matches = sum(1 for decision in decisions if decision.prediction_matches_recorded)
    context_decisions = [
        decision for decision in decisions if decision.context_matches_recorded_action
    ]
    skipped_decisions = [
        decision
        for decision in decisions
        if not decision.context_matches_recorded_action
    ]
    context_matches = sum(
        1 for decision in context_decisions if decision.prediction_matches_recorded
    )
    context_missed = len(context_decisions) - context_matches
    cross_action = sum(1 for decision in decisions if decision.is_cross_action_confusion)
    context_cross_action = sum(
        1 for decision in context_decisions if decision.is_cross_action_confusion
    )
    summary_warnings = list(warnings)
    if total and not context_decisions:
        summary_warnings.append("no_context_matched_accept_positive_recovery_rows")
    context_missed_rate = _ratio(context_missed, len(context_decisions))
    context_cross_action_rate = _ratio(context_cross_action, len(context_decisions))
    blocking_reasons = _blocking_reasons(
        context_missed_rows=context_missed,
        context_missed_rate=context_missed_rate,
        max_context_missed_rows=max_context_missed_rows,
        max_context_missed_rate=max_context_missed_rate,
        context_cross_action_rows=context_cross_action,
        context_cross_action_rate=context_cross_action_rate,
        max_context_cross_action_rows=max_context_cross_action_rows,
        max_context_cross_action_rate=max_context_cross_action_rate,
    )
    representative_decisions = sorted(
        decisions,
        key=lambda decision: (
            0
            if decision.context_matches_recorded_action
            and not decision.prediction_matches_recorded
            else 1,
            0 if decision.is_cross_action_confusion else 1,
            Path(decision.path).name,
            decision.step,
        ),
    )[:max_decisions]

    return StrategyRecoveryContextAudit(
        inputs=input_strings,
        checkpoint_path=str(checkpoint_path),
        prediction_mode=prediction_mode,
        context_filter=context_filter,
        action_critic_checkpoint_path=(
            str(action_critic_checkpoint_path)
            if action_critic_checkpoint_path is not None
            else None
        ),
        action_critic_threshold=action_critic_threshold,
        action_critic_fallback_policy=action_critic_fallback_policy,
        recommendation="ready" if not blocking_reasons else "hold",
        blocking_reasons=blocking_reasons,
        warnings=summary_warnings,
        max_context_missed_rows=int(max_context_missed_rows),
        max_context_missed_rate=float(max_context_missed_rate),
        max_context_cross_action_rows=int(max_context_cross_action_rows),
        max_context_cross_action_rate=float(max_context_cross_action_rate),
        files=files,
        rows=rows,
        accept_positive_recovery_rows=total,
        accept_positive_recovery_matches=matches,
        accept_positive_recovery_match_rate=_ratio(matches, total),
        context_matched_accept_positive_recovery_rows=len(context_decisions),
        context_matched_accept_positive_recovery_matches=context_matches,
        context_matched_accept_positive_recovery_match_rate=_ratio(
            context_matches,
            len(context_decisions),
        ),
        context_missed_accept_positive_recovery_rows=context_missed,
        context_missed_accept_positive_recovery_rate=context_missed_rate,
        context_skipped_accept_positive_recovery_rows=len(skipped_decisions),
        cross_action_confusion_rows=cross_action,
        cross_action_confusion_rate=_ratio(cross_action, total),
        context_matched_cross_action_confusion_rows=context_cross_action,
        context_matched_cross_action_confusion_rate=context_cross_action_rate,
        recorded_counts_by_action=_count(decision.recorded_action for decision in decisions),
        predicted_counts_by_action=_count(decision.predicted_action for decision in decisions),
        context_matched_recorded_counts_by_action=_count(
            decision.recorded_action for decision in context_decisions
        ),
        context_matched_predicted_counts_by_action=_count(
            decision.predicted_action for decision in context_decisions
        ),
        confusion_counts_by_recorded_then_predicted=_confusion_counts(decisions),
        context_matched_confusion_counts_by_recorded_then_predicted=(
            _confusion_counts(context_decisions)
        ),
        action_summaries=_action_summaries(decisions),
        decisions=representative_decisions,
    )


def _action_summaries(
    decisions: list[StrategyRecoveryContextDecision],
) -> list[StrategyRecoveryContextActionSummary]:
    summaries: list[StrategyRecoveryContextActionSummary] = []
    for action in RECOVERY_ACTION_NAMES:
        action_decisions = [
            decision for decision in decisions if decision.recorded_action == action
        ]
        if not action_decisions:
            continue
        context_decisions = [
            decision
            for decision in action_decisions
            if decision.context_matches_recorded_action
        ]
        skipped_decisions = [
            decision
            for decision in action_decisions
            if not decision.context_matches_recorded_action
        ]
        matches = sum(
            1 for decision in action_decisions if decision.prediction_matches_recorded
        )
        context_matches = sum(
            1 for decision in context_decisions if decision.prediction_matches_recorded
        )
        skipped_matches = sum(
            1 for decision in skipped_decisions if decision.prediction_matches_recorded
        )
        cross_action = sum(
            1 for decision in action_decisions if decision.is_cross_action_confusion
        )
        context_cross_action = sum(
            1 for decision in context_decisions if decision.is_cross_action_confusion
        )
        summaries.append(
            StrategyRecoveryContextActionSummary(
                recorded_action=action,
                rows=len(action_decisions),
                matches=matches,
                match_rate=_ratio(matches, len(action_decisions)),
                context_matched_rows=len(context_decisions),
                context_matched_matches=context_matches,
                context_matched_match_rate=_ratio(
                    context_matches,
                    len(context_decisions),
                ),
                context_skipped_rows=len(skipped_decisions),
                context_skipped_matches=skipped_matches,
                context_skipped_match_rate=_ratio(
                    skipped_matches,
                    len(skipped_decisions),
                ),
                cross_action_confusion_rows=cross_action,
                context_matched_cross_action_confusion_rows=context_cross_action,
                predicted_counts_by_action=_count(
                    decision.predicted_action for decision in action_decisions
                ),
                context_matched_predicted_counts_by_action=_count(
                    decision.predicted_action for decision in context_decisions
                ),
                context_skipped_predicted_counts_by_action=_count(
                    decision.predicted_action for decision in skipped_decisions
                ),
            )
        )
    return summaries


def _validate_args(
    *,
    prediction_mode: str,
    context_filter: str,
    max_decisions: int,
    action_critic_fallback_policy: str,
    max_context_missed_rows: int,
    max_context_missed_rate: float,
    max_context_cross_action_rows: int,
    max_context_cross_action_rate: float,
) -> None:
    if prediction_mode not in PREDICTION_MODES:
        names = ", ".join(PREDICTION_MODES)
        raise ValueError(f"Unknown prediction_mode {prediction_mode!r}; expected {names}")
    if context_filter not in RECOVERY_ACCEPT_POSITIVE_CONTEXT_FILTERS:
        names = ", ".join(RECOVERY_ACCEPT_POSITIVE_CONTEXT_FILTERS)
        raise ValueError(f"Unknown context_filter {context_filter!r}; expected {names}")
    if max_decisions < 0:
        raise ValueError("max_decisions must be >= 0")
    if action_critic_fallback_policy not in ACTION_CRITIC_FALLBACK_POLICIES:
        names = ", ".join(ACTION_CRITIC_FALLBACK_POLICIES)
        raise ValueError(
            "Unknown action_critic_fallback_policy "
            f"{action_critic_fallback_policy!r}; expected {names}"
        )
    if max_context_missed_rows < 0:
        raise ValueError("max_context_missed_rows must be >= 0")
    if not 0.0 <= max_context_missed_rate <= 1.0:
        raise ValueError("max_context_missed_rate must be in [0.0, 1.0]")
    if max_context_cross_action_rows < 0:
        raise ValueError("max_context_cross_action_rows must be >= 0")
    if not 0.0 <= max_context_cross_action_rate <= 1.0:
        raise ValueError("max_context_cross_action_rate must be in [0.0, 1.0]")


def _blocking_reasons(
    *,
    context_missed_rows: int,
    context_missed_rate: float,
    max_context_missed_rows: int,
    max_context_missed_rate: float,
    context_cross_action_rows: int,
    context_cross_action_rate: float,
    max_context_cross_action_rows: int,
    max_context_cross_action_rate: float,
) -> list[str]:
    reasons: list[str] = []
    if context_missed_rows > max_context_missed_rows:
        reasons.append("context_missed_accept_positive_recovery_rows")
    if context_missed_rate > max_context_missed_rate:
        reasons.append("context_missed_accept_positive_recovery_rate")
    if context_cross_action_rows > max_context_cross_action_rows:
        reasons.append("context_cross_action_confusion_rows")
    if context_cross_action_rate > max_context_cross_action_rate:
        reasons.append("context_cross_action_confusion_rate")
    return reasons


def _is_accept_positive_recovery(signal: StrategySignalRecord) -> bool:
    return (
        signal.recommended_training_use == "accept_positive"
        and signal.recorded_action in RECOVERY_ACTION_NAMES
    )


def _row_key(row) -> tuple[str, int, str]:
    return _record_key(row.path, row.step, row.action_name)


def _record_key(path: str | Path, step: int, action: str) -> tuple[str, int, str]:
    return (str(Path(path).resolve()), int(step), str(action))


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _confusion_counts(
    decisions: list[StrategyRecoveryContextDecision],
) -> dict[str, dict[str, int]]:
    by_recorded: dict[str, Counter[str]] = {}
    for decision in decisions:
        by_recorded.setdefault(decision.recorded_action, Counter())
        by_recorded[decision.recorded_action][decision.predicted_action] += 1
    return {
        recorded: _count_from_counter(counter)
        for recorded, counter in sorted(by_recorded.items())
    }


def _count(values) -> dict[str, int]:
    return _count_from_counter(Counter(values))


def _count_from_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((str(name), int(count)) for name, count in counter.items()))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
