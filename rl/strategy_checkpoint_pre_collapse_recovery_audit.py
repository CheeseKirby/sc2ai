"""Checkpoint audit for pre-collapse strategy recovery predictions."""
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
from rl.strategy_outcome_diagnostics import (
    _StrategyOutcomeRow,
    _iter_valid_strategy_rows,
    _source_for_file,
)
from rl.strategy_pre_collapse_recovery_analysis import (
    DEFAULT_LOOKBACK_SECONDS,
    PRE_COLLAPSE_START_METRICS,
    RECOVERY_ACTION_NAMES,
    TARGET_TRAINING_USES,
    _executable_recovery_actions,
    _recovery_executability,
)
from rl.strategy_replay_candidate import classify_replay_context, classify_threat_state
from rl.strategy_signal_critic import (
    DEFAULT_CRITIC_MAX_BAD_RATE,
    DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    DEFAULT_CRITIC_MIN_SAMPLES,
)
from rl.strategy_signal_dataset import (
    StrategySignalRecord,
    build_strategy_signal_dataset,
)

DEFAULT_MAX_MISSED_CHECKPOINT_PRE_COLLAPSE_RECOVERY_ROWS = 0
DEFAULT_MAX_MISSED_CHECKPOINT_PRE_COLLAPSE_RECOVERY_RATE = 0.0
DEFAULT_MAX_MISSED_ACCEPT_POSITIVE_RECOVERY_ROWS = 0
DEFAULT_MAX_MISSED_ACCEPT_POSITIVE_RECOVERY_RATE = 0.0


@dataclass(frozen=True)
class StrategyCheckpointPreCollapseRecoveryWindow:
    """One pre-collapse recovery window with checkpoint prediction details."""

    source_path: str
    source: str
    target_step: int
    target_game_time: float
    step: int
    game_time: float
    seconds_before_target: float
    recorded_action: str
    raw_predicted_action: str
    predicted_action: str
    prediction_was_masked: bool
    threat_state: str
    context: str
    executable_recovery_actions: list[str]
    predicted_recovery_action: str | None
    predicted_executable_recovery_action: str | None
    start_metrics: dict[str, float | str]


@dataclass(frozen=True)
class StrategyCheckpointPreCollapseTarget:
    """One veto/action-space target row and its pre-collapse predictions."""

    source_path: str
    source: str
    step: int
    game_time: float
    recorded_action: str
    recorded_training_use: str
    recorded_label_quality: str
    raw_predicted_action: str
    predicted_action: str
    prediction_matches_recorded: bool
    threat_state: str
    lookback_seconds: float
    pre_collapse_rows: int
    pre_collapse_recovery_window_rows: int
    pre_collapse_checkpoint_recovery_prediction_rows: int
    pre_collapse_checkpoint_executable_recovery_prediction_rows: int
    checkpoint_predicted_recovery_counts_by_action: dict[str, int]
    checkpoint_predicted_executable_recovery_counts_by_action: dict[str, int]
    missed_checkpoint_pre_collapse_recovery: bool
    last_checkpoint_executable_recovery_time: float | None
    last_checkpoint_executable_recovery_action: str | None
    start_metrics: dict[str, float | str]
    recovery_windows: list[StrategyCheckpointPreCollapseRecoveryWindow]


@dataclass(frozen=True)
class StrategyCheckpointAcceptPositiveRecoveryDecision:
    """One positive recovery label and the checkpoint prediction on that row."""

    source_path: str
    source: str
    step: int
    game_time: float
    recorded_action: str
    raw_predicted_action: str
    predicted_action: str
    prediction_matches_recorded: bool
    prediction_was_masked: bool
    threat_state: str
    context: str
    start_metrics: dict[str, float | str]


@dataclass(frozen=True)
class StrategyCheckpointPreCollapseFileSummary:
    """Per-file checkpoint pre-collapse recovery summary."""

    path: str
    source: str
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    rows: int
    training_rows: int
    target_rows: int
    rows_with_pre_collapse_recovery_window: int
    rows_with_checkpoint_pre_collapse_executable_recovery: int
    missed_checkpoint_pre_collapse_recovery_rows: int
    accept_positive_recovery_rows: int
    accept_positive_recovery_matches: int


@dataclass(frozen=True)
class StrategyCheckpointPreCollapseRecoveryAudit:
    """Dataset-level checkpoint recovery-window audit."""

    inputs: list[str]
    checkpoint_path: str
    prediction_mode: str
    action_critic_checkpoint_path: str | None
    action_critic_threshold: float | None
    action_critic_fallback_policy: str | None
    lookback_seconds: float
    target_training_uses: list[str]
    recovery_action_names: list[str]
    recommendation: str
    blocking_reasons: list[str]
    warnings: list[str]
    max_missed_checkpoint_pre_collapse_recovery_rows: int
    max_missed_checkpoint_pre_collapse_recovery_rate: float
    max_missed_accept_positive_recovery_rows: int
    max_missed_accept_positive_recovery_rate: float
    files: int
    rows: int
    training_rows: int
    target_rows: int
    rows_with_pre_collapse_recovery_window: int
    rows_with_checkpoint_pre_collapse_recovery: int
    rows_with_checkpoint_pre_collapse_executable_recovery: int
    missed_checkpoint_pre_collapse_recovery_rows: int
    no_pre_collapse_recovery_window_rows: int
    missed_checkpoint_pre_collapse_recovery_rate: float
    accept_positive_recovery_rows: int
    accept_positive_recovery_matches: int
    missed_accept_positive_recovery_rows: int
    accept_positive_recovery_match_rate: float
    checkpoint_predicted_recovery_counts_by_action: dict[str, int]
    checkpoint_predicted_executable_recovery_counts_by_action: dict[str, int]
    accept_positive_recovery_recorded_counts_by_action: dict[str, int]
    accept_positive_recovery_predicted_counts_by_action: dict[str, int]
    target_prediction_counts_by_action: dict[str, int]
    missing_signal_rows: int
    missing_checkpoint_decision_rows: int
    file_summaries: list[StrategyCheckpointPreCollapseFileSummary]
    targets: list[StrategyCheckpointPreCollapseTarget]
    accept_positive_recovery_decisions: list[
        StrategyCheckpointAcceptPositiveRecoveryDecision
    ]


def audit_strategy_checkpoint_pre_collapse_recovery(
    paths: StrategyTrajectoryPathInput,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    prediction_mode: str = "executable-mask",
    lookback_seconds: float = DEFAULT_LOOKBACK_SECONDS,
    max_targets: int = 20,
    max_windows_per_target: int = 6,
    max_accept_positive_decisions: int = 20,
    critic_min_samples: int = DEFAULT_CRITIC_MIN_SAMPLES,
    critic_max_bad_rate: float = DEFAULT_CRITIC_MAX_BAD_RATE,
    critic_max_veto_negative_rate: float = DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    action_critic_checkpoint_path: str | Path | None = None,
    action_critic_threshold: float = 0.5,
    action_critic_fallback_policy: str = "lowest-risk",
    max_missed_checkpoint_pre_collapse_recovery_rows: int = (
        DEFAULT_MAX_MISSED_CHECKPOINT_PRE_COLLAPSE_RECOVERY_ROWS
    ),
    max_missed_checkpoint_pre_collapse_recovery_rate: float = (
        DEFAULT_MAX_MISSED_CHECKPOINT_PRE_COLLAPSE_RECOVERY_RATE
    ),
    max_missed_accept_positive_recovery_rows: int = (
        DEFAULT_MAX_MISSED_ACCEPT_POSITIVE_RECOVERY_ROWS
    ),
    max_missed_accept_positive_recovery_rate: float = (
        DEFAULT_MAX_MISSED_ACCEPT_POSITIVE_RECOVERY_RATE
    ),
) -> StrategyCheckpointPreCollapseRecoveryAudit:
    """Audit checkpoint recovery predictions before veto/action-space targets."""
    _validate_args(
        prediction_mode=prediction_mode,
        lookback_seconds=lookback_seconds,
        max_targets=max_targets,
        max_windows_per_target=max_windows_per_target,
        max_accept_positive_decisions=max_accept_positive_decisions,
        max_missed_checkpoint_pre_collapse_recovery_rows=(
            max_missed_checkpoint_pre_collapse_recovery_rows
        ),
        max_missed_checkpoint_pre_collapse_recovery_rate=(
            max_missed_checkpoint_pre_collapse_recovery_rate
        ),
        max_missed_accept_positive_recovery_rows=(
            max_missed_accept_positive_recovery_rows
        ),
        max_missed_accept_positive_recovery_rate=(
            max_missed_accept_positive_recovery_rate
        ),
        action_critic_fallback_policy=action_critic_fallback_policy,
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
    training_rows_total = 0
    target_rows_total = 0
    rows_with_window = 0
    rows_with_checkpoint_recovery = 0
    rows_with_checkpoint_executable_recovery = 0
    missed_checkpoint_rows = 0
    no_window_rows = 0
    accept_positive_rows = 0
    accept_positive_matches = 0
    missing_signal_rows = 0
    missing_decision_rows = 0
    checkpoint_recovery_counts: Counter[str] = Counter()
    checkpoint_executable_recovery_counts: Counter[str] = Counter()
    accept_positive_recorded_counts: Counter[str] = Counter()
    accept_positive_predicted_counts: Counter[str] = Counter()
    target_prediction_counts: Counter[str] = Counter()
    file_summaries: list[StrategyCheckpointPreCollapseFileSummary] = []
    targets: list[StrategyCheckpointPreCollapseTarget] = []
    accept_positive_decisions: list[
        StrategyCheckpointAcceptPositiveRecoveryDecision
    ] = []

    for path in files:
        source = _source_for_file(path, input_paths)
        rows = list(_iter_valid_strategy_rows(path, source=source))
        training_rows = [row for row in rows if not row.done]
        rows_total += len(rows)
        training_rows_total += len(training_rows)

        file_targets: list[StrategyCheckpointPreCollapseTarget] = []
        file_accept_positive: list[StrategyCheckpointAcceptPositiveRecoveryDecision] = []
        for index, row in enumerate(rows):
            if row.done:
                continue
            signal = signal_by_key.get(_row_key(row))
            decision = decision_by_key.get(_row_key(row))
            if signal is None:
                missing_signal_rows += 1
                continue
            if decision is None:
                missing_decision_rows += 1

            if _is_accept_positive_recovery(signal):
                accept_positive_rows += 1
                accept_positive_recorded_counts[row.action_name] += 1
                positive_decision = _accept_positive_decision(
                    row=row,
                    decision=decision,
                )
                file_accept_positive.append(positive_decision)
                if positive_decision.prediction_matches_recorded:
                    accept_positive_matches += 1
                accept_positive_predicted_counts[positive_decision.predicted_action] += 1

            if signal.recommended_training_use not in TARGET_TRAINING_USES:
                continue

            target = _target_for_row(
                rows=rows,
                target_index=index,
                signal=signal,
                decision=decision,
                decision_by_key=decision_by_key,
                lookback_seconds=float(lookback_seconds),
                max_windows=max_windows_per_target,
            )
            file_targets.append(target)
            target_rows_total += 1
            target_prediction_counts[target.predicted_action] += 1
            if target.pre_collapse_recovery_window_rows:
                rows_with_window += 1
            else:
                no_window_rows += 1
            if target.pre_collapse_checkpoint_recovery_prediction_rows:
                rows_with_checkpoint_recovery += 1
            if target.pre_collapse_checkpoint_executable_recovery_prediction_rows:
                rows_with_checkpoint_executable_recovery += 1
            if target.missed_checkpoint_pre_collapse_recovery:
                missed_checkpoint_rows += 1
            checkpoint_recovery_counts.update(
                target.checkpoint_predicted_recovery_counts_by_action
            )
            checkpoint_executable_recovery_counts.update(
                target.checkpoint_predicted_executable_recovery_counts_by_action
            )

        if file_targets or file_accept_positive:
            first_metrics = _first_metrics(file_targets, file_accept_positive)
            file_summaries.append(
                StrategyCheckpointPreCollapseFileSummary(
                    path=str(path),
                    source=source,
                    map_name=str(first_metrics.get("map_name", "")),
                    difficulty=str(first_metrics.get("difficulty", "")),
                    opponent_race=str(first_metrics.get("opponent_race", "")),
                    opponent_ai_build=str(
                        first_metrics.get("opponent_ai_build", "")
                    ),
                    rows=len(rows),
                    training_rows=len(training_rows),
                    target_rows=len(file_targets),
                    rows_with_pre_collapse_recovery_window=sum(
                        1
                        for target in file_targets
                        if target.pre_collapse_recovery_window_rows
                    ),
                    rows_with_checkpoint_pre_collapse_executable_recovery=sum(
                        1
                        for target in file_targets
                        if target.pre_collapse_checkpoint_executable_recovery_prediction_rows
                    ),
                    missed_checkpoint_pre_collapse_recovery_rows=sum(
                        1
                        for target in file_targets
                        if target.missed_checkpoint_pre_collapse_recovery
                    ),
                    accept_positive_recovery_rows=len(file_accept_positive),
                    accept_positive_recovery_matches=sum(
                        1
                        for decision in file_accept_positive
                        if decision.prediction_matches_recorded
                    ),
                )
            )
        targets.extend(file_targets)
        accept_positive_decisions.extend(file_accept_positive)

    missed_accept_positive_rows = accept_positive_rows - accept_positive_matches
    missed_checkpoint_rate = _ratio(missed_checkpoint_rows, target_rows_total)
    accept_positive_match_rate = _ratio(accept_positive_matches, accept_positive_rows)
    missed_accept_positive_rate = _ratio(
        missed_accept_positive_rows,
        accept_positive_rows,
    )
    blocking_reasons = _blocking_reasons(
        missing_signal_rows=missing_signal_rows,
        missing_decision_rows=missing_decision_rows,
        missed_checkpoint_pre_collapse_recovery_rows=missed_checkpoint_rows,
        missed_checkpoint_pre_collapse_recovery_rate=missed_checkpoint_rate,
        max_missed_checkpoint_pre_collapse_recovery_rows=(
            max_missed_checkpoint_pre_collapse_recovery_rows
        ),
        max_missed_checkpoint_pre_collapse_recovery_rate=(
            max_missed_checkpoint_pre_collapse_recovery_rate
        ),
        missed_accept_positive_recovery_rows=missed_accept_positive_rows,
        missed_accept_positive_recovery_rate=missed_accept_positive_rate,
        max_missed_accept_positive_recovery_rows=(
            max_missed_accept_positive_recovery_rows
        ),
        max_missed_accept_positive_recovery_rate=(
            max_missed_accept_positive_recovery_rate
        ),
    )
    warnings: list[str] = list(signal_audit.warnings)
    if target_rows_total == 0:
        warnings.append("no_target_rows")
    if accept_positive_rows == 0:
        warnings.append("no_accept_positive_recovery_rows")

    sorted_targets = sorted(
        targets,
        key=lambda target: (
            0 if target.missed_checkpoint_pre_collapse_recovery else 1,
            Path(target.source_path).name,
            target.step,
        ),
    )[:max_targets]
    sorted_accept_positive = sorted(
        accept_positive_decisions,
        key=lambda decision: (
            0 if not decision.prediction_matches_recorded else 1,
            Path(decision.source_path).name,
            decision.step,
        ),
    )[:max_accept_positive_decisions]

    return StrategyCheckpointPreCollapseRecoveryAudit(
        inputs=input_strings,
        checkpoint_path=str(checkpoint_path),
        prediction_mode=prediction_mode,
        action_critic_checkpoint_path=(
            str(action_critic_checkpoint_path)
            if action_critic_checkpoint_path is not None
            else None
        ),
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
        lookback_seconds=float(lookback_seconds),
        target_training_uses=list(TARGET_TRAINING_USES),
        recovery_action_names=list(RECOVERY_ACTION_NAMES),
        recommendation="ready" if not blocking_reasons else "hold",
        blocking_reasons=blocking_reasons,
        warnings=warnings,
        max_missed_checkpoint_pre_collapse_recovery_rows=int(
            max_missed_checkpoint_pre_collapse_recovery_rows
        ),
        max_missed_checkpoint_pre_collapse_recovery_rate=float(
            max_missed_checkpoint_pre_collapse_recovery_rate
        ),
        max_missed_accept_positive_recovery_rows=int(
            max_missed_accept_positive_recovery_rows
        ),
        max_missed_accept_positive_recovery_rate=float(
            max_missed_accept_positive_recovery_rate
        ),
        files=len(files),
        rows=rows_total,
        training_rows=training_rows_total,
        target_rows=target_rows_total,
        rows_with_pre_collapse_recovery_window=rows_with_window,
        rows_with_checkpoint_pre_collapse_recovery=rows_with_checkpoint_recovery,
        rows_with_checkpoint_pre_collapse_executable_recovery=(
            rows_with_checkpoint_executable_recovery
        ),
        missed_checkpoint_pre_collapse_recovery_rows=missed_checkpoint_rows,
        no_pre_collapse_recovery_window_rows=no_window_rows,
        missed_checkpoint_pre_collapse_recovery_rate=missed_checkpoint_rate,
        accept_positive_recovery_rows=accept_positive_rows,
        accept_positive_recovery_matches=accept_positive_matches,
        missed_accept_positive_recovery_rows=missed_accept_positive_rows,
        accept_positive_recovery_match_rate=accept_positive_match_rate,
        checkpoint_predicted_recovery_counts_by_action=_action_counts(
            checkpoint_recovery_counts
        ),
        checkpoint_predicted_executable_recovery_counts_by_action=_action_counts(
            checkpoint_executable_recovery_counts
        ),
        accept_positive_recovery_recorded_counts_by_action=_action_counts(
            accept_positive_recorded_counts
        ),
        accept_positive_recovery_predicted_counts_by_action=_sorted_counter(
            accept_positive_predicted_counts
        ),
        target_prediction_counts_by_action=_sorted_counter(target_prediction_counts),
        missing_signal_rows=missing_signal_rows,
        missing_checkpoint_decision_rows=missing_decision_rows,
        file_summaries=file_summaries,
        targets=sorted_targets,
        accept_positive_recovery_decisions=sorted_accept_positive,
    )


def _target_for_row(
    *,
    rows: list[_StrategyOutcomeRow],
    target_index: int,
    signal: StrategySignalRecord,
    decision: StrategyCheckpointSignalDecision | None,
    decision_by_key: dict[tuple[str, int, str], StrategyCheckpointSignalDecision],
    lookback_seconds: float,
    max_windows: int,
) -> StrategyCheckpointPreCollapseTarget:
    target = rows[target_index]
    target_prediction = _prediction_text(decision)
    pre_rows = [
        row
        for row in rows[:target_index]
        if not row.done and 0.0 < target.game_time - row.game_time <= lookback_seconds
    ]
    recovery_windows_all = [
        _window_for_row(
            row,
            target=target,
            decision=decision_by_key.get(_row_key(row)),
        )
        for row in pre_rows
        if _executable_recovery_actions(row)
    ]
    checkpoint_recovery_windows = [
        window
        for window in recovery_windows_all
        if window.predicted_recovery_action is not None
    ]
    checkpoint_executable_windows = [
        window
        for window in recovery_windows_all
        if window.predicted_executable_recovery_action is not None
    ]
    recovery_counts: Counter[str] = Counter(
        window.predicted_recovery_action
        for window in checkpoint_recovery_windows
        if window.predicted_recovery_action is not None
    )
    executable_recovery_counts: Counter[str] = Counter(
        window.predicted_executable_recovery_action
        for window in checkpoint_executable_windows
        if window.predicted_executable_recovery_action is not None
    )
    last_executable = (
        checkpoint_executable_windows[-1] if checkpoint_executable_windows else None
    )

    return StrategyCheckpointPreCollapseTarget(
        source_path=str(target.path),
        source=target.source,
        step=target.step,
        game_time=target.game_time,
        recorded_action=target.action_name,
        recorded_training_use=signal.recommended_training_use,
        recorded_label_quality=signal.label_quality,
        raw_predicted_action=target_prediction[0],
        predicted_action=target_prediction[1],
        prediction_matches_recorded=bool(
            decision is not None and decision.prediction_matches_recorded
        ),
        threat_state=classify_threat_state(target),
        lookback_seconds=lookback_seconds,
        pre_collapse_rows=len(pre_rows),
        pre_collapse_recovery_window_rows=len(recovery_windows_all),
        pre_collapse_checkpoint_recovery_prediction_rows=len(
            checkpoint_recovery_windows
        ),
        pre_collapse_checkpoint_executable_recovery_prediction_rows=len(
            checkpoint_executable_windows
        ),
        checkpoint_predicted_recovery_counts_by_action=_action_counts(
            recovery_counts
        ),
        checkpoint_predicted_executable_recovery_counts_by_action=_action_counts(
            executable_recovery_counts
        ),
        missed_checkpoint_pre_collapse_recovery=bool(
            recovery_windows_all and not checkpoint_executable_windows
        ),
        last_checkpoint_executable_recovery_time=(
            last_executable.game_time if last_executable is not None else None
        ),
        last_checkpoint_executable_recovery_action=(
            last_executable.predicted_executable_recovery_action
            if last_executable is not None
            else None
        ),
        start_metrics=_start_metrics(target),
        recovery_windows=_representative_windows(
            recovery_windows_all,
            max_windows=max_windows,
        ),
    )


def _window_for_row(
    row: _StrategyOutcomeRow,
    *,
    target: _StrategyOutcomeRow,
    decision: StrategyCheckpointSignalDecision | None,
) -> StrategyCheckpointPreCollapseRecoveryWindow:
    executable_recovery = _executable_recovery_actions(row)
    _, predicted_action = _prediction_text(decision)
    predicted_recovery = (
        predicted_action if predicted_action in RECOVERY_ACTION_NAMES else None
    )
    predicted_executable = (
        predicted_action
        if predicted_action in executable_recovery
        and _recovery_executability(row, predicted_action)[0]
        else None
    )
    return StrategyCheckpointPreCollapseRecoveryWindow(
        source_path=str(row.path),
        source=row.source,
        target_step=target.step,
        target_game_time=target.game_time,
        step=row.step,
        game_time=row.game_time,
        seconds_before_target=target.game_time - row.game_time,
        recorded_action=row.action_name,
        raw_predicted_action=_prediction_text(decision)[0],
        predicted_action=predicted_action,
        prediction_was_masked=bool(
            decision is not None and decision.prediction_was_masked
        ),
        threat_state=classify_threat_state(row),
        context=classify_replay_context(row, predicted_action),
        executable_recovery_actions=executable_recovery,
        predicted_recovery_action=predicted_recovery,
        predicted_executable_recovery_action=predicted_executable,
        start_metrics=_start_metrics(row),
    )


def _accept_positive_decision(
    *,
    row: _StrategyOutcomeRow,
    decision: StrategyCheckpointSignalDecision | None,
) -> StrategyCheckpointAcceptPositiveRecoveryDecision:
    raw_predicted, predicted = _prediction_text(decision)
    return StrategyCheckpointAcceptPositiveRecoveryDecision(
        source_path=str(row.path),
        source=row.source,
        step=row.step,
        game_time=row.game_time,
        recorded_action=row.action_name,
        raw_predicted_action=raw_predicted,
        predicted_action=predicted,
        prediction_matches_recorded=bool(
            decision is not None and decision.prediction_matches_recorded
        ),
        prediction_was_masked=bool(
            decision is not None and decision.prediction_was_masked
        ),
        threat_state=classify_threat_state(row),
        context=classify_replay_context(row, predicted),
        start_metrics=_start_metrics(row),
    )


def _representative_windows(
    windows: list[StrategyCheckpointPreCollapseRecoveryWindow],
    *,
    max_windows: int,
) -> list[StrategyCheckpointPreCollapseRecoveryWindow]:
    if max_windows <= 0:
        return []
    predicted = [
        window for window in windows if window.predicted_executable_recovery_action
    ]
    missed = [
        window for window in windows if not window.predicted_executable_recovery_action
    ]
    ordered = [*predicted[-max_windows:], *missed[-max_windows:]]
    deduped = {window.step: window for window in ordered}
    return sorted(deduped.values(), key=lambda window: window.game_time)[
        -max_windows:
    ]


def _is_accept_positive_recovery(signal: StrategySignalRecord) -> bool:
    return (
        signal.recommended_training_use == "accept_positive"
        and signal.recorded_action in RECOVERY_ACTION_NAMES
    )


def _prediction_text(
    decision: StrategyCheckpointSignalDecision | None,
) -> tuple[str, str]:
    if decision is None:
        return "<missing>", "<missing>"
    return decision.raw_predicted_action, decision.predicted_action


def _first_metrics(
    targets: list[StrategyCheckpointPreCollapseTarget],
    positives: list[StrategyCheckpointAcceptPositiveRecoveryDecision],
) -> dict[str, float | str]:
    if targets:
        return targets[0].start_metrics
    if positives:
        return positives[0].start_metrics
    return {}


def _start_metrics(row: _StrategyOutcomeRow) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        field: float(row.observation.get(field, 0.0))
        for field in PRE_COLLAPSE_START_METRICS
    }
    metrics["map_name"] = row.map_name
    metrics["difficulty"] = row.difficulty
    metrics["opponent_race"] = row.opponent_race
    metrics["opponent_ai_build"] = row.opponent_ai_build
    return metrics


def _validate_args(
    *,
    prediction_mode: str,
    lookback_seconds: float,
    max_targets: int,
    max_windows_per_target: int,
    max_accept_positive_decisions: int,
    max_missed_checkpoint_pre_collapse_recovery_rows: int,
    max_missed_checkpoint_pre_collapse_recovery_rate: float,
    max_missed_accept_positive_recovery_rows: int,
    max_missed_accept_positive_recovery_rate: float,
    action_critic_fallback_policy: str,
) -> None:
    if prediction_mode not in PREDICTION_MODES:
        names = ", ".join(PREDICTION_MODES)
        raise ValueError(f"Unknown prediction_mode {prediction_mode!r}; expected {names}")
    if action_critic_fallback_policy not in ACTION_CRITIC_FALLBACK_POLICIES:
        names = ", ".join(ACTION_CRITIC_FALLBACK_POLICIES)
        raise ValueError(
            "Unknown action_critic_fallback_policy "
            f"{action_critic_fallback_policy!r}; expected {names}"
        )
    if lookback_seconds <= 0.0:
        raise ValueError("lookback_seconds must be > 0")
    if max_targets < 0:
        raise ValueError("max_targets must be >= 0")
    if max_windows_per_target < 0:
        raise ValueError("max_windows_per_target must be >= 0")
    if max_accept_positive_decisions < 0:
        raise ValueError("max_accept_positive_decisions must be >= 0")
    if max_missed_checkpoint_pre_collapse_recovery_rows < 0:
        raise ValueError(
            "max_missed_checkpoint_pre_collapse_recovery_rows must be >= 0"
        )
    if not 0.0 <= max_missed_checkpoint_pre_collapse_recovery_rate <= 1.0:
        raise ValueError(
            "max_missed_checkpoint_pre_collapse_recovery_rate must be in [0.0, 1.0]"
        )
    if max_missed_accept_positive_recovery_rows < 0:
        raise ValueError("max_missed_accept_positive_recovery_rows must be >= 0")
    if not 0.0 <= max_missed_accept_positive_recovery_rate <= 1.0:
        raise ValueError(
            "max_missed_accept_positive_recovery_rate must be in [0.0, 1.0]"
        )


def _blocking_reasons(
    *,
    missing_signal_rows: int,
    missing_decision_rows: int,
    missed_checkpoint_pre_collapse_recovery_rows: int,
    missed_checkpoint_pre_collapse_recovery_rate: float,
    max_missed_checkpoint_pre_collapse_recovery_rows: int,
    max_missed_checkpoint_pre_collapse_recovery_rate: float,
    missed_accept_positive_recovery_rows: int,
    missed_accept_positive_recovery_rate: float,
    max_missed_accept_positive_recovery_rows: int,
    max_missed_accept_positive_recovery_rate: float,
) -> list[str]:
    reasons: list[str] = []
    if missing_signal_rows:
        reasons.append("missing_signal_rows")
    if missing_decision_rows:
        reasons.append("missing_checkpoint_decision_rows")
    if (
        missed_checkpoint_pre_collapse_recovery_rows
        > max_missed_checkpoint_pre_collapse_recovery_rows
    ):
        reasons.append("missed_checkpoint_pre_collapse_recovery_rows")
    if (
        missed_checkpoint_pre_collapse_recovery_rate
        > max_missed_checkpoint_pre_collapse_recovery_rate
    ):
        reasons.append("missed_checkpoint_pre_collapse_recovery_rate")
    if missed_accept_positive_recovery_rows > max_missed_accept_positive_recovery_rows:
        reasons.append("missed_accept_positive_recovery_rows")
    if missed_accept_positive_recovery_rate > max_missed_accept_positive_recovery_rate:
        reasons.append("missed_accept_positive_recovery_rate")
    return reasons


def _row_key(row: _StrategyOutcomeRow) -> tuple[str, int, str]:
    return _record_key(row.path, row.step, row.action_name)


def _record_key(path: str | Path, step: int, action: str) -> tuple[str, int, str]:
    return (str(Path(path).resolve()), int(step), str(action))


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _action_counts(counter: Counter[str]) -> dict[str, int]:
    return {
        action: int(counter[action])
        for action in RECOVERY_ACTION_NAMES
        if counter[action]
    }


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in counter.items()))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
