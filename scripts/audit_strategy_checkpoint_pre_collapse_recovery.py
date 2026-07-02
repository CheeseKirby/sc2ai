"""Audit checkpoint predictions on pre-collapse recovery slices."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_checkpoint_pre_collapse_recovery_audit import (  # noqa: E402
    DEFAULT_MAX_MISSED_ACCEPT_POSITIVE_RECOVERY_RATE,
    DEFAULT_MAX_MISSED_ACCEPT_POSITIVE_RECOVERY_ROWS,
    DEFAULT_MAX_MISSED_CHECKPOINT_PRE_COLLAPSE_RECOVERY_RATE,
    DEFAULT_MAX_MISSED_CHECKPOINT_PRE_COLLAPSE_RECOVERY_ROWS,
    StrategyCheckpointPreCollapseRecoveryAudit,
    audit_strategy_checkpoint_pre_collapse_recovery,
)
from rl.strategy_checkpoint_signal_audit import (  # noqa: E402
    ACTION_CRITIC_FALLBACK_POLICIES,
    PREDICTION_MODES,
)
from rl.strategy_pre_collapse_recovery_analysis import (  # noqa: E402
    DEFAULT_LOOKBACK_SECONDS,
)
from rl.strategy_signal_critic import (  # noqa: E402
    DEFAULT_CRITIC_MAX_BAD_RATE,
    DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    DEFAULT_CRITIC_MIN_SAMPLES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit checkpoint predictions on pre-collapse recovery slices"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Strategy checkpoint path.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=PREDICTION_MODES,
        default="executable-mask",
        help="How checkpoint logits are converted into audited predictions.",
    )
    parser.add_argument(
        "--lookback-seconds",
        type=float,
        default=DEFAULT_LOOKBACK_SECONDS,
        help="Seconds before each target row to inspect for recovery predictions.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=20,
        help="Maximum target rows to include in the report.",
    )
    parser.add_argument(
        "--max-windows-per-target",
        type=int,
        default=6,
        help="Maximum representative recovery windows per target row.",
    )
    parser.add_argument(
        "--max-accept-positive-decisions",
        type=int,
        default=20,
        help="Maximum accept-positive recovery decisions to include in the report.",
    )
    parser.add_argument(
        "--critic-min-samples",
        type=int,
        default=DEFAULT_CRITIC_MIN_SAMPLES,
        help="Minimum samples required before signal-risk critic can veto a group.",
    )
    parser.add_argument(
        "--critic-max-bad-rate",
        type=float,
        default=DEFAULT_CRITIC_MAX_BAD_RATE,
        help="Maximum bad-label rate allowed before signal-risk critic vetoes.",
    )
    parser.add_argument(
        "--critic-max-veto-negative-rate",
        type=float,
        default=DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
        help="Maximum veto-negative rate allowed before signal-risk critic vetoes.",
    )
    parser.add_argument(
        "--action-critic-checkpoint",
        type=Path,
        default=None,
        help="Strategy action critic checkpoint for action-critic-mask mode.",
    )
    parser.add_argument(
        "--action-critic-threshold",
        type=float,
        default=0.5,
        help="Unsafe-probability threshold for action-critic-mask vetoes.",
    )
    parser.add_argument(
        "--action-critic-fallback-policy",
        choices=ACTION_CRITIC_FALLBACK_POLICIES,
        default="lowest-risk",
        help="Fallback policy when action critic vetoes every executable candidate.",
    )
    parser.add_argument(
        "--max-missed-checkpoint-pre-collapse-recovery-rows",
        type=int,
        default=DEFAULT_MAX_MISSED_CHECKPOINT_PRE_COLLAPSE_RECOVERY_ROWS,
        help="Allowed missed target rows before recommendation becomes hold.",
    )
    parser.add_argument(
        "--max-missed-checkpoint-pre-collapse-recovery-rate",
        type=float,
        default=DEFAULT_MAX_MISSED_CHECKPOINT_PRE_COLLAPSE_RECOVERY_RATE,
        help="Allowed missed target-row rate before recommendation becomes hold.",
    )
    parser.add_argument(
        "--max-missed-accept-positive-recovery-rows",
        type=int,
        default=DEFAULT_MAX_MISSED_ACCEPT_POSITIVE_RECOVERY_ROWS,
        help="Allowed missed positive recovery rows before recommendation is hold.",
    )
    parser.add_argument(
        "--max-missed-accept-positive-recovery-rate",
        type=float,
        default=DEFAULT_MAX_MISSED_ACCEPT_POSITIVE_RECOVERY_RATE,
        help="Allowed missed positive recovery rate before recommendation is hold.",
    )
    parser.add_argument(
        "--fail-on-hold",
        action="store_true",
        help="Return exit code 1 when the gate recommendation is hold.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable audit output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable audit report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_strategy_checkpoint_pre_collapse_recovery(
        args.inputs,
        args.checkpoint,
        prediction_mode=args.prediction_mode,
        lookback_seconds=args.lookback_seconds,
        max_targets=args.max_targets,
        max_windows_per_target=args.max_windows_per_target,
        max_accept_positive_decisions=args.max_accept_positive_decisions,
        critic_min_samples=args.critic_min_samples,
        critic_max_bad_rate=args.critic_max_bad_rate,
        critic_max_veto_negative_rate=args.critic_max_veto_negative_rate,
        action_critic_checkpoint_path=args.action_critic_checkpoint,
        action_critic_threshold=args.action_critic_threshold,
        action_critic_fallback_policy=args.action_critic_fallback_policy,
        max_missed_checkpoint_pre_collapse_recovery_rows=(
            args.max_missed_checkpoint_pre_collapse_recovery_rows
        ),
        max_missed_checkpoint_pre_collapse_recovery_rate=(
            args.max_missed_checkpoint_pre_collapse_recovery_rate
        ),
        max_missed_accept_positive_recovery_rows=(
            args.max_missed_accept_positive_recovery_rows
        ),
        max_missed_accept_positive_recovery_rate=(
            args.max_missed_accept_positive_recovery_rate
        ),
    )
    report = format_strategy_checkpoint_pre_collapse_recovery_audit(audit)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(audit))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    if args.fail_on_hold and audit.recommendation == "hold":
        return 1
    return 0


def format_strategy_checkpoint_pre_collapse_recovery_audit(
    audit: StrategyCheckpointPreCollapseRecoveryAudit,
) -> str:
    """Return a compact human-readable checkpoint recovery slice report."""
    lines = [
        "Strategy checkpoint pre-collapse recovery audit",
        f"inputs: {', '.join(audit.inputs)}",
        f"checkpoint: {audit.checkpoint_path}",
        f"prediction_mode: {audit.prediction_mode}",
        f"action_critic_checkpoint: {audit.action_critic_checkpoint_path or '<none>'}",
        f"action_critic_threshold: {_optional_float(audit.action_critic_threshold)}",
        "action_critic_fallback_policy: "
        f"{audit.action_critic_fallback_policy or '<none>'}",
        f"lookback_seconds: {audit.lookback_seconds:.1f}",
        f"target_training_uses: {_inline_items(audit.target_training_uses)}",
        f"recovery_actions: {_inline_items(audit.recovery_action_names)}",
        f"recommendation: {audit.recommendation}",
        f"blocking_reasons: {_inline_items(audit.blocking_reasons)}",
        f"warnings: {_inline_items(audit.warnings)}",
        "gate_thresholds: "
        "max_missed_checkpoint_pre_collapse_recovery_rows="
        f"{audit.max_missed_checkpoint_pre_collapse_recovery_rows}, "
        "max_missed_checkpoint_pre_collapse_recovery_rate="
        f"{audit.max_missed_checkpoint_pre_collapse_recovery_rate:.3f}, "
        "max_missed_accept_positive_recovery_rows="
        f"{audit.max_missed_accept_positive_recovery_rows}, "
        "max_missed_accept_positive_recovery_rate="
        f"{audit.max_missed_accept_positive_recovery_rate:.3f}",
        f"files: {audit.files}",
        f"rows: {audit.rows}",
        f"training_rows: {audit.training_rows}",
        f"target_rows: {audit.target_rows}",
        "rows_with_pre_collapse_recovery_window: "
        f"{audit.rows_with_pre_collapse_recovery_window}/{audit.target_rows}",
        "rows_with_checkpoint_pre_collapse_recovery: "
        f"{audit.rows_with_checkpoint_pre_collapse_recovery}/{audit.target_rows}",
        "rows_with_checkpoint_pre_collapse_executable_recovery: "
        f"{audit.rows_with_checkpoint_pre_collapse_executable_recovery}/"
        f"{audit.target_rows}",
        "missed_checkpoint_pre_collapse_recovery_rows: "
        f"{audit.missed_checkpoint_pre_collapse_recovery_rows}/"
        f"{audit.target_rows}",
        "missed_checkpoint_pre_collapse_recovery_rate: "
        f"{audit.missed_checkpoint_pre_collapse_recovery_rate:.3f}",
        "no_pre_collapse_recovery_window_rows: "
        f"{audit.no_pre_collapse_recovery_window_rows}/{audit.target_rows}",
        "accept_positive_recovery_match: "
        f"{audit.accept_positive_recovery_matches}/"
        f"{audit.accept_positive_recovery_rows} "
        f"ratio={audit.accept_positive_recovery_match_rate:.3f}",
        "missed_accept_positive_recovery_rows: "
        f"{audit.missed_accept_positive_recovery_rows}/"
        f"{audit.accept_positive_recovery_rows}",
        "checkpoint_predicted_recovery_counts_by_action: "
        f"{_inline_counts(audit.checkpoint_predicted_recovery_counts_by_action)}",
        "checkpoint_predicted_executable_recovery_counts_by_action: "
        f"{_inline_counts(audit.checkpoint_predicted_executable_recovery_counts_by_action)}",
        "accept_positive_recovery_recorded_counts_by_action: "
        f"{_inline_counts(audit.accept_positive_recovery_recorded_counts_by_action)}",
        "accept_positive_recovery_predicted_counts_by_action: "
        f"{_inline_counts(audit.accept_positive_recovery_predicted_counts_by_action)}",
        "target_prediction_counts_by_action: "
        f"{_inline_counts(audit.target_prediction_counts_by_action)}",
        f"missing_signal_rows: {audit.missing_signal_rows}",
        f"missing_checkpoint_decision_rows: {audit.missing_checkpoint_decision_rows}",
        "files:",
    ]
    if not audit.file_summaries:
        lines.append("  <none>")
    for summary in audit.file_summaries:
        lines.append(
            f"  {Path(summary.path).name}: targets={summary.target_rows} "
            f"window={summary.rows_with_pre_collapse_recovery_window} "
            "checkpoint_exec="
            f"{summary.rows_with_checkpoint_pre_collapse_executable_recovery} "
            f"missed={summary.missed_checkpoint_pre_collapse_recovery_rows} "
            "accept_positive="
            f"{summary.accept_positive_recovery_matches}/"
            f"{summary.accept_positive_recovery_rows}"
        )
    lines.append("targets:")
    if not audit.targets:
        lines.append("  <none>")
    for target in audit.targets:
        lines.append(
            f"  {Path(target.source_path).name}: step={target.step} "
            f"t={target.game_time:.1f} use={target.recorded_training_use} "
            f"quality={target.recorded_label_quality} "
            f"recorded={target.recorded_action} "
            f"predicted={target.predicted_action} "
            f"threat={target.threat_state} "
            f"pre_rows={target.pre_collapse_rows} "
            f"recovery_windows={target.pre_collapse_recovery_window_rows} "
            "checkpoint_exec_windows="
            f"{target.pre_collapse_checkpoint_executable_recovery_prediction_rows} "
            f"missed={_bool_text(target.missed_checkpoint_pre_collapse_recovery)} "
            "last_checkpoint_exec="
            f"{_time(target.last_checkpoint_executable_recovery_time)}:"
            f"{target.last_checkpoint_executable_recovery_action or '<none>'}"
        )
        for window in target.recovery_windows:
            lines.append(
                f"    window step={window.step} t={window.game_time:.1f} "
                f"before={window.seconds_before_target:.1f}s "
                f"recorded={window.recorded_action} "
                f"predicted={window.predicted_action} "
                f"exec={_inline_items(window.executable_recovery_actions)} "
                "predicted_exec="
                f"{window.predicted_executable_recovery_action or '<none>'} "
                f"context={window.context} threat={window.threat_state}"
            )
    lines.append("accept_positive_recovery:")
    if not audit.accept_positive_recovery_decisions:
        lines.append("  <none>")
    for decision in audit.accept_positive_recovery_decisions:
        lines.append(
            f"  {Path(decision.source_path).name}: step={decision.step} "
            f"t={decision.game_time:.1f} recorded={decision.recorded_action} "
            f"predicted={decision.predicted_action} "
            f"match={_bool_text(decision.prediction_matches_recorded)} "
            f"context={decision.context} threat={decision.threat_state}"
        )
    return "\n".join(lines)


def _inline_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _optional_float(value: float | None) -> str:
    if value is None:
        return "<none>"
    return f"{value:.3f}"


def _time(value: float | None) -> str:
    if value is None:
        return "<none>"
    return f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
