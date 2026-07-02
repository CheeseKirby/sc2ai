"""Audit checkpoint recovery-positive predictions by context slice."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_checkpoint_signal_audit import (  # noqa: E402
    ACTION_CRITIC_FALLBACK_POLICIES,
    PREDICTION_MODES,
)
from rl.strategy_filtered_datasets import (  # noqa: E402
    RECOVERY_ACCEPT_POSITIVE_CONTEXT_FILTERS,
)
from rl.strategy_recovery_context_audit import (  # noqa: E402
    DEFAULT_MAX_CONTEXT_CROSS_ACTION_RATE,
    DEFAULT_MAX_CONTEXT_CROSS_ACTION_ROWS,
    DEFAULT_MAX_CONTEXT_MISSED_RATE,
    DEFAULT_MAX_CONTEXT_MISSED_ROWS,
    StrategyRecoveryContextAudit,
    audit_strategy_recovery_context,
)
from rl.strategy_signal_critic import (  # noqa: E402
    DEFAULT_CRITIC_MAX_BAD_RATE,
    DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    DEFAULT_CRITIC_MIN_SAMPLES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit checkpoint recovery positives by context slice"
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
        "--context-filter",
        choices=RECOVERY_ACCEPT_POSITIVE_CONTEXT_FILTERS,
        default="pre-collapse-recovery",
        help="Recovery-positive context slice to audit.",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=40,
        help="Maximum representative decisions to include in the report.",
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
        "--max-context-missed-rows",
        type=int,
        default=DEFAULT_MAX_CONTEXT_MISSED_ROWS,
        help="Allowed missed context-positive rows before recommendation is hold.",
    )
    parser.add_argument(
        "--max-context-missed-rate",
        type=float,
        default=DEFAULT_MAX_CONTEXT_MISSED_RATE,
        help="Allowed missed context-positive rate before recommendation is hold.",
    )
    parser.add_argument(
        "--max-context-cross-action-rows",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CROSS_ACTION_ROWS,
        help="Allowed context-positive cross-action confusions before hold.",
    )
    parser.add_argument(
        "--max-context-cross-action-rate",
        type=float,
        default=DEFAULT_MAX_CONTEXT_CROSS_ACTION_RATE,
        help="Allowed context-positive cross-action confusion rate before hold.",
    )
    parser.add_argument(
        "--fail-on-hold",
        action="store_true",
        help="Return exit code 1 when the recommendation is hold.",
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
    audit = audit_strategy_recovery_context(
        args.inputs,
        args.checkpoint,
        prediction_mode=args.prediction_mode,
        context_filter=args.context_filter,
        max_decisions=args.max_decisions,
        critic_min_samples=args.critic_min_samples,
        critic_max_bad_rate=args.critic_max_bad_rate,
        critic_max_veto_negative_rate=args.critic_max_veto_negative_rate,
        action_critic_checkpoint_path=args.action_critic_checkpoint,
        action_critic_threshold=args.action_critic_threshold,
        action_critic_fallback_policy=args.action_critic_fallback_policy,
        max_context_missed_rows=args.max_context_missed_rows,
        max_context_missed_rate=args.max_context_missed_rate,
        max_context_cross_action_rows=args.max_context_cross_action_rows,
        max_context_cross_action_rate=args.max_context_cross_action_rate,
    )
    report = format_strategy_recovery_context_audit(audit)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(audit))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    if args.fail_on_hold and audit.recommendation == "hold":
        return 1
    return 0


def format_strategy_recovery_context_audit(
    audit: StrategyRecoveryContextAudit,
) -> str:
    """Return a compact human-readable recovery context report."""
    lines = [
        "Strategy recovery context audit",
        f"inputs: {', '.join(audit.inputs)}",
        f"checkpoint: {audit.checkpoint_path}",
        f"prediction_mode: {audit.prediction_mode}",
        f"context_filter: {audit.context_filter}",
        f"action_critic_checkpoint: {audit.action_critic_checkpoint_path or '<none>'}",
        f"action_critic_threshold: {_optional_float(audit.action_critic_threshold)}",
        "action_critic_fallback_policy: "
        f"{audit.action_critic_fallback_policy or '<none>'}",
        f"recommendation: {audit.recommendation}",
        f"blocking_reasons: {_inline_items(audit.blocking_reasons)}",
        f"warnings: {_inline_items(audit.warnings)}",
        "gate_thresholds: "
        f"max_context_missed_rows={audit.max_context_missed_rows}, "
        f"max_context_missed_rate={audit.max_context_missed_rate:.3f}, "
        "max_context_cross_action_rows="
        f"{audit.max_context_cross_action_rows}, "
        "max_context_cross_action_rate="
        f"{audit.max_context_cross_action_rate:.3f}",
        f"files: {audit.files}",
        f"rows: {audit.rows}",
        "accept_positive_recovery_match: "
        f"{audit.accept_positive_recovery_matches}/"
        f"{audit.accept_positive_recovery_rows} "
        f"ratio={audit.accept_positive_recovery_match_rate:.3f}",
        "context_matched_accept_positive_recovery_match: "
        f"{audit.context_matched_accept_positive_recovery_matches}/"
        f"{audit.context_matched_accept_positive_recovery_rows} "
        f"ratio={audit.context_matched_accept_positive_recovery_match_rate:.3f}",
        "context_missed_accept_positive_recovery_rows: "
        f"{audit.context_missed_accept_positive_recovery_rows}/"
        f"{audit.context_matched_accept_positive_recovery_rows} "
        f"ratio={audit.context_missed_accept_positive_recovery_rate:.3f}",
        "context_skipped_accept_positive_recovery_rows: "
        f"{audit.context_skipped_accept_positive_recovery_rows}",
        "cross_action_confusion_rows: "
        f"{audit.cross_action_confusion_rows}/"
        f"{audit.accept_positive_recovery_rows} "
        f"ratio={audit.cross_action_confusion_rate:.3f}",
        "context_matched_cross_action_confusion_rows: "
        f"{audit.context_matched_cross_action_confusion_rows}/"
        f"{audit.context_matched_accept_positive_recovery_rows} "
        f"ratio={audit.context_matched_cross_action_confusion_rate:.3f}",
        "recorded_counts_by_action: "
        f"{_inline_counts(audit.recorded_counts_by_action)}",
        "predicted_counts_by_action: "
        f"{_inline_counts(audit.predicted_counts_by_action)}",
        "context_matched_recorded_counts_by_action: "
        f"{_inline_counts(audit.context_matched_recorded_counts_by_action)}",
        "context_matched_predicted_counts_by_action: "
        f"{_inline_counts(audit.context_matched_predicted_counts_by_action)}",
        "action_summaries:",
    ]
    if not audit.action_summaries:
        lines.append("  <none>")
    for summary in audit.action_summaries:
        lines.append(
            f"  {summary.recorded_action}: rows={summary.rows} "
            f"match={summary.matches}/{summary.rows} "
            f"context_match={summary.context_matched_matches}/"
            f"{summary.context_matched_rows} "
            "context_cross_action="
            f"{summary.context_matched_cross_action_confusion_rows}/"
            f"{summary.context_matched_rows} "
            f"predicted={_inline_counts(summary.predicted_counts_by_action)}"
        )
    lines.append("confusion:")
    _append_confusion(lines, audit.confusion_counts_by_recorded_then_predicted)
    lines.append("context_matched_confusion:")
    _append_confusion(
        lines,
        audit.context_matched_confusion_counts_by_recorded_then_predicted,
    )
    lines.append("decisions:")
    if not audit.decisions:
        lines.append("  <none>")
    for decision in audit.decisions:
        lines.append(
            f"  {Path(decision.path).name}: step={decision.step} "
            f"t={decision.game_time:.1f} recorded={decision.recorded_action} "
            f"predicted={decision.predicted_action} "
            f"match={_bool_text(decision.prediction_matches_recorded)} "
            "context_match="
            f"{_bool_text(decision.context_matches_recorded_action)} "
            "cross_action="
            f"{_bool_text(decision.is_cross_action_confusion)} "
            f"recorded_context={decision.recorded_action_replay_context} "
            f"predicted_context={decision.predicted_action_replay_context} "
            f"threat={decision.threat_state}"
        )
    return "\n".join(lines)


def _append_confusion(lines: list[str], confusion: dict[str, dict[str, int]]) -> None:
    if not confusion:
        lines.append("  <none>")
        return
    for recorded, predicted in confusion.items():
        lines.append(f"  {recorded}: {_inline_counts(predicted)}")


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


if __name__ == "__main__":
    raise SystemExit(main())
