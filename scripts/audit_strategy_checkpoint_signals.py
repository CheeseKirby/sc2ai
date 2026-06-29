"""Audit offline strategy signal quality for a checkpoint policy."""
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
    StrategyCheckpointSignalAudit,
    audit_strategy_checkpoint_signals,
)
from rl.strategy_signal_critic import (  # noqa: E402
    DEFAULT_CRITIC_MAX_BAD_RATE,
    DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    DEFAULT_CRITIC_MIN_SAMPLES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit strategy checkpoint predictions against signal labels"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Strategy checkpoint path.",
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
        help="Optional human-readable audit report output path.",
    )
    parser.add_argument(
        "--show-decisions",
        action="store_true",
        help="Print compact per-row checkpoint decisions.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=PREDICTION_MODES,
        default="raw",
        help="How checkpoint logits are converted into audited predictions.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_strategy_checkpoint_signals(
        args.inputs,
        args.checkpoint,
        prediction_mode=args.prediction_mode,
        critic_min_samples=args.critic_min_samples,
        critic_max_bad_rate=args.critic_max_bad_rate,
        critic_max_veto_negative_rate=args.critic_max_veto_negative_rate,
        action_critic_checkpoint_path=args.action_critic_checkpoint,
        action_critic_threshold=args.action_critic_threshold,
        action_critic_fallback_policy=args.action_critic_fallback_policy,
    )
    report = format_strategy_checkpoint_signal_audit(
        audit,
        show_decisions=args.show_decisions,
    )
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(audit))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_checkpoint_signal_audit(
    audit: StrategyCheckpointSignalAudit,
    *,
    show_decisions: bool = False,
) -> str:
    """Return a compact human-readable checkpoint signal audit report."""
    lines = [
        "Strategy checkpoint signal audit",
        f"inputs: {', '.join(audit.inputs)}",
        f"checkpoint: {audit.checkpoint_path}",
        f"prediction_mode: {audit.prediction_mode}",
        f"action_critic_checkpoint: {audit.action_critic_checkpoint_path or '<none>'}",
        f"action_critic_threshold: {_optional_float(audit.action_critic_threshold)}",
        f"action_critic_fallback_policy: {audit.action_critic_fallback_policy or '<none>'}",
        f"files: {audit.files}",
        f"rows: {audit.rows}",
        f"signal_healthy: {_bool_text(audit.signal_healthy)}",
        f"blocking_reasons: {_inline_items(audit.blocking_reasons)}",
        f"warnings: {_inline_items(audit.warnings)}",
        "prediction_match: "
        f"{audit.prediction_matches_recorded}/{audit.rows} "
        f"ratio={audit.prediction_match_ratio:.3f}",
        "predicted_non_executable: "
        f"{audit.predicted_non_executable_rows}/{audit.rows} "
        f"ratio={audit.predicted_non_executable_ratio:.3f}",
        "raw_predicted_non_executable: "
        f"{audit.raw_predicted_non_executable_rows}/{audit.rows} "
        f"ratio={audit.raw_predicted_non_executable_ratio:.3f}",
        "masked_prediction_changes: "
        f"{audit.masked_prediction_changes}/{audit.rows} "
        f"ratio={audit.masked_prediction_change_ratio:.3f}",
        f"critic_vetoed_candidates: {audit.critic_vetoed_candidates}",
        "action_critic_selected_unsafe_probability: "
        f"avg={_optional_float(audit.action_critic_selected_unsafe_probability_avg)} "
        f"max={_optional_float(audit.action_critic_selected_unsafe_probability_max)}",
        "action_critic_vetoed_probability: "
        f"avg={_optional_float(audit.action_critic_vetoed_probability_avg)} "
        f"max={_optional_float(audit.action_critic_vetoed_probability_max)}",
        f"action_critic_fallback_rows: {audit.action_critic_fallback_rows}",
        "action_critic_fallback_policies: "
        f"{_inline_counts(audit.action_critic_fallback_policy_counts)}",
        "accept_positive_match: "
        f"{audit.accept_positive_prediction_matches}/"
        f"{audit.accept_positive_rows} "
        f"ratio={audit.accept_positive_prediction_match_ratio:.3f}",
        "bad_recorded_match: "
        f"{audit.bad_recorded_prediction_matches}/{audit.bad_recorded_rows} "
        f"ratio={audit.bad_recorded_prediction_match_ratio:.3f}",
        "veto_negative_match: "
        f"{audit.veto_negative_prediction_matches}/{audit.veto_negative_rows} "
        f"ratio={audit.veto_negative_prediction_match_ratio:.3f}",
        "drop_non_executable_match: "
        f"{audit.drop_non_executable_prediction_matches}/"
        f"{audit.drop_non_executable_rows} "
        f"ratio={audit.drop_non_executable_prediction_match_ratio:.3f}",
        "action_space_exhausted_match: "
        f"{audit.action_space_exhausted_prediction_matches}/"
        f"{audit.action_space_exhausted_rows} "
        f"ratio={audit.action_space_exhausted_prediction_match_ratio:.3f}",
        f"recorded_training_use: {_inline_counts(audit.recorded_training_use_counts)}",
        f"recorded_label_quality: {_inline_counts(audit.recorded_label_quality_counts)}",
        f"recorded_action_counts: {_inline_counts(audit.recorded_action_counts_by_name)}",
        "raw_predicted_action_counts: "
        f"{_inline_counts(audit.raw_predicted_action_counts_by_name)}",
        f"predicted_action_counts: {_inline_counts(audit.predicted_action_counts_by_name)}",
        f"raw_predicted_blockers: {_inline_counts(audit.raw_predicted_blocker_counts)}",
        f"predicted_blockers: {_inline_counts(audit.predicted_blocker_counts)}",
        f"critic_veto_reasons: {_inline_counts(audit.critic_veto_reason_counts)}",
        f"critic_veto_actions: {_inline_counts(audit.critic_veto_action_counts)}",
        "predicted_non_executable_by_training_use: "
        f"{_inline_counts(audit.predicted_non_executable_by_recorded_training_use)}",
    ]
    if show_decisions:
        lines.append("decisions:")
        if not audit.decisions:
            lines.append("  <none>")
        for decision in audit.decisions:
            lines.append(
                f"  {Path(decision.path).name}: step={decision.step} "
                f"t={decision.game_time:.1f} "
                f"recorded={decision.recorded_action} "
                f"raw={decision.raw_predicted_action} "
                f"predicted={decision.predicted_action} "
                f"use={decision.recorded_training_use} "
                f"quality={decision.recorded_label_quality} "
                f"match={_bool_text(decision.prediction_matches_recorded)} "
                f"masked={_bool_text(decision.prediction_was_masked)} "
                f"executable={_bool_text(decision.predicted_immediate_executable)} "
                f"blocker={decision.predicted_blocker or '<none>'} "
                f"critic_vetoed={_inline_items(decision.critic_vetoed_actions)} "
                "action_critic_candidates="
                f"{_inline_items(decision.action_critic_candidate_actions)} "
                "action_critic_selected_unsafe="
                f"{_optional_float(decision.action_critic_selected_unsafe_probability)} "
                "action_critic_fallback="
                f"{_bool_text(decision.action_critic_fallback_selected)} "
                "fallback_policy="
                f"{decision.action_critic_fallback_policy_used or '<none>'} "
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


if __name__ == "__main__":
    raise SystemExit(main())
