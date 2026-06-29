"""Sweep action critic thresholds for strategy checkpoint audits."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_action_critic_threshold_sweep import (  # noqa: E402
    DEFAULT_ACTION_CRITIC_SWEEP_THRESHOLDS,
    StrategyActionCriticThresholdSweep,
    sweep_strategy_action_critic_thresholds,
)
from rl.strategy_checkpoint_signal_audit import ACTION_CRITIC_FALLBACK_POLICIES  # noqa: E402
from rl.strategy_signal_critic import (  # noqa: E402
    DEFAULT_CRITIC_MAX_BAD_RATE,
    DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    DEFAULT_CRITIC_MIN_SAMPLES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep action critic thresholds for strategy checkpoint audits"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Strategy checkpoint path.",
    )
    parser.add_argument(
        "--action-critic-checkpoint",
        type=Path,
        required=True,
        help="Strategy action critic checkpoint.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_ACTION_CRITIC_SWEEP_THRESHOLDS),
        help="Unsafe probability thresholds to audit.",
    )
    parser.add_argument(
        "--fallback-policies",
        nargs="+",
        choices=ACTION_CRITIC_FALLBACK_POLICIES,
        default=["lowest-risk"],
        help="Fallback policies to sweep.",
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
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable sweep output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable sweep report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sweep = sweep_strategy_action_critic_thresholds(
        args.inputs,
        args.checkpoint,
        args.action_critic_checkpoint,
        thresholds=args.thresholds,
        fallback_policies=args.fallback_policies,
        critic_min_samples=args.critic_min_samples,
        critic_max_bad_rate=args.critic_max_bad_rate,
        critic_max_veto_negative_rate=args.critic_max_veto_negative_rate,
    )
    report = format_strategy_action_critic_threshold_sweep(sweep)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(sweep))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_action_critic_threshold_sweep(
    sweep: StrategyActionCriticThresholdSweep,
) -> str:
    """Return a compact human-readable threshold sweep report."""
    selected = sweep.selected_trial
    lines = [
        "Strategy action critic threshold sweep",
        f"inputs: {', '.join(sweep.inputs)}",
        f"checkpoint: {sweep.checkpoint_path}",
        f"action_critic_checkpoint: {sweep.action_critic_checkpoint_path}",
        f"thresholds: {', '.join(f'{value:.3f}' for value in sweep.thresholds)}",
        f"fallback_policies: {', '.join(sweep.fallback_policies)}",
        f"recommendation: {sweep.recommendation}",
        f"blocking_reasons: {_inline_items(sweep.blocking_reasons)}",
        "selected: "
        + (
            "<none>"
            if selected is None
            else (
                f"threshold={selected.threshold:.3f} "
                f"fallback={selected.fallback_policy} "
                f"veto={selected.veto_negative_prediction_matches}/"
                f"{selected.veto_negative_rows} "
                f"fallback_rows={selected.action_critic_fallback_rows} "
                f"accept={selected.accept_positive_prediction_matches}/"
                f"{selected.accept_positive_rows} "
                f"nonexec={selected.predicted_non_executable_rows} "
                f"match_ratio={selected.prediction_match_ratio:.3f}"
            )
        ),
        "trials:",
    ]
    if not sweep.trials:
        lines.append("  <none>")
    for trial in sweep.trials:
        lines.append(
            f"  threshold={trial.threshold:.3f} "
            f"fallback={trial.fallback_policy} "
            f"healthy={_bool_text(trial.signal_healthy)} "
            f"veto={trial.veto_negative_prediction_matches}/"
            f"{trial.veto_negative_rows} "
            f"drop={trial.drop_non_executable_prediction_matches}/"
            f"{trial.drop_non_executable_rows} "
            f"nonexec={trial.predicted_non_executable_rows} "
            f"fallback_rows={trial.action_critic_fallback_rows} "
            f"accept={trial.accept_positive_prediction_matches}/"
            f"{trial.accept_positive_rows} "
            f"match={trial.prediction_matches_recorded}/{trial.rows} "
            f"match_ratio={trial.prediction_match_ratio:.3f} "
            "selected_unsafe_avg="
            f"{_optional_float(trial.action_critic_selected_unsafe_probability_avg)} "
            "selected_unsafe_max="
            f"{_optional_float(trial.action_critic_selected_unsafe_probability_max)} "
            f"reasons={_inline_items(trial.blocking_reasons)}"
        )
    return "\n".join(lines)


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
