"""Analyze row-level errors that block strategy checkpoint promotion."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_gate_error_analysis import (  # noqa: E402
    StrategyGateErrorAnalysis,
    analyze_strategy_gate_errors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze errors that block strategy checkpoint promotion"
    )
    parser.add_argument("audits", nargs="+", type=Path, help="Audit JSON files")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable analysis output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable analysis report path.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=12,
        help="Maximum representative issue examples to include.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = analyze_strategy_gate_errors(
        args.audits,
        max_examples=args.max_examples,
    )
    report = format_strategy_gate_error_analysis(analysis)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(analysis))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_gate_error_analysis(
    analysis: StrategyGateErrorAnalysis,
) -> str:
    """Return a compact human-readable strategy gate error report."""
    lines = [
        "Strategy gate error analysis",
        f"audits: {analysis.audits}",
        f"rows: {analysis.rows}",
        f"issue_rows: {analysis.issue_rows}",
        f"issue_counts: {_inline_counts(analysis.issue_counts)}",
        f"veto_negative_matches: {analysis.veto_negative_matches}",
        "action_space_exhausted_matches: "
        f"{analysis.action_space_exhausted_matches}",
        f"drop_non_executable_matches: {analysis.drop_non_executable_matches}",
        f"predicted_non_executable_rows: {analysis.predicted_non_executable_rows}",
        f"action_critic_fallback_rows: {analysis.action_critic_fallback_rows}",
        "fallback_and_veto_negative_rows: "
        f"{analysis.fallback_and_veto_negative_rows}",
        "fallback_and_accept_positive_rows: "
        f"{analysis.fallback_and_accept_positive_rows}",
        "action_critic_selected_unsafe_probability: "
        f"avg={_optional_float(analysis.action_critic_selected_unsafe_probability_avg)} "
        f"max={_optional_float(analysis.action_critic_selected_unsafe_probability_max)}",
        f"issue_by_predicted_action: {_inline_counts(analysis.issue_by_predicted_action)}",
        f"issue_by_recorded_action: {_inline_counts(analysis.issue_by_recorded_action)}",
        f"issue_by_training_use: {_inline_counts(analysis.issue_by_training_use)}",
        f"issue_by_threat_state: {_inline_counts(analysis.issue_by_threat_state)}",
        f"issue_by_context: {_inline_counts(analysis.issue_by_context)}",
        "fallback_by_predicted_action: "
        f"{_inline_counts(analysis.fallback_by_predicted_action)}",
        f"fallback_by_training_use: {_inline_counts(analysis.fallback_by_training_use)}",
        f"fallback_by_threat_state: {_inline_counts(analysis.fallback_by_threat_state)}",
        f"fallback_by_context: {_inline_counts(analysis.fallback_by_context)}",
        "fallback_by_candidate_action_count: "
        f"{_inline_counts(analysis.fallback_by_candidate_action_count)}",
        "fallback_by_candidate_action_set: "
        f"{_inline_counts(analysis.fallback_by_candidate_action_set)}",
        "fallback_single_candidate_action: "
        f"{_inline_counts(analysis.fallback_single_candidate_action)}",
        "veto_match_by_predicted_action: "
        f"{_inline_counts(analysis.veto_match_by_predicted_action)}",
        "veto_match_by_recorded_action: "
        f"{_inline_counts(analysis.veto_match_by_recorded_action)}",
        "veto_match_by_threat_state: "
        f"{_inline_counts(analysis.veto_match_by_threat_state)}",
        f"veto_match_by_context: {_inline_counts(analysis.veto_match_by_context)}",
        "veto_match_by_candidate_action_set: "
        f"{_inline_counts(analysis.veto_match_by_candidate_action_set)}",
        "action_space_match_by_predicted_action: "
        f"{_inline_counts(analysis.action_space_match_by_predicted_action)}",
        "action_space_match_by_threat_state: "
        f"{_inline_counts(analysis.action_space_match_by_threat_state)}",
        "action_space_match_by_candidate_action_set: "
        f"{_inline_counts(analysis.action_space_match_by_candidate_action_set)}",
        f"non_executable_by_blocker: {_inline_counts(analysis.non_executable_by_blocker)}",
        f"critic_veto_actions: {_inline_counts(analysis.critic_veto_action_counts)}",
        f"critic_veto_reasons: {_inline_counts(analysis.critic_veto_reason_counts)}",
        f"warnings: {_inline_counts(analysis.warning_counts)}",
        "examples:",
    ]
    if not analysis.examples:
        lines.append("  <none>")
    for example in analysis.examples:
        lines.append(
            f"  {Path(example.source_path).name}: step={example.step} "
            f"t={example.game_time:.1f} "
            f"recorded={example.recorded_action} "
            f"raw={example.raw_predicted_action} "
            f"predicted={example.predicted_action} "
            f"use={example.recorded_training_use} "
            f"quality={example.recorded_label_quality} "
            f"issues={_inline_items(example.issues)} "
            "selected_unsafe="
            f"{_optional_float(example.action_critic_selected_unsafe_probability)} "
            f"context={example.context} threat={example.threat_state} "
            f"critic_vetoed={_inline_items(example.critic_vetoed_actions)}"
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


def _optional_float(value: float | None) -> str:
    if value is None:
        return "<none>"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
