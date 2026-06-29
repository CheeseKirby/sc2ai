"""Analyze veto-negative strategy signal clusters."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_veto_negative_analysis import (  # noqa: E402
    StrategyVetoNegativeAnalysis,
    analyze_strategy_veto_negatives,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze veto-negative strategy signal clusters"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--audit",
        dest="audits",
        action="append",
        type=Path,
        default=[],
        help="Optional checkpoint signal audit JSON to join against.",
    )
    parser.add_argument(
        "--include-before-filter-candidates",
        action="store_true",
        help="Include before-filter candidate rows in the signal dataset.",
    )
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
        help="Maximum representative veto examples to include.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = analyze_strategy_veto_negatives(
        args.inputs,
        audit_paths=args.audits,
        max_examples=args.max_examples,
        include_before_filter_candidates=args.include_before_filter_candidates,
    )
    report = format_strategy_veto_negative_analysis(analysis)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(analysis))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_veto_negative_analysis(
    analysis: StrategyVetoNegativeAnalysis,
) -> str:
    """Return a compact human-readable veto-negative cluster report."""
    lines = [
        "Strategy veto-negative analysis",
        f"inputs: {', '.join(analysis.inputs)}",
        f"audits: {_inline_items(analysis.audit_paths)}",
        f"files: {analysis.files}",
        f"trajectory_rows: {analysis.trajectory_rows}",
        f"training_rows: {analysis.training_rows}",
        f"signal_records: {analysis.signal_records}",
        f"veto_negative_records: {analysis.veto_negative_records}",
        f"audit_decisions: {analysis.audit_decisions}",
        f"matched_by_audit_decisions: {analysis.matched_by_audit_decisions}",
        "matched_by_any_audit_records: "
        f"{analysis.matched_by_any_audit_records}",
        f"by_action: {_inline_counts(analysis.by_action)}",
        f"by_threat_state: {_inline_counts(analysis.by_threat_state)}",
        f"by_context: {_inline_counts(analysis.by_context)}",
        f"by_reason: {_inline_counts(analysis.by_reason)}",
        f"by_file: {_inline_counts(analysis.by_file)}",
        f"by_source: {_inline_counts(analysis.by_source)}",
        f"by_action_threat: {_inline_counts(analysis.by_action_threat)}",
        "by_action_context_threat: "
        f"{_inline_counts(analysis.by_action_context_threat)}",
        f"start_metric_averages: {_inline_floats(analysis.start_metric_averages)}",
        "start_metric_buckets:",
    ]
    lines.extend(_nested_counts_lines(analysis.start_metric_buckets))
    lines.append("negative_events_by_window:")
    lines.extend(_nested_counts_lines(analysis.negative_events_by_window))
    lines.append("payoff_events_by_window:")
    lines.extend(_nested_counts_lines(analysis.payoff_events_by_window))
    lines.extend(
        [
            "last_window_metric_averages: "
            f"{_inline_floats(analysis.last_window_metric_averages)}",
            "last_window_metric_buckets:",
        ]
    )
    lines.extend(_nested_counts_lines(analysis.last_window_metric_buckets))
    lines.extend(
        [
            "matched_by_audit_action: "
            f"{_inline_counts(analysis.matched_by_audit_action)}",
            "matched_by_audit_threat_state: "
            f"{_inline_counts(analysis.matched_by_audit_threat_state)}",
            "matched_by_audit_context: "
            f"{_inline_counts(analysis.matched_by_audit_context)}",
            "matched_by_audit_path: "
            f"{_inline_counts(analysis.matched_by_audit_path)}",
            "matched_by_audit_fallback_selected: "
            f"{_inline_counts(analysis.matched_by_audit_fallback_selected)}",
            "examples:",
        ]
    )
    if not analysis.examples:
        lines.append("  <none>")
    for example in analysis.examples:
        lines.append(
            f"  {Path(example.source_path).name}: step={example.step} "
            f"t={example.game_time:.1f} "
            f"action={example.candidate_action} "
            f"context={example.context} threat={example.threat_state} "
            f"reasons={_inline_items(example.reasons)} "
            f"last_window={example.last_window} "
            f"last_negatives={_inline_items(example.last_window_negative_events)} "
            f"army={_metric(example.start_metrics, 'army_count')} "
            f"workers={_metric(example.start_metrics, 'workers')} "
            f"gateways={_metric(example.start_metrics, 'ready_gateways')} "
            f"static={_metric(example.start_metrics, 'ready_static_defense')} "
            f"matched_audits={len(example.matched_audits)} "
            f"matched_predictions={_inline_items(example.matched_predicted_actions)} "
            f"fallback_selected={_bool_text(example.audit_fallback_selected)}"
        )
    return "\n".join(lines)


def _nested_counts_lines(counts_by_name: dict[str, dict[str, int]]) -> list[str]:
    if not counts_by_name:
        return ["  <none>"]
    return [
        f"  {name}: {_inline_counts(counts)}"
        for name, counts in counts_by_name.items()
    ]


def _inline_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _inline_floats(values: dict[str, float]) -> str:
    if not values:
        return "<none>"
    return ", ".join(f"{name}={value:.2f}" for name, value in values.items())


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


def _metric(metrics: dict[str, float], name: str) -> str:
    return f"{float(metrics.get(name, 0.0)):.0f}"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


if __name__ == "__main__":
    raise SystemExit(main())
