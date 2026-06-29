"""Analyze analysis-only emergency strategy action hypotheses."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_emergency_action_analysis import (  # noqa: E402
    StrategyEmergencyActionAnalysis,
    analyze_strategy_emergency_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze analysis-only emergency strategy actions"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
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
        help="Maximum representative examples to include.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = analyze_strategy_emergency_actions(
        args.inputs,
        max_examples=args.max_examples,
    )
    report = format_strategy_emergency_action_analysis(analysis)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(analysis))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_emergency_action_analysis(
    analysis: StrategyEmergencyActionAnalysis,
) -> str:
    """Return a compact human-readable emergency-action coverage report."""
    lines = [
        "Strategy emergency action analysis",
        f"inputs: {', '.join(analysis.inputs)}",
        f"files: {analysis.files}",
        f"rows: {analysis.rows}",
        f"training_rows: {analysis.training_rows}",
        f"signal_records: {analysis.signal_records}",
        f"emergency_actions: {_inline_items(analysis.emergency_action_names)}",
        "observation_details: "
        f"{analysis.observation_detail_rows}/{analysis.rows} "
        f"ratio={analysis.observation_detail_ratio:.3f}",
        "threatened_only_stay_course: "
        f"{analysis.threatened_only_stay_course_rows}/{analysis.rows} "
        f"ratio={analysis.threatened_only_stay_course_ratio:.3f}",
        "threatened_only_stay_course_details: "
        f"{analysis.threatened_only_stay_course_detail_rows}/"
        f"{analysis.threatened_only_stay_course_rows} "
        f"ratio={analysis.threatened_only_stay_course_detail_ratio:.3f}",
        "air_threat_only_stay_course_details: "
        f"{analysis.air_threat_only_stay_course_detail_rows}/"
        f"{analysis.air_threat_only_stay_course_rows} "
        f"ratio={analysis.air_threat_only_stay_course_detail_ratio:.3f}",
        "action_space_exhausted: "
        f"{analysis.action_space_exhausted_rows}/"
        f"{analysis.threatened_only_stay_course_rows} "
        f"ratio={analysis.action_space_exhausted_ratio:.3f}",
        "addressable_threatened_only_stay_course: "
        f"{analysis.addressable_threatened_only_stay_course_rows}/"
        f"{analysis.threatened_only_stay_course_rows} "
        f"ratio={analysis.addressable_threatened_only_stay_course_ratio:.3f}",
        "addressable_action_space_exhausted: "
        f"{analysis.addressable_action_space_exhausted_rows}/"
        f"{analysis.action_space_exhausted_rows} "
        f"ratio={analysis.addressable_action_space_exhausted_ratio:.3f}",
        f"emergency_action_count: {_inline_counts(analysis.emergency_action_count)}",
        f"emergency_action_sets: {_inline_counts(analysis.emergency_action_sets)}",
        "addressable_by_training_use: "
        f"{_inline_counts(analysis.addressable_by_training_use)}",
        "addressable_by_threat_state: "
        f"{_inline_counts(analysis.addressable_by_threat_state)}",
        "unaddressed_by_training_use: "
        f"{_inline_counts(analysis.unaddressed_by_training_use)}",
        "unaddressed_by_threat_state: "
        f"{_inline_counts(analysis.unaddressed_by_threat_state)}",
        "unaddressed_air_defense_gap_by_reason: "
        f"{_inline_counts(analysis.unaddressed_air_defense_gap_by_reason)}",
        "emergency_blockers_by_action:",
    ]
    lines.extend(_nested_counts_lines(analysis.emergency_blockers_by_action))
    lines.extend(
        [
            "addressable_start_metric_averages: "
            f"{_inline_floats(analysis.addressable_start_metric_averages)}",
            "unaddressed_start_metric_averages: "
            f"{_inline_floats(analysis.unaddressed_start_metric_averages)}",
            "examples:",
        ]
    )
    if not analysis.examples:
        lines.append("  <none>")
    for example in analysis.examples:
        lines.append(
            f"  {Path(example.source_path).name}: step={example.step} "
            f"t={example.game_time:.1f} "
            f"use={example.recorded_training_use} "
            f"quality={example.recorded_label_quality} "
            f"threat={example.threat_state} "
            f"standard={_inline_items(example.standard_executable_actions)} "
            f"emergency={_inline_items(example.emergency_actions)} "
            f"blockers={_inline_counts(example.emergency_blockers)} "
            f"air_gap={example.air_defense_gap_reason or '<none>'} "
            f"army={_metric(example.start_metrics, 'army_count')} "
            f"stalkers={_metric(example.start_metrics, 'stalkers')} "
            f"workers={_metric(example.start_metrics, 'workers')} "
            f"minerals={_metric(example.start_metrics, 'minerals')} "
            f"supply={_metric(example.start_metrics, 'supply_left')} "
            f"negatives={_inline_items(example.last_window_negative_events)} "
            f"reasons={_inline_items(example.reasons)}"
        )
    return "\n".join(lines)


def _nested_counts_lines(counts_by_name: dict[str, dict[str, int]]) -> list[str]:
    if not counts_by_name:
        return ["  <none>"]
    return [
        f"  {name}: {_inline_counts(counts)}"
        for name, counts in counts_by_name.items()
    ]


def _inline_counts(counts: dict[str, int] | dict[str, str]) -> str:
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


if __name__ == "__main__":
    raise SystemExit(main())
