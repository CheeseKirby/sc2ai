"""Analyze executable strategy action-space coverage from trajectories."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_action_space_analysis import (  # noqa: E402
    StrategyActionSpaceAnalysis,
    analyze_strategy_action_space,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze executable strategy action-space coverage"
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
        help="Maximum representative only-STAY_COURSE examples to include.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = analyze_strategy_action_space(
        args.inputs,
        max_examples=args.max_examples,
    )
    report = format_strategy_action_space_analysis(analysis)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(analysis))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_action_space_analysis(
    analysis: StrategyActionSpaceAnalysis,
) -> str:
    """Return a compact human-readable action-space coverage report."""
    lines = [
        "Strategy action-space analysis",
        f"inputs: {', '.join(analysis.inputs)}",
        f"files: {analysis.files}",
        f"rows: {analysis.rows}",
        f"training_rows: {analysis.training_rows}",
        f"signal_records: {analysis.signal_records}",
        f"actions: {_inline_items(analysis.action_names)}",
        f"executable_action_count: {_inline_counts(analysis.executable_action_count)}",
        f"executable_action_sets: {_inline_counts(analysis.executable_action_sets)}",
        "only_stay_course: "
        f"{analysis.only_stay_course_rows}/{analysis.rows} "
        f"ratio={analysis.only_stay_course_ratio:.3f}",
        "only_stay_course_under_threat: "
        f"{analysis.only_stay_course_under_threat_rows}/{analysis.rows} "
        f"ratio={analysis.only_stay_course_under_threat_ratio:.3f}",
        "only_stay_course_veto_negative: "
        f"{analysis.only_stay_course_veto_negative_rows}/"
        f"{analysis.only_stay_course_rows} "
        f"ratio={analysis.only_stay_course_veto_negative_ratio:.3f}",
        "only_stay_course_by_training_use: "
        f"{_inline_counts(analysis.only_stay_course_by_training_use)}",
        "only_stay_course_by_label_quality: "
        f"{_inline_counts(analysis.only_stay_course_by_label_quality)}",
        "only_stay_course_by_threat_state: "
        f"{_inline_counts(analysis.only_stay_course_by_threat_state)}",
        "only_stay_course_by_recorded_action: "
        f"{_inline_counts(analysis.only_stay_course_by_recorded_action)}",
        "only_stay_course_by_file: "
        f"{_inline_counts(analysis.only_stay_course_by_file)}",
        "only_stay_course_blockers_by_action:",
    ]
    lines.extend(_nested_counts_lines(analysis.only_stay_course_blockers_by_action))
    lines.append("threatened_only_stay_course_blockers_by_action:")
    lines.extend(
        _nested_counts_lines(analysis.threatened_only_stay_course_blockers_by_action)
    )
    lines.append("veto_only_stay_course_blockers_by_action:")
    lines.extend(_nested_counts_lines(analysis.veto_only_stay_course_blockers_by_action))
    lines.extend(
        [
            "threatened_only_stay_start_metric_averages: "
            f"{_inline_floats(analysis.threatened_only_stay_start_metric_averages)}",
            "veto_only_stay_start_metric_averages: "
            f"{_inline_floats(analysis.veto_only_stay_start_metric_averages)}",
            "examples:",
        ]
    )
    if not analysis.examples:
        lines.append("  <none>")
    for example in analysis.examples:
        lines.append(
            f"  {Path(example.source_path).name}: step={example.step} "
            f"t={example.game_time:.1f} "
            f"recorded={example.recorded_action} "
            f"use={example.recorded_training_use} "
            f"quality={example.recorded_label_quality} "
            f"context={example.context} threat={example.threat_state} "
            f"exec={_inline_items(example.executable_actions)} "
            f"army={_metric(example.start_metrics, 'army_count')} "
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


if __name__ == "__main__":
    raise SystemExit(main())
