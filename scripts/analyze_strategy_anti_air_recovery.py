"""Analyze anti-air recovery windows before air-threat gaps."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_anti_air_recovery_analysis import (  # noqa: E402
    StrategyAntiAirRecoveryAnalysis,
    analyze_strategy_anti_air_recovery,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze anti-air recovery windows before air-threat gaps"
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
        help="Maximum representative rows to include.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = analyze_strategy_anti_air_recovery(
        args.inputs,
        max_examples=args.max_examples,
    )
    report = format_strategy_anti_air_recovery_analysis(analysis)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(analysis))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_anti_air_recovery_analysis(
    analysis: StrategyAntiAirRecoveryAnalysis,
) -> str:
    """Return a compact human-readable anti-air recovery report."""
    lines = [
        "Strategy anti-air recovery analysis",
        f"inputs: {', '.join(analysis.inputs)}",
        f"files: {analysis.files}",
        f"rows: {analysis.rows}",
        f"training_rows: {analysis.training_rows}",
        f"results: {_inline_counts(analysis.result_counts)}",
        f"recovery_actions: {_inline_items(analysis.recovery_action_names)}",
        f"anti_air_asset_rows: {analysis.anti_air_asset_rows}/{analysis.training_rows}",
        f"air_threat_rows: {analysis.air_threat_rows}/{analysis.training_rows}",
        "air_threat_rows_without_anti_air: "
        f"{analysis.air_threat_rows_without_anti_air}/{analysis.air_threat_rows}",
        "air_threat_rows_with_anti_air: "
        f"{analysis.air_threat_rows_with_anti_air}/{analysis.air_threat_rows}",
        f"anti_air_gap_files: {analysis.anti_air_gap_files}/{analysis.files}",
        "files_with_pre_gap_recovery_window: "
        f"{analysis.files_with_pre_gap_recovery_window}/{analysis.anti_air_gap_files}",
        "files_with_pre_gap_recovery_selected: "
        f"{analysis.files_with_pre_gap_recovery_selected}/"
        f"{analysis.anti_air_gap_files}",
        "files_with_pre_gap_executable_recovery_selected: "
        f"{analysis.files_with_pre_gap_executable_recovery_selected}/"
        f"{analysis.anti_air_gap_files}",
        "missed_recovery_windows: "
        f"{analysis.missed_recovery_windows}/{analysis.anti_air_gap_files}",
        "recovery_executable_counts_by_action: "
        f"{_inline_counts(analysis.recovery_executable_counts_by_action)}",
        "recovery_selected_counts_by_action: "
        f"{_inline_counts(analysis.recovery_selected_counts_by_action)}",
        "recovery_selected_executable_counts_by_action: "
        f"{_inline_counts(analysis.recovery_selected_executable_counts_by_action)}",
        "missed_executable_recovery_counts_by_action: "
        f"{_inline_counts(analysis.missed_executable_recovery_counts_by_action)}",
        "blockers_by_action:",
    ]
    lines.extend(_nested_counts_lines(analysis.blockers_by_action))
    lines.append("files:")
    if not analysis.file_summaries:
        lines.append("  <none>")
    for summary in analysis.file_summaries:
        lines.append(
            f"  {Path(summary.path).name}: result={summary.result or '<none>'} "
            f"first_air={_time(summary.first_air_threat_time)} "
            f"first_gap={_time(summary.first_air_threat_without_anti_air_time)} "
            f"last_aa_before_gap={_time(summary.last_anti_air_before_gap_time)} "
            f"window_rows={summary.recovery_window_rows} "
            f"exec={_inline_counts(summary.recovery_executable_counts_by_action)} "
            f"selected_exec="
            f"{_inline_counts(summary.recovery_selected_executable_counts_by_action)} "
            f"missed_window={str(summary.missed_recovery_window).lower()}"
        )
    lines.append("examples:")
    if not analysis.examples:
        lines.append("  <none>")
    for example in analysis.examples:
        lines.append(
            f"  {Path(example.source_path).name}: role={example.row_role} "
            f"step={example.step} t={example.game_time:.1f} "
            f"before_gap={_time(example.seconds_before_gap)} "
            f"recorded={example.recorded_action} "
            f"threat={example.threat_state} "
            f"anti_air={str(example.anti_air_assets_present).lower()} "
            f"exec={_inline_items(example.executable_recovery_actions)} "
            f"missed={_inline_items(example.missed_executable_recovery_actions)} "
            f"blockers={_inline_counts(example.recovery_blockers)} "
            f"air_gap={example.air_defense_gap_reason or '<none>'} "
            f"army={_metric(example.start_metrics, 'army_count')} "
            f"stalkers={_metric(example.start_metrics, 'stalkers')} "
            f"cannons={_metric(example.start_metrics, 'ready_photon_cannons')} "
            f"minerals={_metric(example.start_metrics, 'minerals')} "
            f"vespene={_metric(example.start_metrics, 'vespene')} "
            f"supply={_metric(example.start_metrics, 'supply_left')}"
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


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


def _metric(metrics: dict[str, float], name: str) -> str:
    return f"{float(metrics.get(name, 0.0)):.0f}"


def _time(value: float | None) -> str:
    if value is None:
        return "<none>"
    return f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
