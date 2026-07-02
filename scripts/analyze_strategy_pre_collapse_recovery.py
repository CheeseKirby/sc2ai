"""Analyze recovery windows before strategy action-space collapse rows."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_pre_collapse_recovery_analysis import (  # noqa: E402
    DEFAULT_LOOKBACK_SECONDS,
    DEFAULT_MAX_MISSED_PRE_COLLAPSE_RECOVERY_RATE,
    DEFAULT_MAX_MISSED_PRE_COLLAPSE_RECOVERY_ROWS,
    StrategyPreCollapseRecoveryAnalysis,
    analyze_strategy_pre_collapse_recovery,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze recovery windows before strategy action-space collapse rows"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--lookback-seconds",
        type=float,
        default=DEFAULT_LOOKBACK_SECONDS,
        help="Seconds before each target row to inspect for recovery windows.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=20,
        help="Maximum target failure rows to include.",
    )
    parser.add_argument(
        "--max-windows-per-failure",
        type=int,
        default=6,
        help="Maximum representative recovery-window rows per failure.",
    )
    parser.add_argument(
        "--max-missed-pre-collapse-recovery-rows",
        type=int,
        default=DEFAULT_MAX_MISSED_PRE_COLLAPSE_RECOVERY_ROWS,
        help=(
            "Maximum missed pre-collapse recovery rows allowed before the "
            "gate recommendation becomes hold."
        ),
    )
    parser.add_argument(
        "--max-missed-pre-collapse-recovery-rate",
        type=float,
        default=DEFAULT_MAX_MISSED_PRE_COLLAPSE_RECOVERY_RATE,
        help=(
            "Maximum missed pre-collapse recovery row rate allowed before the "
            "gate recommendation becomes hold."
        ),
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
        help="Optional machine-readable analysis output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable analysis report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = analyze_strategy_pre_collapse_recovery(
        args.inputs,
        lookback_seconds=args.lookback_seconds,
        max_failures=args.max_failures,
        max_windows_per_failure=args.max_windows_per_failure,
        max_missed_pre_collapse_recovery_rows=(
            args.max_missed_pre_collapse_recovery_rows
        ),
        max_missed_pre_collapse_recovery_rate=(
            args.max_missed_pre_collapse_recovery_rate
        ),
    )
    report = format_strategy_pre_collapse_recovery_analysis(analysis)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(analysis))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    if args.fail_on_hold and analysis.recommendation == "hold":
        return 1
    return 0


def format_strategy_pre_collapse_recovery_analysis(
    analysis: StrategyPreCollapseRecoveryAnalysis,
) -> str:
    """Return a compact human-readable pre-collapse recovery report."""
    lines = [
        "Strategy pre-collapse recovery analysis",
        f"inputs: {', '.join(analysis.inputs)}",
        f"lookback_seconds: {analysis.lookback_seconds:.1f}",
        f"target_training_uses: {_inline_items(analysis.target_training_uses)}",
        f"recovery_actions: {_inline_items(analysis.recovery_action_names)}",
        f"recommendation: {analysis.recommendation}",
        f"blocking_reasons: {_inline_items(analysis.blocking_reasons)}",
        f"warnings: {_inline_items(analysis.warnings)}",
        "gate_thresholds: "
        "max_missed_pre_collapse_recovery_rows="
        f"{analysis.max_missed_pre_collapse_recovery_rows}, "
        "max_missed_pre_collapse_recovery_rate="
        f"{analysis.max_missed_pre_collapse_recovery_rate:.3f}",
        f"files: {analysis.files}",
        f"rows: {analysis.rows}",
        f"training_rows: {analysis.training_rows}",
        f"target_rows: {analysis.target_rows}",
        "missed_pre_collapse_recovery_rate: "
        f"{analysis.missed_pre_collapse_recovery_rate:.3f}",
        f"avoidability_counts: {_inline_counts(analysis.avoidability_counts)}",
        "target_training_use_counts: "
        f"{_inline_counts(analysis.target_training_use_counts)}",
        "target_label_quality_counts: "
        f"{_inline_counts(analysis.target_label_quality_counts)}",
        "target_threat_state_counts: "
        f"{_inline_counts(analysis.target_threat_state_counts)}",
        "target_executable_action_sets: "
        f"{_inline_counts(analysis.target_executable_action_sets)}",
        "rows_with_pre_collapse_recovery_window: "
        f"{analysis.rows_with_pre_collapse_recovery_window}/{analysis.target_rows}",
        "rows_with_pre_collapse_selected_recovery: "
        f"{analysis.rows_with_pre_collapse_selected_recovery}/{analysis.target_rows}",
        "rows_with_pre_collapse_selected_executable_recovery: "
        f"{analysis.rows_with_pre_collapse_selected_executable_recovery}/"
        f"{analysis.target_rows}",
        "missed_pre_collapse_recovery_rows: "
        f"{analysis.missed_pre_collapse_recovery_rows}/{analysis.target_rows}",
        "no_pre_collapse_recovery_window_rows: "
        f"{analysis.no_pre_collapse_recovery_window_rows}/{analysis.target_rows}",
        "pre_collapse_recovery_executable_counts_by_action: "
        f"{_inline_counts(analysis.pre_collapse_recovery_executable_counts_by_action)}",
        "pre_collapse_recovery_selected_counts_by_action: "
        f"{_inline_counts(analysis.pre_collapse_recovery_selected_counts_by_action)}",
        "pre_collapse_recovery_selected_executable_counts_by_action: "
        f"{_inline_counts(analysis.pre_collapse_recovery_selected_executable_counts_by_action)}",
        "recovery_blockers_at_target_by_action:",
    ]
    lines.extend(_nested_counts_lines(analysis.recovery_blockers_at_target_by_action))
    lines.append("files:")
    if not analysis.file_summaries:
        lines.append("  <none>")
    for summary in analysis.file_summaries:
        lines.append(
            f"  {Path(summary.path).name}: targets={summary.target_rows} "
            f"avoidability={_inline_counts(summary.avoidability_counts)} "
            f"uses={_inline_counts(summary.target_training_use_counts)} "
            f"window={summary.rows_with_pre_collapse_recovery_window} "
            "selected_exec="
            f"{summary.rows_with_pre_collapse_selected_executable_recovery} "
            f"missed={summary.missed_pre_collapse_recovery_rows}"
        )
    lines.append("failures:")
    if not analysis.failures:
        lines.append("  <none>")
    for failure in analysis.failures:
        lines.append(
            f"  {Path(failure.source_path).name}: step={failure.step} "
            f"t={failure.game_time:.1f} use={failure.recorded_training_use} "
            f"quality={failure.recorded_label_quality} "
            f"avoidability={failure.avoidability} threat={failure.threat_state} "
            f"recorded={failure.recorded_action} "
            f"exec={_inline_items(failure.executable_actions)} "
            f"recovery_at_target={_inline_items(failure.executable_recovery_actions)} "
            f"pre_rows={failure.pre_collapse_rows} "
            f"recovery_windows={failure.pre_collapse_recovery_window_rows} "
            "selected_exec_windows="
            f"{failure.pre_collapse_selected_executable_recovery_rows} "
            f"missed={str(failure.missed_pre_collapse_recovery).lower()} "
            "last_exec="
            f"{_time(failure.last_executable_recovery_time)}:"
            f"{_inline_items(failure.last_executable_recovery_actions)} "
            "last_selected_exec="
            f"{_time(failure.last_selected_executable_recovery_time)}:"
            f"{failure.last_selected_executable_recovery_action or '<none>'} "
            f"army={_metric(failure.start_metrics, 'army_count')} "
            f"minerals={_metric(failure.start_metrics, 'minerals')} "
            f"vespene={_metric(failure.start_metrics, 'vespene')} "
            f"supply={_metric(failure.start_metrics, 'supply_left')}"
        )
        for window in failure.recovery_windows:
            lines.append(
                f"    window step={window.step} t={window.game_time:.1f} "
                f"before={window.seconds_before_target:.1f}s "
                f"recorded={window.recorded_action} threat={window.threat_state} "
                f"exec={_inline_items(window.executable_recovery_actions)} "
                "selected_exec="
                f"{window.selected_executable_recovery_action or '<none>'} "
                f"minerals={_metric(window.start_metrics, 'minerals')} "
                f"vespene={_metric(window.start_metrics, 'vespene')} "
                f"supply={_metric(window.start_metrics, 'supply_left')}"
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


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


def _metric(metrics: dict[str, float | str], name: str) -> str:
    return f"{float(metrics.get(name, 0.0)):.0f}"


def _time(value: float | None) -> str:
    if value is None:
        return "<none>"
    return f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
