"""Diagnose timing patterns in strategy trajectory JSONL data."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_timing_diagnostics import (  # noqa: E402
    StrategyTimingDiagnostics,
    diagnose_strategy_timing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose strategy action timing and signal latency"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable timing diagnostics output path.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file timing summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = diagnose_strategy_timing(args.inputs)
    print(format_strategy_timing_diagnostics(diagnostics, show_files=args.show_files))
    if args.json_output is not None:
        write_json(args.json_output, asdict(diagnostics))
    return 0


def format_strategy_timing_diagnostics(
    diagnostics: StrategyTimingDiagnostics,
    *,
    show_files: bool = False,
) -> str:
    """Return a compact human-readable strategy timing report."""
    lines = [
        "Strategy timing diagnostics",
        f"inputs: {', '.join(diagnostics.inputs)}",
        f"files: {diagnostics.files}",
        f"rows: {diagnostics.rows}",
        f"training_rows: {diagnostics.training_rows}",
        "results:",
    ]
    if diagnostics.result_counts:
        for result, count in diagnostics.result_counts.items():
            lines.append(f"  {result}: {count}")
    else:
        lines.append("  <none>")

    lines.append("action_timing:")
    if diagnostics.action_timing_by_name:
        for name, stats in diagnostics.action_timing_by_name.items():
            lines.append(
                f"  {name}: count={stats.count} "
                f"first={_format_optional_float(stats.first_game_time)} "
                f"avg={_format_optional_float(stats.avg_game_time)} "
                f"min={_format_optional_float(stats.min_game_time)} "
                f"max={_format_optional_float(stats.max_game_time)}"
            )
    else:
        lines.append("  <none>")

    lines.append("threat_action_counts:")
    if diagnostics.threat_action_counts_by_name:
        for name, count in diagnostics.threat_action_counts_by_name.items():
            lines.append(f"  {name}: {count}")
    else:
        lines.append("  <none>")

    lines.append("pending_repeat_counts:")
    if diagnostics.pending_repeat_counts_by_name:
        for name, count in diagnostics.pending_repeat_counts_by_name.items():
            lines.append(f"  {name}: {count}")
    else:
        lines.append("  <none>")

    lines.append("tech_robo_latency:")
    for signal_name, summary in diagnostics.tech_robo_latency.items():
        lines.append(
            f"  {signal_name}: files_with_signal={summary.files_with_signal} "
            f"files_with_tech_after_signal={summary.files_with_tech_after_signal} "
            f"tech_before_signal={summary.files_with_tech_before_signal} "
            f"no_tech={summary.files_without_tech} "
            f"without_tech_after_signal={summary.files_without_tech_after_signal} "
            f"avg_delay={summary.avg_delay:.1f} "
            f"min_delay={summary.min_delay:.1f} "
            f"max_delay={summary.max_delay:.1f} "
            f"avg_early_lead={summary.avg_early_lead:.1f}"
        )

    lines.append("hard_defeat_files:")
    if diagnostics.hard_defeat_file_paths:
        for path in diagnostics.hard_defeat_file_paths:
            lines.append(f"  {path}")
    else:
        lines.append("  <none>")

    if show_files:
        lines.append("files:")
        for summary in diagnostics.file_summaries:
            actions = ", ".join(
                f"{name}={count}"
                for name, count in summary.action_counts_by_name.items()
            )
            threat_actions = ", ".join(
                f"{name}={count}"
                for name, count in summary.threat_action_counts_by_name.items()
            )
            pending_repeats = ", ".join(
                f"{name}={count}"
                for name, count in summary.pending_repeat_counts_by_name.items()
            )
            lines.append(
                f"  {summary.path}: difficulty={summary.difficulty or '<unknown>'} "
                f"opponent={summary.opponent_race or '<unknown>'} "
                f"result={summary.result or '<none>'} "
                f"rows={summary.rows} training_rows={summary.training_rows} "
                f"time={_format_optional_float(summary.first_game_time)}-"
                f"{_format_optional_float(summary.last_game_time)} "
                f"actions={actions or '<none>'} "
                f"threat_actions={threat_actions or '<none>'} "
                f"pending_repeats={pending_repeats or '<none>'}"
            )
            timeline = " -> ".join(
                (
                    f"{segment.action_name}"
                    f"({segment.count}@{segment.start_game_time:.0f}-"
                    f"{segment.end_game_time:.0f})"
                )
                for segment in summary.timeline
            )
            lines.append(f"    timeline: {timeline or '<none>'}")
            if summary.signal_first_game_time:
                signals = ", ".join(
                    f"{name}={value:.1f}"
                    for name, value in summary.signal_first_game_time.items()
                )
                lines.append(
                    f"    signals: {signals}; "
                    f"tech_robo_first={_format_optional_float(summary.tech_robo_first_game_time)}"
                )

    return "\n".join(lines)


def _format_optional_float(value: float | None) -> str:
    return "none" if value is None else f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
