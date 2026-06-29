"""Diagnose trajectory JSONL data before training or policy comparison."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.diagnostics import diagnose_trajectories  # noqa: E402
from rl.experiments import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose trajectory datasets")
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable diagnostics output path.",
    )
    parser.add_argument(
        "--allow-missing-terminal",
        action="store_true",
        help="Do not warn when a trajectory file has no done=true terminal row.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file summaries after the dataset summary.",
    )
    parser.add_argument(
        "--kind",
        choices=["army", "strategy"],
        default="army",
        help=(
            "Trajectory kind to diagnose. army reads action/observation; "
            "strategy reads strategy_action/strategy_observation."
        ),
    )
    parser.add_argument(
        "--min-action-count",
        type=int,
        default=10,
        help="Warn when a present action has fewer than this many training rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = diagnose_trajectories(
        args.inputs,
        require_terminal=not args.allow_missing_terminal,
        min_action_count=args.min_action_count,
        trajectory_kind=args.kind,
    )
    print(format_diagnostics(diagnostics, show_files=args.show_files))

    if args.json_output is not None:
        write_json(args.json_output, asdict(diagnostics))

    return 0


def format_diagnostics(diagnostics, *, show_files: bool = False) -> str:
    """Return a compact human-readable diagnostics report."""
    lines = [
        "Trajectory diagnostics",
        f"inputs: {', '.join(diagnostics.inputs)}",
        f"files: {diagnostics.files}",
        f"rows: {diagnostics.rows}",
        f"training_rows: {diagnostics.training_rows}",
        f"terminal_rows: {diagnostics.terminal_rows}",
        f"empty_files: {diagnostics.empty_files}",
        f"files_missing_terminal: {diagnostics.files_missing_terminal}",
        f"observation_dim: {diagnostics.observation_dim}",
        "observation_schemas:",
    ]
    if diagnostics.observation_schema_counts:
        for version, count in diagnostics.observation_schema_counts.items():
            lines.append(f"  {_format_schema_name(version)}: {count}")
    else:
        lines.append("  <none>")
    lines.extend(
        [
            "rows_defaulted_observation_fields: "
            f"{diagnostics.rows_defaulted_observation_fields}",
            "observation_feature_stats:",
        ]
    )
    if diagnostics.observation_feature_stats:
        for field, stats in diagnostics.observation_feature_stats.items():
            lines.append(
                f"  {field}: min={stats['min']:.3f} "
                f"max={stats['max']:.3f} avg={stats['avg']:.3f}"
            )
    else:
        lines.append("  <none>")
    lines.extend(
        [
            f"action_coverage: {diagnostics.action_coverage:.1%}",
            "actions:",
        ]
    )
    if diagnostics.action_counts_by_name:
        for name, count in diagnostics.action_counts_by_name.items():
            lines.append(f"  {name}: {count}")
    else:
        lines.append("  <none>")

    lines.append("missing_actions:")
    if diagnostics.missing_action_names:
        for name in diagnostics.missing_action_names:
            lines.append(f"  {name}")
    else:
        lines.append("  <none>")

    lines.append(f"low_count_actions (<{diagnostics.min_action_count}):")
    if diagnostics.low_count_action_names:
        for name in diagnostics.low_count_action_names:
            lines.append(f"  {name}")
    else:
        lines.append("  <none>")

    lines.append("results:")
    if diagnostics.result_counts:
        for result, count in diagnostics.result_counts.items():
            lines.append(f"  {result}: {count}")
    else:
        lines.append("  <none>")

    lines.extend(
        [
            "rows_per_file:",
            f"  min: {diagnostics.rows_per_file['min']:.0f}",
            f"  max: {diagnostics.rows_per_file['max']:.0f}",
            f"  avg: {diagnostics.rows_per_file['avg']:.1f}",
            "training_rows_per_file:",
            f"  min: {diagnostics.training_rows_per_file['min']:.0f}",
            f"  max: {diagnostics.training_rows_per_file['max']:.0f}",
            f"  avg: {diagnostics.training_rows_per_file['avg']:.1f}",
            "warnings:",
        ]
    )
    if diagnostics.warnings:
        for warning in diagnostics.warnings:
            lines.append(f"  {warning}")
    else:
        lines.append("  <none>")

    if show_files:
        lines.append("files:")
        for summary in diagnostics.file_summaries:
            result = ", ".join(
                f"{name}={count}" for name, count in summary.result_counts.items()
            )
            actions = ", ".join(
                f"{name}={count}"
                for name, count in summary.action_counts_by_name.items()
            )
            lines.append(
                "  "
                f"{summary.path}: rows={summary.rows} "
                f"terminal={summary.terminal_rows} "
                f"steps={summary.first_step}-{summary.last_step} "
                f"actions={actions or '<none>'} "
                f"results={result or '<none>'}"
            )

    return "\n".join(lines)


def _format_schema_name(version: str) -> str:
    return f"v{version}" if str(version).isdigit() else str(version)


if __name__ == "__main__":
    raise SystemExit(main())
