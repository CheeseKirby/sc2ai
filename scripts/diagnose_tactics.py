"""Diagnose tactic metadata in strategy trajectory JSONL data."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.tactic_diagnostics import TacticDiagnostics, diagnose_tactics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose tactic metadata and strategy action filter changes"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable tactic diagnostics output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable tactic diagnostics report path.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file tactic summaries.",
    )
    parser.add_argument(
        "--show-filter-timeline",
        action="store_true",
        help="Print per-row tactic filter before/after metadata for each file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = diagnose_tactics(args.inputs)
    report = format_tactic_diagnostics(
        diagnostics,
        show_files=args.show_files,
        show_filter_timeline=args.show_filter_timeline,
    )
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(diagnostics))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_tactic_diagnostics(
    diagnostics: TacticDiagnostics,
    *,
    show_files: bool = False,
    show_filter_timeline: bool = False,
) -> str:
    """Return a compact human-readable tactic metadata report."""
    lines = [
        "Tactic diagnostics",
        f"inputs: {', '.join(diagnostics.inputs)}",
        f"files: {diagnostics.files}",
        f"rows: {diagnostics.rows}",
        f"training_rows: {diagnostics.training_rows}",
        f"rows_with_tactic_metadata: {diagnostics.rows_with_tactic_metadata}",
        f"rows_with_filter_metadata: {diagnostics.rows_with_filter_metadata}",
        f"filter_change_rows: {diagnostics.filter_change_rows}",
        "training_rows_with_tactic_metadata: "
        f"{diagnostics.training_rows_with_tactic_metadata}",
        "training_rows_with_filter_metadata: "
        f"{diagnostics.training_rows_with_filter_metadata}",
        f"training_filter_change_rows: {diagnostics.training_filter_change_rows}",
        "opponent_ai_builds:",
    ]
    lines.extend(_format_counts(diagnostics.opponent_ai_build_counts))
    lines.append("tactic_counts:")
    lines.extend(_format_counts(diagnostics.tactic_counts))
    lines.append("tactic_phase_counts:")
    lines.extend(_format_counts(diagnostics.tactic_phase_counts))
    lines.append("tactic_source_counts:")
    lines.extend(_format_counts(diagnostics.tactic_source_counts))
    lines.append("filter_changes:")
    if diagnostics.filter_changes:
        for change in diagnostics.filter_changes:
            lines.append(
                f"  {change.opponent_ai_build}, {change.tactic_id}, "
                f"{change.before_action} -> {change.after_action}: {change.count}"
            )
    else:
        lines.append("  <none>")

    lines.append("results:")
    lines.extend(_format_counts(diagnostics.result_counts))

    if show_files:
        lines.append("files:")
        for summary in diagnostics.file_summaries:
            tactics = _inline_counts(summary.tactic_counts)
            changes = ", ".join(
                (
                    f"{change.opponent_ai_build}/{change.tactic_id}/"
                    f"{change.before_action}->{change.after_action}={change.count}"
                )
                for change in summary.filter_changes
            )
            lines.append(
                f"  {summary.path}: build={summary.opponent_ai_build} "
                f"difficulty={summary.difficulty or '<unknown>'} "
                f"opponent={summary.opponent_race or '<unknown>'} "
                f"result={summary.result or '<none>'} "
                f"rows={summary.rows} training_rows={summary.training_rows} "
                f"metadata={summary.rows_with_tactic_metadata}/"
                f"{summary.rows} "
                f"training_metadata={summary.training_rows_with_tactic_metadata}/"
                f"{summary.training_rows} "
                f"tactics={tactics or '<none>'} "
                f"filter_changes={changes or '<none>'}"
            )
            timeline = " -> ".join(
                (
                    f"{segment.tactic_id}/{segment.tactic_phase}"
                    f"({segment.count}@{segment.start_game_time:.0f}-"
                    f"{segment.end_game_time:.0f})"
                )
                for segment in summary.timeline
            )
            lines.append(f"    timeline: {timeline or '<none>'}")
            if show_filter_timeline:
                lines.append("    filter_timeline:")
                if summary.filter_timeline:
                    for event in summary.filter_timeline:
                        changed = "*" if event.changed else "="
                        lines.append(
                            f"      line={event.line_number} "
                            f"t={event.game_time:.1f} "
                            f"{event.tactic_id}/{event.tactic_phase} "
                            f"{event.original_action} -> {event.selected_action} "
                            f"changed={changed} "
                            f"minerals={event.minerals:.1f} "
                            f"vespene={event.vespene:.1f} "
                            f"supply_left={event.supply_left:.1f} "
                            f"pending_gateways={event.pending_gateways:.1f} "
                            f"ready_gateways={event.ready_gateways:.1f} "
                            f"pending_robo={event.pending_robo:.1f} "
                            f"ready_robo={event.ready_robo:.1f} "
                            f"pending_static={event.pending_static_defense:.1f} "
                            f"ready_static={event.ready_static_defense:.1f} "
                            f"base_under_threat={event.base_under_threat:.1f} "
                            f"gateway_idle={event.gateway_idle_count:.1f} "
                            f"robo_idle={event.robo_idle_count:.1f}"
                        )
                else:
                    lines.append("      <none>")

    return "\n".join(lines)


def _format_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["  <none>"]
    return [f"  {name}: {count}" for name, count in counts.items()]


def _inline_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in counts.items())


if __name__ == "__main__":
    raise SystemExit(main())
