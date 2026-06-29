"""Diagnose active-threat outcomes for tactic-filtered static-defense rows."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.active_threat_outcome_diagnostics import (  # noqa: E402
    ActiveThreatOutcomeDiagnostics,
    diagnose_active_threat_outcomes,
)
from rl.experiments import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose active-threat outcomes for BUILD_STATIC_DEFENSE "
            "tactic-filter rewrites"
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable active-threat diagnostics output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable active-threat diagnostics report path.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file active-threat summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = diagnose_active_threat_outcomes(args.inputs)
    report = format_active_threat_outcome_diagnostics(
        diagnostics,
        show_files=args.show_files,
    )
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(diagnostics))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_active_threat_outcome_diagnostics(
    diagnostics: ActiveThreatOutcomeDiagnostics,
    *,
    show_files: bool = False,
) -> str:
    """Return a compact active-threat static-defense outcome report."""
    lines = [
        "Active-threat outcome diagnostics",
        f"inputs: {', '.join(diagnostics.inputs)}",
        "lookahead_seconds: "
        f"{', '.join(f'{value:g}' for value in diagnostics.lookahead_seconds)}",
        f"files: {diagnostics.files}",
        f"rows: {diagnostics.rows}",
        f"training_rows: {diagnostics.training_rows}",
        f"active_threat_filter_rows: {diagnostics.active_threat_filter_rows}",
        "results:",
    ]
    lines.extend(_format_counts(diagnostics.result_counts))
    lines.append("context_outcomes:")
    if diagnostics.context_summaries:
        for summary in diagnostics.context_summaries:
            lines.append(
                f"  {summary.opponent_ai_build}, {summary.tactic_id}, "
                f"{summary.before_action} -> {summary.after_action}, "
                f"{summary.context}: count={summary.count} "
                f"start={_select_start_metrics(summary.avg_start_metrics)}"
            )
            for window, outcome in summary.outcomes_by_window.items():
                lines.append(
                    f"    {window}: samples={outcome.samples} "
                    f"metrics={_select_outcome_metrics(outcome.avg_metrics)} "
                    f"events={_select_events(outcome.event_counts)} "
                    f"rates={_select_event_rates(outcome.event_rates)} "
                    f"firsts={_select_firsts(outcome.avg_event_times)}"
                )
    else:
        lines.append("  <none>")

    if show_files:
        lines.append("files:")
        for summary in diagnostics.file_summaries:
            lines.append(
                f"  {summary.path}: source={summary.source} "
                f"build={summary.opponent_ai_build} "
                f"difficulty={summary.difficulty or '<unknown>'} "
                f"opponent={summary.opponent_race or '<unknown>'} "
                f"result={summary.result or '<none>'} "
                f"rows={summary.rows} training_rows={summary.training_rows} "
                "active_threat_filter_rows="
                f"{summary.active_threat_filter_rows} "
                "first_active_threat_filter_time="
                f"{_fmt(summary.first_active_threat_filter_time)}"
            )
            lines.append(
                f"    contexts: {_inline_counts(summary.context_counts)}"
            )
            lines.append(
                f"    filter_changes: {_inline_counts(summary.filter_change_counts)}"
            )

    return "\n".join(lines)


def _select_start_metrics(metrics: dict[str, float]) -> str:
    selected = (
        "minerals",
        "vespene",
        "supply_left",
        "army_count",
        "pending_static_defense",
        "ready_static_defense",
        "gateway_idle_count",
    )
    return _inline_float_counts(
        {name: metrics[name] for name in selected if name in metrics}
    )


def _select_outcome_metrics(metrics: dict[str, float]) -> str:
    selected = (
        "army_count_delta",
        "static_defense_delta",
        "pending_static_defense_delta",
        "base_under_threat_after",
        "minerals_after",
        "gateway_idle_after",
    )
    return _inline_float_counts(
        {name: metrics[name] for name in selected if name in metrics}
    )


def _select_events(counts: dict[str, int]) -> str:
    selected = (
        "threat_cleared",
        "threat_persisted",
        "army_count_increased",
        "static_defense_increased",
        "base_under_threat_after",
    )
    return _inline_counts(
        {name: counts[name] for name in selected if name in counts}
    )


def _select_event_rates(rates: dict[str, float]) -> str:
    selected = (
        "threat_cleared",
        "threat_persisted",
        "army_count_increased",
        "static_defense_increased",
    )
    return _inline_float_counts(
        {name: rates[name] for name in selected if name in rates}
    )


def _select_firsts(times: dict[str, float]) -> str:
    selected = (
        "first_threat_clear_time",
        "first_static_defense_delta_time",
        "first_ready_gateway_delta_time",
    )
    return _inline_float_counts(
        {name: times[name] for name in selected if name in times}
    )


def _format_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["  <none>"]
    return [f"  {name}: {count}" for name, count in counts.items()]


def _inline_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _inline_float_counts(counts: dict[str, float]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={value:.1f}" for name, value in counts.items())


def _fmt(value: float | None) -> str:
    return "none" if value is None else f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
