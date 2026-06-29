"""Diagnose active-threat suppression and replay-only candidate impact."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.active_threat_suppression_diagnostics import (  # noqa: E402
    ActiveThreatSuppressionDiagnostics,
    diagnose_active_threat_suppression,
)
from rl.experiments import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose RECOVERY/TECH_POWER suppression of BUILD_STATIC_DEFENSE "
            "and TECH_ROBO tactic-filter rows"
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable diagnostics output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable diagnostics report path.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file suppression summaries.",
    )
    parser.add_argument(
        "--show-timeline",
        action="store_true",
        help="Print per-file suppression event timelines.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = diagnose_active_threat_suppression(args.inputs)
    report = format_active_threat_suppression_diagnostics(
        diagnostics,
        show_files=args.show_files,
        show_timeline=args.show_timeline,
    )
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(diagnostics))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_active_threat_suppression_diagnostics(
    diagnostics: ActiveThreatSuppressionDiagnostics,
    *,
    show_files: bool = False,
    show_timeline: bool = False,
) -> str:
    """Return a compact suppression diagnostics report."""
    lines = [
        "Active-threat suppression diagnostics",
        f"inputs: {', '.join(diagnostics.inputs)}",
        "lookahead_seconds: "
        f"{', '.join(f'{value:g}' for value in diagnostics.lookahead_seconds)}",
        f"files: {diagnostics.files}",
        f"rows: {diagnostics.rows}",
        f"training_rows: {diagnostics.training_rows}",
        f"target_suppression_rows: {diagnostics.target_suppression_rows}",
        "results:",
    ]
    lines.extend(_format_counts(diagnostics.result_counts))
    lines.append("sources:")
    for summary in diagnostics.source_summaries:
        lines.append(
            f"  {summary.source}: files={summary.files} rows={summary.rows} "
            f"training_rows={summary.training_rows} "
            f"target_suppression_rows={summary.target_suppression_rows} "
            f"filter_change_rows={summary.filter_change_rows} "
            f"results={_inline_counts(summary.result_counts)}"
        )
        lines.append(f"    actions: {_inline_counts(summary.action_counts_by_name)}")
        lines.append(
            "    threat_actions: "
            f"{_inline_counts(summary.threat_action_counts_by_name)}"
        )

    lines.append("context_outcomes:")
    if diagnostics.context_summaries:
        for summary in diagnostics.context_summaries:
            lines.append(
                f"  {summary.source}, {summary.opponent_ai_build}, "
                f"{summary.tactic_id}, {summary.before_action} -> "
                f"{summary.after_action}, context={summary.context}, "
                f"threat_state={summary.threat_state}: count={summary.count} "
                f"candidate={summary.candidate_action} "
                "candidate_executable="
                f"{summary.immediate_candidate_executable_rows}/{summary.count} "
                f"start={_select_start_metrics(summary.avg_start_metrics)} "
                f"replay_delta={_inline_counts(summary.replay_action_delta_by_name)}"
            )
            for window, outcome in summary.outcomes_by_window.items():
                lines.append(
                    f"    {window}: samples={outcome.samples} "
                    f"metrics={_select_outcome_metrics(outcome.avg_metrics)} "
                    f"events={_select_events(outcome.event_counts)} "
                    f"rates={_select_rates(outcome.event_rates)} "
                    f"firsts={_select_firsts(outcome.avg_event_times)}"
                )
    else:
        lines.append("  <none>")

    impact = diagnostics.replay_candidate_impact
    lines.append("replay_candidate_impact:")
    lines.append(
        f"  name: {impact.name} affected_rows={impact.affected_rows} "
        "candidate_executable="
        f"{impact.immediate_candidate_executable_rows}/{impact.affected_rows}"
    )
    lines.append(f"  action_delta: {_inline_counts(impact.action_delta_by_name)}")
    lines.append("  context_impacts:")
    if impact.context_impacts:
        for item in impact.context_impacts:
            lines.append(
                f"    {item.source}, {item.tactic_id}, "
                f"{item.before_action}->{item.after_action}, "
                f"context={item.context}, threat_state={item.threat_state}, "
                f"count={item.count}, candidate={item.candidate_action}, "
                "candidate_executable="
                f"{item.immediate_candidate_executable_rows}/{item.count}, "
                f"delta={_inline_counts(item.action_delta_by_name)}"
            )
    else:
        lines.append("    <none>")

    if show_files:
        lines.append("files:")
        for summary in diagnostics.file_summaries:
            lines.append(
                f"  {summary.path}: source={summary.source} "
                f"build={summary.opponent_ai_build} "
                f"difficulty={summary.difficulty or '<unknown>'} "
                f"opponent={summary.opponent_race or '<unknown>'} "
                f"result={summary.result or '<none>'} rows={summary.rows} "
                f"training_rows={summary.training_rows} "
                f"target_suppression_rows={summary.target_suppression_rows} "
                "first_target_suppression_time="
                f"{_fmt(summary.first_target_suppression_time)}"
            )
            lines.append(
                f"    contexts: {_inline_counts(summary.context_counts)}"
            )
            lines.append(
                f"    filter_changes: {_inline_counts(summary.filter_change_counts)}"
            )
            lines.append(
                "    replay_delta: "
                f"{_inline_counts(summary.replay_action_delta_by_name)}"
            )
            if show_timeline:
                lines.append("    timeline:")
                if summary.timeline_events:
                    for event in summary.timeline_events:
                        lines.append(
                            f"      step={event.step} t={event.game_time:.1f} "
                            f"{event.tactic_id} {event.before_action}->"
                            f"{event.after_action} original_action={event.before_action} "
                            f"selected_action={event.after_action} "
                            f"candidate={event.candidate_action} "
                            f"context={event.context} "
                            f"threat_state={event.threat_state} "
                            "candidate_executable="
                            f"{event.immediate_candidate_executable} "
                            f"start={_select_start_metrics(event.start_metrics)} "
                            f"outcomes={_timeline_outcomes(event.outcomes_by_window)}"
                        )
                else:
                    lines.append("      <none>")

    return "\n".join(lines)


def _timeline_outcomes(outcomes) -> str:
    parts = []
    for window, outcome in outcomes.items():
        threat = "cleared" if outcome.events.get("threat_cleared") else None
        if outcome.events.get("threat_persisted"):
            threat = "persisted"
        metrics = outcome.metrics
        parts.append(
            f"{window}:{threat or 'no_threat_event'} "
            f"army_delta={metrics.get('army_count_delta', 0.0):.1f} "
            f"static_delta={metrics.get('static_defense_delta', 0.0):.1f} "
            f"ready_robo_delta={metrics.get('ready_robo_delta', 0.0):.1f}"
        )
    return "; ".join(parts)


def _select_start_metrics(metrics: dict[str, float]) -> str:
    selected = (
        "minerals",
        "vespene",
        "supply_left",
        "army_count",
        "ready_gateways",
        "pending_gateways",
        "pending_robo",
        "ready_robo",
        "pending_static_defense",
        "ready_static_defense",
        "base_under_threat",
        "base_under_air_threat",
        "base_under_ground_threat",
        "gateway_idle_count",
        "robo_idle_count",
    )
    return _inline_float_counts(
        {name: metrics[name] for name in selected if name in metrics}
    )


def _select_outcome_metrics(metrics: dict[str, float]) -> str:
    selected = (
        "army_count_delta",
        "static_defense_delta",
        "pending_static_defense_delta",
        "ready_robo_delta",
        "observer_delta",
        "immortal_delta",
        "base_under_threat_after",
        "minerals_after",
        "gateway_idle_after",
        "robo_idle_after",
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
        "pending_robo_seen",
        "ready_robo_seen",
        "observer_increased",
        "immortal_increased",
    )
    return _inline_counts({name: counts[name] for name in selected if name in counts})


def _select_rates(rates: dict[str, float]) -> str:
    selected = (
        "threat_cleared",
        "threat_persisted",
        "army_count_increased",
        "static_defense_increased",
        "pending_robo_seen",
        "ready_robo_seen",
        "observer_increased",
    )
    return _inline_float_counts(
        {name: rates[name] for name in selected if name in rates}
    )


def _select_firsts(times: dict[str, float]) -> str:
    selected = (
        "first_threat_clear_time",
        "first_static_defense_delta_time",
        "first_pending_robo_after_action",
        "first_ready_robo_after_action",
        "first_observer_after_action",
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
