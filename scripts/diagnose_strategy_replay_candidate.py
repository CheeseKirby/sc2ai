"""Diagnose replay-only strategy candidate action impact."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_replay_candidate import (  # noqa: E402
    DEFAULT_CANDIDATE_SOURCE,
    StrategyReplayCandidateDiagnostics,
    diagnose_strategy_replay_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose replay-only strategy candidate action impact"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--candidate-source",
        default=DEFAULT_CANDIDATE_SOURCE,
        choices=(DEFAULT_CANDIDATE_SOURCE,),
        help="Where to read the candidate action from.",
    )
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
        help="Print per-file replay candidate summaries.",
    )
    parser.add_argument(
        "--show-timeline",
        action="store_true",
        help="Print per-file changed-row timelines.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = diagnose_strategy_replay_candidate(
        args.inputs,
        candidate_source=args.candidate_source,
    )
    report = format_strategy_replay_candidate_diagnostics(
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


def format_strategy_replay_candidate_diagnostics(
    diagnostics: StrategyReplayCandidateDiagnostics,
    *,
    show_files: bool = False,
    show_timeline: bool = False,
) -> str:
    """Return a compact human-readable replay candidate report."""
    lines = [
        "Strategy replay candidate diagnostics",
        f"inputs: {', '.join(diagnostics.inputs)}",
        f"candidate_source: {diagnostics.candidate_source}",
        f"gate_decision: {diagnostics.gate_decision.recommendation}",
        "runtime_patch_candidate: "
        f"{_bool_text(diagnostics.gate_decision.runtime_patch_candidate)}",
        "gate_blocking_reasons: "
        f"{_inline_items(diagnostics.gate_decision.blocking_reasons)}",
        f"gate_warnings: {_inline_items(diagnostics.gate_decision.warnings)}",
        "gate_ratios: "
        f"candidate_executable_ratio="
        f"{diagnostics.gate_decision.executable_ratio:.2f}, "
        f"largest_group_count={diagnostics.gate_decision.largest_group_count}, "
        "largest_group_executable_ratio="
        f"{diagnostics.gate_decision.largest_group_executable_ratio:.2f}",
        "gate_thresholds: "
        f"max_changed_rows="
        f"{diagnostics.gate_decision.max_changed_rows_for_patch}, "
        f"max_largest_group_rows="
        f"{diagnostics.gate_decision.max_largest_group_rows_for_patch}, "
        "min_executable_ratio="
        f"{diagnostics.gate_decision.min_executable_ratio_for_patch:.2f}",
        "lookahead_seconds: "
        f"{', '.join(f'{value:g}' for value in diagnostics.lookahead_seconds)}",
        f"files: {diagnostics.files}",
        f"rows: {diagnostics.rows}",
        f"training_rows: {diagnostics.training_rows}",
        f"candidate_rows: {diagnostics.candidate_rows}",
        f"changed_rows: {diagnostics.changed_rows}",
        "candidate_executable="
        f"{diagnostics.immediate_candidate_executable_rows}/"
        f"{diagnostics.changed_rows}",
        f"action_delta: {_inline_counts(diagnostics.action_delta_by_name)}",
        "groups:",
    ]
    if diagnostics.group_summaries:
        for summary in diagnostics.group_summaries:
            lines.append(
                f"  {summary.source}, {summary.opponent_ai_build}, "
                f"tactic={summary.tactic_id}, "
                f"{summary.recorded_action} -> {summary.candidate_action}, "
                f"context={summary.context}, "
                f"threat_state={summary.threat_state}: count={summary.count} "
                f"candidate={summary.candidate_action} "
                "candidate_executable="
                f"{summary.immediate_candidate_executable_rows}/{summary.count} "
                f"delta={_inline_counts(summary.action_delta_by_name)} "
                f"start={_select_start_metrics(summary.avg_start_metrics)}"
            )
            for window, outcome in summary.outcomes_by_window.items():
                lines.append(
                    f"    {window}: samples={outcome.samples} "
                    f"metrics={_select_metrics(outcome.avg_metrics)} "
                    f"events={_select_events(outcome.event_counts)} "
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
                f"changed_rows={summary.changed_rows} "
                "candidate_executable="
                f"{summary.immediate_candidate_executable_rows}/"
                f"{summary.changed_rows}"
            )
            lines.append(
                f"    action_delta: {_inline_counts(summary.action_delta_by_name)}"
            )
            if show_timeline:
                lines.append("    timeline:")
                if summary.timeline_events:
                    for event in summary.timeline_events:
                        lines.append(
                            f"      step={event.step} t={event.game_time:.1f} "
                            f"recorded={event.recorded_action} "
                            f"candidate={event.candidate_action} "
                            "candidate_executable="
                            f"{event.immediate_candidate_executable} "
                            f"blocker={event.candidate_blocker or '<none>'} "
                            f"context={event.context} "
                            f"threat_state={event.threat_state} "
                            f"outcomes={_timeline_outcomes(event.outcomes_by_window)}"
                        )
                else:
                    lines.append("      <none>")

    return "\n".join(lines)


def _select_start_metrics(metrics: dict[str, float]) -> str:
    selected = (
        "minerals",
        "vespene",
        "supply_left",
        "own_bases",
        "ready_gateways",
        "pending_gateways",
        "ready_robo",
        "pending_robo",
        "ready_static_defense",
        "pending_static_defense",
        "base_under_threat",
    )
    return _inline_float_counts(
        {name: metrics[name] for name in selected if name in metrics}
    )


def _select_metrics(metrics: dict[str, float]) -> str:
    selected = (
        "ready_gateway_delta",
        "pending_gateway_after",
        "ready_robo_delta",
        "observer_delta",
        "immortal_delta",
        "army_count_delta",
        "static_defense_delta",
        "pending_static_defense_delta",
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


def _select_firsts(times: dict[str, float]) -> str:
    selected = (
        "first_threat_clear_time",
        "first_static_defense_delta_time",
        "first_pending_robo_after_action",
        "first_ready_robo_after_action",
        "first_observer_after_action",
        "first_immortal_after_action",
    )
    return _inline_float_counts(
        {name: times[name] for name in selected if name in times}
    )


def _timeline_outcomes(outcomes) -> str:
    parts = []
    for window, outcome in outcomes.items():
        parts.append(
            f"{window}:events={_select_events(outcome.event_counts)} "
            f"metrics={_select_metrics(outcome.avg_metrics)}"
        )
    return "; ".join(parts)


def _inline_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _inline_float_counts(counts: dict[str, float]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={value:.1f}" for name, value in counts.items())


if __name__ == "__main__":
    raise SystemExit(main())
