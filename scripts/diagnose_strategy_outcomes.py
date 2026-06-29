"""Diagnose strategy action lookahead outcomes in strategy trajectories."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_outcome_diagnostics import (  # noqa: E402
    StrategyOutcomeDiagnostics,
    diagnose_strategy_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose strategy action lookahead outcomes"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable strategy outcome diagnostics output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable strategy outcome diagnostics report path.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file strategy outcome summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = diagnose_strategy_outcomes(args.inputs)
    report = format_strategy_outcome_diagnostics(
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


def format_strategy_outcome_diagnostics(
    diagnostics: StrategyOutcomeDiagnostics,
    *,
    show_files: bool = False,
) -> str:
    """Return a compact human-readable strategy outcome report."""
    lines = [
        "Strategy outcome diagnostics",
        f"inputs: {', '.join(diagnostics.inputs)}",
        f"lookahead_seconds: {', '.join(f'{value:g}' for value in diagnostics.lookahead_seconds)}",
        f"files: {diagnostics.files}",
        f"rows: {diagnostics.rows}",
        f"training_rows: {diagnostics.training_rows}",
        "results:",
    ]
    lines.extend(_format_counts(diagnostics.result_counts))
    lines.append(
        "execution: "
        f"effects={_inline_counts(diagnostics.execution_effect_counts)} "
        f"blockers={_inline_counts(diagnostics.execution_blocker_counts)}"
    )
    lines.append("sources:")
    if diagnostics.source_summaries:
        for summary in diagnostics.source_summaries:
            lines.append(
                f"  {summary.source}: files={summary.files} "
                f"rows={summary.rows} training_rows={summary.training_rows} "
                f"filter_change_rows={summary.filter_change_rows} "
                f"execution_effects={_inline_counts(summary.execution_effect_counts)} "
                f"execution_blockers={_inline_counts(summary.execution_blocker_counts)} "
                f"results={_inline_counts(summary.result_counts)} "
                f"actions={_inline_counts(summary.action_counts_by_name)}"
            )
            lines.append(
                "    first_actions: "
                f"{_inline_float_counts(summary.action_first_game_time_by_name)}"
            )
    else:
        lines.append("  <none>")

    lines.append("action_outcomes:")
    if diagnostics.action_summaries_by_name:
        for action_name, action_summary in diagnostics.action_summaries_by_name.items():
            lines.append(
                f"  {action_name}: count={action_summary.count} "
                f"first={_fmt(action_summary.first_game_time)} "
                f"avg={_fmt(action_summary.avg_game_time)} "
                f"early_before_240={action_summary.early_before_240_count}"
            )
            for window, outcome in diagnostics.action_window_summaries.get(
                action_name,
                {},
            ).items():
                lines.append(
                    f"    {window}: samples={outcome.samples} "
                    f"metrics={_select_metrics(outcome.avg_metrics)} "
                    f"events={_inline_counts(outcome.event_counts)} "
                    f"firsts={_inline_float_counts(outcome.avg_event_times)}"
                )
    else:
        lines.append("  <none>")

    lines.append("filter_change_outcomes:")
    if diagnostics.filter_change_summaries:
        for summary in diagnostics.filter_change_summaries:
            lines.append(
                f"  {summary.opponent_ai_build}, {summary.tactic_id}, "
                f"{summary.before_action} -> {summary.after_action}: "
                f"count={summary.count} "
                f"early_before_240={summary.early_before_240_count}"
            )
            for window, outcome in summary.outcomes_by_window.items():
                lines.append(
                    f"    {window}: samples={outcome.samples} "
                    f"metrics={_select_metrics(outcome.avg_metrics)} "
                    f"events={_inline_counts(outcome.event_counts)} "
                    f"firsts={_inline_float_counts(outcome.avg_event_times)}"
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
                f"time={_fmt(summary.first_game_time)}-{_fmt(summary.last_game_time)} "
                f"filter_change_rows={summary.filter_change_rows}"
            )
            lines.append(
                f"    actions: {_inline_counts(summary.action_counts_by_name)}"
            )
            lines.append(
                "    execution: "
                f"effects={_inline_counts(summary.execution_effect_counts)} "
                f"blockers={_inline_counts(summary.execution_blocker_counts)}"
            )
            lines.append(
                "    first_actions: "
                f"{_inline_float_counts(summary.action_first_game_time_by_name)}"
            )
            lines.append(
                "    robo_payoff: "
                f"ready_robo={_fmt(summary.ready_robo_first_game_time)} "
                f"observer={_fmt(summary.observer_first_game_time)} "
                f"immortal={_fmt(summary.immortal_first_game_time)} "
                f"observer_status={summary.robo_payoff.observer_status} "
                f"observer_delay="
                f"{_fmt(summary.robo_payoff.observer_after_ready_delay_seconds)} "
                f"immortal_status={summary.robo_payoff.immortal_status} "
                f"immortal_blocker={summary.robo_payoff.immortal_blocker} "
                f"robo_actions_after_ready="
                f"{summary.robo_payoff.robo_action_rows_after_ready} "
                f"robo_idle_rows_after_ready="
                f"{summary.robo_payoff.robo_idle_rows_after_ready} "
                f"immortal_candidate_rows="
                f"{summary.robo_payoff.immortal_candidate_rows_after_ready} "
                f"immortal_affordable_candidates="
                f"{summary.robo_payoff.immortal_affordable_candidate_rows_after_ready} "
                f"immortal_resource_supply_blocks="
                f"m{summary.robo_payoff.immortal_mineral_blocked_candidate_rows}/"
                f"v{summary.robo_payoff.immortal_vespene_blocked_candidate_rows}/"
                f"s{summary.robo_payoff.immortal_supply_blocked_candidate_rows}"
            )
            lines.append(f"    base_threat_rows: {summary.base_threat_rows}")

    return "\n".join(lines)


def _select_metrics(metrics: dict[str, float]) -> str:
    selected_names = (
        "ready_gateway_delta",
        "pending_gateway_after",
        "ready_robo_delta",
        "observer_delta",
        "immortal_delta",
        "observer_immortal_delta",
        "army_count_delta",
        "static_defense_delta",
        "base_under_threat_after",
        "worker_delta",
        "minerals_after",
        "vespene_after",
        "gateway_idle_after",
        "robo_idle_after",
    )
    selected = {
        name: metrics[name]
        for name in selected_names
        if name in metrics
    }
    return _inline_float_counts(selected)


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
