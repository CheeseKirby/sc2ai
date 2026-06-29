"""Diagnose Power-build strategy/tactic failure modes in strategy trajectories."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.power_tactic_diagnostics import (  # noqa: E402
    PowerTacticDiagnostics,
    diagnose_power_tactics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Power-specific strategy timing, tactics, and filters"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable Power diagnostics output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable Power diagnostics report path.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file Power summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = diagnose_power_tactics(args.inputs)
    report = format_power_tactic_diagnostics(
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


def format_power_tactic_diagnostics(
    diagnostics: PowerTacticDiagnostics,
    *,
    show_files: bool = False,
) -> str:
    """Return a compact human-readable Power tactic diagnostic report."""
    lines = [
        "Power tactic diagnostics",
        f"inputs: {', '.join(diagnostics.inputs)}",
        f"files: {diagnostics.files}",
        f"rows: {diagnostics.rows}",
        f"training_rows: {diagnostics.training_rows}",
        "results:",
    ]
    lines.extend(_format_counts(diagnostics.result_counts))
    lines.append("opponent_ai_builds:")
    lines.extend(_format_counts(diagnostics.opponent_ai_build_counts))
    lines.append("action_timing:")
    if diagnostics.action_timing_by_name:
        for name, stats in diagnostics.action_timing_by_name.items():
            lines.append(
                f"  {name}: count={stats.count} "
                f"first={_fmt(stats.first_game_time)} "
                f"avg={_fmt(stats.avg_game_time)} "
                f"min={_fmt(stats.min_game_time)} "
                f"max={_fmt(stats.max_game_time)}"
            )
    else:
        lines.append("  <none>")
    lines.append("threat_action_counts:")
    lines.extend(_format_counts(diagnostics.threat_action_counts_by_name))
    lines.append("tactic_counts:")
    lines.extend(_format_counts(diagnostics.tactic_counts))
    lines.append("filter_metadata:")
    lines.extend(
        [
            f"  rows_with_tactic_metadata: {diagnostics.rows_with_tactic_metadata}",
            f"  rows_with_filter_metadata: {diagnostics.rows_with_filter_metadata}",
            f"  filter_change_rows: {diagnostics.filter_change_rows}",
            "  training_rows_with_tactic_metadata: "
            f"{diagnostics.training_rows_with_tactic_metadata}",
            "  training_rows_with_filter_metadata: "
            f"{diagnostics.training_rows_with_filter_metadata}",
            f"  training_filter_change_rows: {diagnostics.training_filter_change_rows}",
        ]
    )
    lines.append("filter_changes:")
    lines.extend(_format_filter_changes(diagnostics.filter_changes))
    lines.append("robo_banking_filter_context:")
    lines.extend(
        _format_robo_banking_filter_contexts(
            diagnostics.robo_banking_filter_contexts,
        )
    )
    lines.append("static_defense_filter_context:")
    lines.extend(
        _format_static_defense_filter_contexts(
            diagnostics.static_defense_filter_contexts,
        )
    )
    lines.append("counterfactual_filter_delta:")
    lines.extend(_format_signed_counts(diagnostics.filter_action_delta_by_name))

    if show_files:
        lines.append("files:")
        for summary in diagnostics.file_summaries:
            lines.append(
                f"  {summary.path}: build={summary.opponent_ai_build} "
                f"difficulty={summary.difficulty or '<unknown>'} "
                f"opponent={summary.opponent_race or '<unknown>'} "
                f"result={summary.result or '<none>'} "
                f"rows={summary.rows} training_rows={summary.training_rows} "
                f"time={_fmt(summary.first_game_time)}-{_fmt(summary.last_game_time)}"
            )
            lines.append(
                f"    actions: {_inline_counts(summary.action_counts_by_name)}"
            )
            lines.append(
                "    tech_robo: "
                f"action={_fmt(summary.signals.first_tech_robo_action_time)} "
                f"pending={_fmt(summary.signals.first_pending_robo_time)} "
                f"ready={_fmt(summary.signals.first_ready_robo_time)} "
                f"observer={_fmt(summary.signals.first_observer_time)} "
                f"immortal={_fmt(summary.signals.first_immortal_time)}"
            )
            lines.append(
                "    forge_upgrade_static: "
                f"pending_forge={_fmt(summary.signals.first_pending_forge_time)} "
                f"ready_forge={_fmt(summary.signals.first_ready_forge_time)} "
                "upgrade_pending="
                f"{_fmt(summary.signals.first_ground_upgrade_pending_time)} "
                "upgrade_complete="
                f"{_fmt(summary.signals.first_ground_upgrade_complete_time)} "
                "pending_static="
                f"{_fmt(summary.signals.first_pending_static_defense_time)} "
                f"ready_static={_fmt(summary.signals.first_ready_static_defense_time)}"
            )
            lines.append(
                "    army: "
                f"produce_first={_fmt(summary.signals.first_produce_army_action_time)} "
                f"max={summary.signals.max_army_count:.1f} "
                f"avg={_fmt(summary.signals.avg_army_count)} "
                f"thresholds={_inline_float_counts(summary.signals.army_count_first_reached)}"
            )
            lines.append(
                "    threat: "
                f"first={_fmt(summary.threat.first_base_under_threat_time)} "
                f"rows={summary.threat.threat_rows} "
                f"actions={_inline_counts(summary.threat.threat_action_counts_by_name)} "
                f"produce={summary.threat.produce_army_under_threat_rows} "
                f"static={summary.threat.static_defense_under_threat_rows} "
                f"tech_robo={summary.threat.tech_robo_under_threat_rows}"
            )
            lines.append(
                "    economy: "
                f"minerals_avg={_fmt(summary.economy.avg_minerals)} "
                f"minerals_max={summary.economy.max_minerals:.1f} "
                f"vespene_avg={_fmt(summary.economy.avg_vespene)} "
                f"vespene_max={summary.economy.max_vespene:.1f} "
                f"worker_sat_min={_fmt(summary.economy.min_worker_saturation_ratio)} "
                f"worker_sat_avg={_fmt(summary.economy.avg_worker_saturation_ratio)} "
                "bank_rows="
                f"{summary.economy.mineral_bank_rows_ge_500}/"
                f"{summary.economy.vespene_bank_rows_ge_300}/"
                f"{summary.economy.dual_bank_rows_ge_500_300}"
            )
            lines.append(
                "    gateways: "
                f"ready_avg={_fmt(summary.gateways.avg_ready_gateways)} "
                f"ready_max={summary.gateways.max_ready_gateways:.1f} "
                f"pending_max={summary.gateways.max_pending_gateways:.1f} "
                f"idle_avg={_fmt(summary.gateways.avg_gateway_idle_count)} "
                f"idle_max={summary.gateways.max_gateway_idle_count:.1f} "
                f"idle_rows={summary.gateways.gateway_idle_rows} "
                "idle_bank_rows="
                f"{summary.gateways.idle_gateway_bank_rows_ge_150_minerals}"
            )
            lines.append(
                "    filter_changes: "
                f"{_inline_filter_changes(summary.filter_changes)}"
            )
            lines.append(
                "    robo_banking_filter_context: "
                f"{_inline_robo_banking_filter_contexts(summary.robo_banking_filter_contexts)}"
            )
            lines.append(
                "    static_defense_filter_context: "
                f"{_inline_static_defense_filter_contexts(summary.static_defense_filter_contexts)}"
            )
            lines.append(
                "    filter_delta: "
                f"{_inline_signed_counts(summary.filter_action_delta_by_name)}"
            )
            lines.append(
                f"    action_timeline: {_inline_action_timeline(summary.action_timeline)}"
            )
            lines.append(
                f"    tactic_timeline: {_inline_tactic_timeline(summary.tactic_timeline)}"
            )

    return "\n".join(lines)


def _format_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["  <none>"]
    return [f"  {name}: {count}" for name, count in counts.items()]


def _format_signed_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["  <none>"]
    return [f"  {name}: {count:+d}" for name, count in counts.items()]


def _format_filter_changes(changes: list) -> list[str]:
    if not changes:
        return ["  <none>"]
    return [
        f"  {change.opponent_ai_build}, {change.tactic_id}, "
        f"{change.before_action} -> {change.after_action}: {change.count}"
        for change in changes
    ]


def _format_robo_banking_filter_contexts(contexts: list) -> list[str]:
    if not contexts:
        return ["  <none>"]
    return [
        f"  {item.opponent_ai_build}, {item.tactic_id}, {item.context}: {item.count}"
        for item in contexts
    ]


def _format_static_defense_filter_contexts(contexts: list) -> list[str]:
    if not contexts:
        return ["  <none>"]
    return [
        f"  {item.opponent_ai_build}, {item.tactic_id}, "
        f"{item.after_action}, {item.context}: {item.count}"
        for item in contexts
    ]


def _inline_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _inline_signed_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={count:+d}" for name, count in counts.items())


def _inline_float_counts(counts: dict[str, float]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={value:.1f}" for name, value in counts.items())


def _inline_filter_changes(changes: list) -> str:
    if not changes:
        return "<none>"
    return ", ".join(
        (
            f"{change.opponent_ai_build}/{change.tactic_id}/"
            f"{change.before_action}->{change.after_action}={change.count}"
        )
        for change in changes
    )


def _inline_robo_banking_filter_contexts(contexts: list) -> str:
    if not contexts:
        return "<none>"
    return ", ".join(
        f"{item.tactic_id}/{item.context}={item.count}"
        for item in contexts
    )


def _inline_static_defense_filter_contexts(contexts: list) -> str:
    if not contexts:
        return "<none>"
    return ", ".join(
        (
            f"{item.tactic_id}/{item.context}->"
            f"{item.after_action}={item.count}"
        )
        for item in contexts
    )


def _inline_action_timeline(segments: list) -> str:
    if not segments:
        return "<none>"
    return " -> ".join(
        (
            f"{segment.action_name}({segment.count}@"
            f"{segment.start_game_time:.0f}-{segment.end_game_time:.0f})"
        )
        for segment in segments
    )


def _inline_tactic_timeline(segments: list) -> str:
    if not segments:
        return "<none>"
    return " -> ".join(
        (
            f"{segment.tactic_id}/{segment.tactic_phase}({segment.count}@"
            f"{segment.start_game_time:.0f}-{segment.end_game_time:.0f})"
        )
        for segment in segments
    )


def _fmt(value: float | None) -> str:
    return "none" if value is None else f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
