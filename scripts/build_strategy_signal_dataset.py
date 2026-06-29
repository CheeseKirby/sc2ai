"""Build row-level strategy action signal datasets from trajectories."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_signal_dataset import (  # noqa: E402
    StrategySignalDataset,
    build_strategy_signal_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build row-level strategy action quality signals"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable signal dataset output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable signal report path.",
    )
    parser.add_argument(
        "--recorded-only",
        action="store_true",
        help="Do not add before-filter counterfactual candidate rows.",
    )
    parser.add_argument(
        "--show-records",
        action="store_true",
        help="Print compact per-record signal lines.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = build_strategy_signal_dataset(
        args.inputs,
        include_before_filter_candidates=not args.recorded_only,
    )
    report = format_strategy_signal_dataset(dataset, show_records=args.show_records)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(dataset))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_signal_dataset(
    dataset: StrategySignalDataset,
    *,
    show_records: bool = False,
) -> str:
    """Return a compact human-readable strategy signal report."""
    lines = [
        "Strategy signal dataset",
        f"inputs: {', '.join(dataset.inputs)}",
        "lookahead_seconds: "
        f"{', '.join(f'{value:g}' for value in dataset.lookahead_seconds)}",
        f"files: {dataset.files}",
        f"rows: {dataset.rows}",
        f"training_rows: {dataset.training_rows}",
        f"records: {len(dataset.records)}",
        f"training_use: {_inline_counts(dataset.records_by_training_use)}",
        f"label_quality: {_inline_counts(dataset.records_by_label_quality)}",
        f"candidate_sources: {_inline_counts(dataset.records_by_candidate_source)}",
        f"candidate_actions: {_inline_counts(dataset.records_by_candidate_action)}",
    ]
    if show_records:
        lines.append("records:")
        if not dataset.records:
            lines.append("  <none>")
        for record in dataset.records:
            lines.append(
                f"  {Path(record.path).name}: step={record.step} "
                f"t={record.game_time:.1f} source={record.candidate_source} "
                f"recorded={record.recorded_action} "
                f"candidate={record.candidate_action} "
                f"executable={_bool_text(record.immediate_executable)} "
                f"blocker={record.candidate_blocker or '<none>'} "
                f"quality={record.label_quality} "
                f"use={record.recommended_training_use} "
                f"context={record.context} threat={record.threat_state} "
                f"payoff={_payoff_summary(record.payoff_events_by_window)} "
                f"reasons={_inline_items(record.reasons)}"
            )
    return "\n".join(lines)


def _payoff_summary(events_by_window: dict[str, list[str]]) -> str:
    compact = {
        window: ",".join(events)
        for window, events in events_by_window.items()
        if events
    }
    if not compact:
        return "<none>"
    return ";".join(f"{window}:{events}" for window, events in compact.items())


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


if __name__ == "__main__":
    raise SystemExit(main())
