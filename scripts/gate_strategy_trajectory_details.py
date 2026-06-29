"""Gate raw strategy trajectory observation detail integrity."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_trajectory_detail_gate import (  # noqa: E402
    StrategyTrajectoryDetailGateConfig,
    StrategyTrajectoryDetailGateResult,
    evaluate_strategy_trajectory_detail_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate raw strategy trajectory observation details"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-observation-detail-ratio", type=float, default=1.0)
    parser.add_argument(
        "--min-observation-detail-complete-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-ready-static-defense-mismatch-rows",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-pending-static-defense-mismatch-rows",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable gate output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable gate report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_strategy_trajectory_detail_gate(
        args.inputs,
        config=StrategyTrajectoryDetailGateConfig(
            min_rows=args.min_rows,
            min_observation_detail_ratio=args.min_observation_detail_ratio,
            min_observation_detail_complete_ratio=(
                args.min_observation_detail_complete_ratio
            ),
            max_ready_static_defense_mismatch_rows=(
                args.max_ready_static_defense_mismatch_rows
            ),
            max_pending_static_defense_mismatch_rows=(
                args.max_pending_static_defense_mismatch_rows
            ),
        ),
    )
    report = format_strategy_trajectory_detail_gate(result)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(result))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_trajectory_detail_gate(
    result: StrategyTrajectoryDetailGateResult,
) -> str:
    """Return a compact human-readable trajectory detail gate report."""
    lines = [
        "Strategy trajectory detail gate",
        f"recommendation: {result.recommendation}",
        f"ready: {_bool_text(result.ready)}",
        f"inputs: {_inline_items(result.inputs)}",
        f"files: {result.files}",
        f"rows: {result.rows}",
        f"blocking_reasons: {_inline_items(result.blocking_reasons)}",
        "observation_details: "
        f"{result.observation_detail_rows}/{result.rows} "
        f"ratio={result.observation_detail_ratio:.3f}",
        "observation_detail_complete: "
        f"{result.observation_detail_complete_rows}/{result.rows} "
        f"ratio={result.observation_detail_complete_ratio:.3f}",
        "missing_detail_field_counts: "
        f"{_inline_counts(result.missing_detail_field_counts)}",
        "ready_static_defense_mismatch_rows: "
        f"{result.ready_static_defense_mismatch_rows}",
        "pending_static_defense_mismatch_rows: "
        f"{result.pending_static_defense_mismatch_rows}",
        "config: "
        f"min_rows={result.config.min_rows}, "
        "min_observation_detail_ratio="
        f"{result.config.min_observation_detail_ratio:.3f}, "
        "min_observation_detail_complete_ratio="
        f"{result.config.min_observation_detail_complete_ratio:.3f}, "
        "max_ready_static_defense_mismatch_rows="
        f"{result.config.max_ready_static_defense_mismatch_rows}, "
        "max_pending_static_defense_mismatch_rows="
        f"{result.config.max_pending_static_defense_mismatch_rows}",
    ]
    return "\n".join(lines)


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


def _inline_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


if __name__ == "__main__":
    raise SystemExit(main())
