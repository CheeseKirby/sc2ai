"""Run the offline strategy data readiness pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.strategy_data_readiness_pipeline import (  # noqa: E402
    StrategyDataReadinessPipelineResult,
    run_strategy_data_readiness_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline strategy data readiness checks"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--prefix", default="strategy_data_readiness_latest")
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument(
        "--promotion-gate",
        type=Path,
        default=None,
        help="Optional promotion gate JSON to include in final readiness.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable pipeline summary path.",
    )
    parser.add_argument(
        "--fail-on-hold",
        action="store_true",
        help="Return exit code 1 when the pipeline recommendation is hold.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_strategy_data_readiness_pipeline(
        args.inputs,
        output_dir=args.output_dir,
        prefix=args.prefix,
        max_examples=args.max_examples,
        promotion_gate_path=args.promotion_gate,
    )
    report = format_strategy_data_readiness_pipeline(result)
    print(report)
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return pipeline_exit_code(result, fail_on_hold=args.fail_on_hold)


def pipeline_exit_code(
    result: StrategyDataReadinessPipelineResult,
    *,
    fail_on_hold: bool,
) -> int:
    """Return a CLI exit code for a completed readiness pipeline."""
    if fail_on_hold and not result.training_ready:
        return 1
    return 0


def format_strategy_data_readiness_pipeline(
    result: StrategyDataReadinessPipelineResult,
) -> str:
    """Return a compact human-readable pipeline summary."""
    lines = [
        "Strategy data readiness pipeline",
        f"recommendation: {result.recommendation}",
        f"training_ready: {_bool_text(result.training_ready)}",
        f"trajectory_detail_gate: {_bool_gate(result.trajectory_detail_ready)}",
        f"policy_explanation_gate: {_bool_gate(result.policy_explanation_ready)}",
        f"observation_detail_gate: {_bool_gate(result.observation_detail_ready)}",
        f"promotion_ready: {_optional_bool_text(result.promotion_ready)}",
        f"blocking_reasons: {_inline_items(result.blocking_reasons)}",
        f"inputs: {_inline_items(result.inputs)}",
        "artifacts:",
    ]
    for name, path in result.artifacts.items():
        lines.append(f"  {name}: {path}")
    return "\n".join(lines)


def _bool_gate(value: bool) -> str:
    return "ready" if value else "hold"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _optional_bool_text(value: bool | None) -> str:
    if value is None:
        return "unchecked"
    return _bool_text(value)


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


if __name__ == "__main__":
    raise SystemExit(main())
