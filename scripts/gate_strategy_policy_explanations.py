"""Gate raw strategy trajectory policy explanation metadata."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_policy_explanation_gate import (  # noqa: E402
    StrategyPolicyExplanationGateConfig,
    StrategyPolicyExplanationGateResult,
    evaluate_strategy_policy_explanation_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate raw strategy trajectory policy explanations"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-policy-source-ratio", type=float, default=1.0)
    parser.add_argument("--min-policy-reason-ratio", type=float, default=1.0)
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
    result = evaluate_strategy_policy_explanation_gate(
        args.inputs,
        config=StrategyPolicyExplanationGateConfig(
            min_rows=args.min_rows,
            min_policy_source_ratio=args.min_policy_source_ratio,
            min_policy_reason_ratio=args.min_policy_reason_ratio,
        ),
    )
    report = format_strategy_policy_explanation_gate(result)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(result))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_policy_explanation_gate(
    result: StrategyPolicyExplanationGateResult,
) -> str:
    """Return a compact human-readable policy explanation gate report."""
    lines = [
        "Strategy policy explanation gate",
        f"recommendation: {result.recommendation}",
        f"ready: {_bool_text(result.ready)}",
        f"inputs: {_inline_items(result.inputs)}",
        f"files: {result.files}",
        f"rows: {result.rows}",
        f"blocking_reasons: {_inline_items(result.blocking_reasons)}",
        "policy_sources: "
        f"{result.policy_source_rows}/{result.rows} "
        f"ratio={result.policy_source_ratio:.3f}",
        "policy_reasons: "
        f"{result.policy_reason_rows}/{result.rows} "
        f"ratio={result.policy_reason_ratio:.3f}",
        f"missing_policy_source_rows: {result.missing_policy_source_rows}",
        f"missing_policy_reason_rows: {result.missing_policy_reason_rows}",
        f"policy_source_counts: {_inline_counts(result.policy_source_counts)}",
        f"policy_reason_counts: {_inline_counts(result.policy_reason_counts)}",
        "config: "
        f"min_rows={result.config.min_rows}, "
        f"min_policy_source_ratio={result.config.min_policy_source_ratio:.3f}, "
        f"min_policy_reason_ratio={result.config.min_policy_reason_ratio:.3f}",
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
