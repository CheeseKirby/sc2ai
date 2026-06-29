"""Gate strategy training readiness from existing offline gate artifacts."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_training_readiness import (  # noqa: E402
    StrategyTrainingReadinessConfig,
    StrategyTrainingReadinessResult,
    evaluate_strategy_training_readiness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate strategy training readiness from offline artifacts"
    )
    parser.add_argument(
        "observation_detail_gate",
        type=Path,
        help="Strategy observation detail gate JSON file.",
    )
    parser.add_argument(
        "--expected-input",
        action="append",
        default=[],
        help="Expected trajectory input for the training run. May be repeated.",
    )
    parser.add_argument(
        "--promotion-gate",
        type=Path,
        default=None,
        help="Optional promotion gate JSON to include in the report.",
    )
    parser.add_argument(
        "--trajectory-detail-gate",
        type=Path,
        default=None,
        help="Optional raw trajectory detail gate JSON to include in the report.",
    )
    parser.add_argument(
        "--policy-explanation-gate",
        type=Path,
        default=None,
        help="Optional raw policy explanation gate JSON to include in the report.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable readiness output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable readiness report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_strategy_training_readiness(
        args.observation_detail_gate,
        config=StrategyTrainingReadinessConfig(
            expected_inputs=tuple(args.expected_input),
            trajectory_detail_gate_path=(
                str(args.trajectory_detail_gate)
                if args.trajectory_detail_gate is not None
                else None
            ),
            policy_explanation_gate_path=(
                str(args.policy_explanation_gate)
                if args.policy_explanation_gate is not None
                else None
            ),
            promotion_gate_path=(
                str(args.promotion_gate) if args.promotion_gate is not None else None
            ),
        ),
    )
    report = format_strategy_training_readiness(result)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(result))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_training_readiness(
    result: StrategyTrainingReadinessResult,
) -> str:
    """Return a compact human-readable training readiness report."""
    lines = [
        "Strategy training readiness",
        f"recommendation: {result.recommendation}",
        f"training_ready: {_bool_text(result.training_ready)}",
        f"promotion_ready: {_optional_bool_text(result.promotion_ready)}",
        f"blocking_reasons: {_inline_items(result.blocking_reasons)}",
        f"expected_inputs: {_inline_items(result.expected_inputs)}",
        "trajectory_detail_gate: "
        f"{_optional_gate_status(result.trajectory_detail_gate_path, result.trajectory_detail_gate_ready)}",
        f"trajectory_detail_gate_path: {result.trajectory_detail_gate_path or '<none>'}",
        "trajectory_detail_gate_inputs: "
        f"{_inline_items(result.trajectory_detail_gate_inputs)}",
        "trajectory_detail_gate_reasons: "
        f"{_inline_items(result.trajectory_detail_gate_blocking_reasons)}",
        "policy_explanation_gate: "
        f"{_optional_gate_status(result.policy_explanation_gate_path, result.policy_explanation_gate_ready)}",
        "policy_explanation_gate_path: "
        f"{result.policy_explanation_gate_path or '<none>'}",
        "policy_explanation_gate_inputs: "
        f"{_inline_items(result.policy_explanation_gate_inputs)}",
        "policy_explanation_gate_reasons: "
        f"{_inline_items(result.policy_explanation_gate_blocking_reasons)}",
        "observation_detail_gate: "
        f"{'ready' if result.observation_detail_gate_ready else 'hold'}",
        f"observation_detail_gate_path: {result.observation_detail_gate_path}",
        "observation_detail_gate_inputs: "
        f"{_inline_items(result.observation_detail_gate_inputs)}",
        "observation_detail_gate_reasons: "
        f"{_inline_items(result.observation_detail_gate_blocking_reasons)}",
        "promotion_gate: "
        f"{_promotion_gate_status(result)}",
        f"promotion_gate_path: {result.promotion_gate_path or '<none>'}",
        "promotion_gate_reasons: "
        f"{_inline_items(result.promotion_gate_blocking_reasons)}",
        f"selected_checkpoint: {result.selected_checkpoint_path or '<none>'}",
    ]
    return "\n".join(lines)


def _promotion_gate_status(result: StrategyTrainingReadinessResult) -> str:
    if result.promotion_gate_path is None:
        return "unchecked"
    return "promote" if result.promotion_gate_promotable else "hold"


def _optional_gate_status(path: str | None, ready: bool | None) -> str:
    if path is None:
        return "unchecked"
    return "ready" if ready else "hold"


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _optional_bool_text(value: bool | None) -> str:
    if value is None:
        return "unchecked"
    return _bool_text(value)


if __name__ == "__main__":
    raise SystemExit(main())
