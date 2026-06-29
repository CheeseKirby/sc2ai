"""Gate strategy checkpoint promotion from offline signal audit artifacts."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_promotion_gate import (  # noqa: E402
    StrategyPromotionGateConfig,
    StrategyPromotionGateResult,
    evaluate_strategy_promotion_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate strategy checkpoint promotion from audit JSON files"
    )
    parser.add_argument("audits", nargs="+", type=Path, help="Audit JSON files")
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
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-predicted-non-executable-rows", type=int, default=0)
    parser.add_argument("--max-predicted-non-executable-ratio", type=float, default=0.0)
    parser.add_argument("--max-veto-negative-matches", type=int, default=0)
    parser.add_argument("--max-drop-non-executable-matches", type=int, default=0)
    parser.add_argument("--max-action-space-exhausted-matches", type=int, default=0)
    parser.add_argument("--max-action-critic-fallback-rows", type=int, default=0)
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Do not block promotion solely because audit warnings are present.",
    )
    parser.add_argument(
        "--observation-detail-gate",
        type=Path,
        default=None,
        help=(
            "Optional strategy observation detail gate JSON. "
            "Promotion holds unless it is ready and matches each audit input."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = StrategyPromotionGateConfig(
        min_rows=args.min_rows,
        max_predicted_non_executable_rows=args.max_predicted_non_executable_rows,
        max_predicted_non_executable_ratio=args.max_predicted_non_executable_ratio,
        max_veto_negative_matches=args.max_veto_negative_matches,
        max_drop_non_executable_matches=args.max_drop_non_executable_matches,
        max_action_space_exhausted_matches=(
            args.max_action_space_exhausted_matches
        ),
        max_action_critic_fallback_rows=args.max_action_critic_fallback_rows,
        fail_on_warnings=not args.allow_warnings,
        observation_detail_gate_path=(
            str(args.observation_detail_gate)
            if args.observation_detail_gate is not None
            else None
        ),
    )
    result = evaluate_strategy_promotion_gate(args.audits, config=config)
    report = format_strategy_promotion_gate(result)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(result))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_promotion_gate(result: StrategyPromotionGateResult) -> str:
    """Return a compact human-readable promotion gate report."""
    lines = [
        "Strategy promotion gate",
        f"recommendation: {result.recommendation}",
        f"promotable: {_bool_text(result.promotable)}",
        f"selected_audit: {result.selected_audit_path or '<none>'}",
        f"selected_checkpoint: {result.selected_checkpoint_path or '<none>'}",
        f"blocking_reasons: {_inline_items(result.blocking_reasons)}",
        "config: "
        f"min_rows={result.config.min_rows}, "
        "max_predicted_non_executable_rows="
        f"{result.config.max_predicted_non_executable_rows}, "
        "max_predicted_non_executable_ratio="
        f"{result.config.max_predicted_non_executable_ratio:.3f}, "
        f"max_veto_negative_matches={result.config.max_veto_negative_matches}, "
        "max_drop_non_executable_matches="
        f"{result.config.max_drop_non_executable_matches}, "
        "max_action_space_exhausted_matches="
        f"{result.config.max_action_space_exhausted_matches}, "
        "max_action_critic_fallback_rows="
        f"{result.config.max_action_critic_fallback_rows}, "
        f"fail_on_warnings={_bool_text(result.config.fail_on_warnings)}, "
        "observation_detail_gate="
        f"{result.config.observation_detail_gate_path or '<none>'}",
        "candidates:",
    ]
    if not result.candidates:
        lines.append("  <none>")
    for candidate in result.candidates:
        lines.append(
            f"  promotable={_bool_text(candidate.promotable)} "
            f"mode={candidate.prediction_mode} "
            f"audit={candidate.audit_path} "
            f"checkpoint={candidate.checkpoint_path} "
            f"rows={candidate.rows} "
            f"match={candidate.prediction_matches_recorded}/{candidate.rows} "
            f"match_ratio={candidate.prediction_match_ratio:.3f} "
            f"accept={candidate.accept_positive_prediction_matches}/"
            f"{candidate.accept_positive_rows} "
            f"veto={candidate.veto_negative_prediction_matches}/"
            f"{candidate.veto_negative_rows} "
            f"drop={candidate.drop_non_executable_prediction_matches}/"
            f"{candidate.drop_non_executable_rows} "
            f"space={candidate.action_space_exhausted_prediction_matches}/"
            f"{candidate.action_space_exhausted_rows} "
            f"nonexec={candidate.predicted_non_executable_rows} "
            f"fallback={candidate.action_critic_fallback_rows} "
            f"detail_gate={_detail_gate_status(candidate)} "
            "detail_reasons="
            f"{_inline_items(candidate.observation_detail_gate_blocking_reasons)} "
            f"reasons={_inline_items(candidate.blocking_reasons)}"
        )
    return "\n".join(lines)


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _detail_gate_status(candidate) -> str:
    if candidate.observation_detail_gate_path is None:
        return "unchecked"
    return "ready" if candidate.observation_detail_gate_ready else "hold"


if __name__ == "__main__":
    raise SystemExit(main())
