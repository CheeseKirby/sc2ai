"""Gate strategy observation detail readiness from emergency analysis artifacts."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_observation_detail_gate import (  # noqa: E402
    StrategyObservationDetailGateConfig,
    StrategyObservationDetailGateResult,
    evaluate_strategy_observation_detail_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate strategy observation detail coverage"
    )
    parser.add_argument("analysis", type=Path, help="Emergency analysis JSON file")
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
    parser.add_argument("--min-observation-detail-ratio", type=float, default=1.0)
    parser.add_argument(
        "--min-threatened-only-stay-course-detail-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-air-threat-only-stay-course-detail-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-static-defense-type-ambiguous-rows",
        type=int,
        default=0,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = StrategyObservationDetailGateConfig(
        min_rows=args.min_rows,
        min_observation_detail_ratio=args.min_observation_detail_ratio,
        min_threatened_only_stay_course_detail_ratio=(
            args.min_threatened_only_stay_course_detail_ratio
        ),
        min_air_threat_only_stay_course_detail_ratio=(
            args.min_air_threat_only_stay_course_detail_ratio
        ),
        max_static_defense_type_ambiguous_rows=(
            args.max_static_defense_type_ambiguous_rows
        ),
    )
    result = evaluate_strategy_observation_detail_gate(args.analysis, config=config)
    report = format_strategy_observation_detail_gate(result)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(result))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_observation_detail_gate(
    result: StrategyObservationDetailGateResult,
) -> str:
    """Return a compact human-readable observation detail gate report."""
    lines = [
        "Strategy observation detail gate",
        f"recommendation: {result.recommendation}",
        f"ready: {_bool_text(result.ready)}",
        f"analysis: {result.analysis_path}",
        f"inputs: {_inline_items(result.analysis_inputs)}",
        f"blocking_reasons: {_inline_items(result.blocking_reasons)}",
        "observation_details: "
        f"{result.observation_detail_rows}/{result.rows} "
        f"ratio={result.observation_detail_ratio:.3f}",
        "threatened_only_stay_course_details: "
        f"{result.threatened_only_stay_course_detail_rows}/"
        f"{result.threatened_only_stay_course_rows} "
        f"ratio={result.threatened_only_stay_course_detail_ratio:.3f}",
        "air_threat_only_stay_course_details: "
        f"{result.air_threat_only_stay_course_detail_rows}/"
        f"{result.air_threat_only_stay_course_rows} "
        f"ratio={result.air_threat_only_stay_course_detail_ratio:.3f}",
        "static_defense_type_ambiguous_rows: "
        f"{result.static_defense_type_ambiguous_rows}",
        "config: "
        f"min_rows={result.config.min_rows}, "
        "min_observation_detail_ratio="
        f"{result.config.min_observation_detail_ratio:.3f}, "
        "min_threatened_only_stay_course_detail_ratio="
        f"{result.config.min_threatened_only_stay_course_detail_ratio:.3f}, "
        "min_air_threat_only_stay_course_detail_ratio="
        f"{result.config.min_air_threat_only_stay_course_detail_ratio:.3f}, "
        "max_static_defense_type_ambiguous_rows="
        f"{result.config.max_static_defense_type_ambiguous_rows}",
    ]
    return "\n".join(lines)


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


if __name__ == "__main__":
    raise SystemExit(main())
