"""Audit candidate strategy signal quality against a baseline."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_signal_audit import (  # noqa: E402
    StrategySignalAudit,
    audit_strategy_signals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit candidate strategy signal quality against a baseline"
    )
    parser.add_argument(
        "baseline",
        type=Path,
        help="Baseline strategy trajectory JSONL file or directory.",
    )
    parser.add_argument(
        "candidate",
        type=Path,
        help="Candidate strategy trajectory JSONL file or directory.",
    )
    parser.add_argument(
        "--include-before-filter-candidates",
        action="store_true",
        help="Include before-filter counterfactual rows in the signal audit.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable audit output path.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional human-readable audit report output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_strategy_signals(
        args.baseline,
        args.candidate,
        include_before_filter_candidates=args.include_before_filter_candidates,
    )
    report = format_strategy_signal_audit(audit)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(audit))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_signal_audit(audit: StrategySignalAudit) -> str:
    """Return a compact human-readable strategy signal audit report."""
    lines = [
        "Strategy signal audit",
        f"baseline_inputs: {', '.join(audit.baseline.inputs)}",
        f"candidate_inputs: {', '.join(audit.candidate.inputs)}",
        f"signal_healthy: {str(audit.signal_healthy).lower()}",
        "blocking_reasons:",
    ]
    lines.extend(_format_items(audit.blocking_reasons))
    lines.append("warnings:")
    lines.extend(_format_items(audit.warnings))
    lines.extend(
        [
            "signal_ratios:",
            "  accept_positive: "
            f"baseline={audit.baseline.accept_positive_ratio:.3f} "
            f"candidate={audit.candidate.accept_positive_ratio:.3f} "
            f"delta={audit.accept_positive_ratio_delta:.3f}",
            "  bad_signal: "
            f"baseline={audit.baseline.bad_signal_ratio:.3f} "
            f"candidate={audit.candidate.bad_signal_ratio:.3f} "
            f"delta={audit.bad_signal_ratio_delta:.3f}",
            "  drop_non_executable: "
            f"baseline={audit.baseline.drop_non_executable_ratio:.3f} "
            f"candidate={audit.candidate.drop_non_executable_ratio:.3f} "
            f"delta={audit.drop_non_executable_ratio_delta:.3f}",
            "  veto_negative: "
            f"baseline={audit.baseline.veto_negative_ratio:.3f} "
            f"candidate={audit.candidate.veto_negative_ratio:.3f} "
            f"delta={audit.veto_negative_ratio_delta:.3f}",
            "  weak_context: "
            f"baseline={audit.baseline.weak_context_ratio:.3f} "
            f"candidate={audit.candidate.weak_context_ratio:.3f} "
            f"delta={audit.weak_context_ratio_delta:.3f}",
            "records:",
            f"  baseline={audit.baseline.records} candidate={audit.candidate.records}",
            "training_use:",
            f"  baseline: {_inline_counts(audit.baseline.records_by_training_use)}",
            f"  candidate: {_inline_counts(audit.candidate.records_by_training_use)}",
            "label_quality:",
            f"  baseline: {_inline_counts(audit.baseline.records_by_label_quality)}",
            f"  candidate: {_inline_counts(audit.candidate.records_by_label_quality)}",
            "candidate_sources:",
            f"  baseline: {_inline_counts(audit.baseline.records_by_candidate_source)}",
            f"  candidate: {_inline_counts(audit.candidate.records_by_candidate_source)}",
            "candidate_actions:",
            f"  baseline: {_inline_counts(audit.baseline.records_by_candidate_action)}",
            f"  candidate: {_inline_counts(audit.candidate.records_by_candidate_action)}",
        ]
    )
    return "\n".join(lines)


def _format_items(items: list[str]) -> list[str]:
    if not items:
        return ["  <none>"]
    return [f"  {item}" for item in items]


def _inline_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


if __name__ == "__main__":
    raise SystemExit(main())
