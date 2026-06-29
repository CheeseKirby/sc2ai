"""Audit a candidate strategy trajectory set against a baseline."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_candidate_audit import (  # noqa: E402
    StrategyCandidateAudit,
    audit_strategy_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a candidate strategy trajectory set against a baseline"
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
    audit = audit_strategy_candidate(args.baseline, args.candidate)
    report = format_strategy_candidate_audit(audit)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(audit))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_candidate_audit(audit: StrategyCandidateAudit) -> str:
    """Return a compact human-readable strategy candidate audit report."""
    lines = [
        "Strategy candidate audit",
        f"baseline_inputs: {', '.join(audit.baseline.inputs)}",
        f"candidate_inputs: {', '.join(audit.candidate.inputs)}",
        f"promotable: {str(audit.promotable).lower()}",
        "blocking_reasons:",
    ]
    lines.extend(_format_items(audit.blocking_reasons))
    lines.append("warnings:")
    lines.extend(_format_items(audit.warnings))
    lines.extend(
        [
            "results:",
            f"  baseline: score={audit.baseline.result_score:.3f} "
            f"counts={_inline_counts(audit.baseline.result_counts)}",
            f"  candidate: score={audit.candidate.result_score:.3f} "
            f"counts={_inline_counts(audit.candidate.result_counts)}",
            f"  score_delta: {audit.result_score_delta:.3f}",
            "base_threat_rows:",
            f"  baseline={audit.baseline.base_threat_rows} "
            f"candidate={audit.candidate.base_threat_rows} "
            f"delta={audit.base_threat_rows_delta}",
            "execution_blockers:",
            f"  baseline={audit.baseline.execution_blocker_total} "
            f"candidate={audit.candidate.execution_blocker_total} "
            f"delta={audit.execution_blocker_delta}",
            f"  baseline_counts: {_inline_counts(audit.baseline.execution_blocker_counts)}",
            f"  candidate_counts: {_inline_counts(audit.candidate.execution_blocker_counts)}",
            "filter_change_rows:",
            f"  baseline={audit.baseline.filter_change_rows} "
            f"candidate={audit.candidate.filter_change_rows}",
            "action_delta:",
            f"  {_inline_counts(audit.action_count_delta_by_name)}",
            "action_counts:",
            f"  baseline: {_inline_counts(audit.baseline.action_counts_by_name)}",
            f"  candidate: {_inline_counts(audit.candidate.action_counts_by_name)}",
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
