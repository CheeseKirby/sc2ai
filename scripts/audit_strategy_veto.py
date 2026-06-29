"""Audit a conservative strategy veto baseline over trajectory signals."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_veto_audit import (  # noqa: E402
    StrategyVetoAudit,
    audit_strategy_veto_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a conservative strategy veto baseline"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--include-before-filter-candidates",
        action="store_true",
        help="Include before-filter counterfactual rows in the veto audit.",
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
    parser.add_argument(
        "--show-decisions",
        action="store_true",
        help="Print compact per-record veto decisions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_strategy_veto_baseline(
        args.inputs,
        include_before_filter_candidates=args.include_before_filter_candidates,
    )
    report = format_strategy_veto_audit(audit, show_decisions=args.show_decisions)
    print(report)
    if args.json_output is not None:
        write_json(args.json_output, asdict(audit))
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report + "\n", encoding="utf-8")
    return 0


def format_strategy_veto_audit(
    audit: StrategyVetoAudit,
    *,
    show_decisions: bool = False,
) -> str:
    """Return a compact human-readable veto audit report."""
    lines = [
        "Strategy veto audit",
        f"inputs: {', '.join(audit.inputs)}",
        f"files: {audit.files}",
        f"records: {audit.records}",
        f"hard_veto_records: {audit.hard_veto_records}",
        f"review_records: {audit.review_records}",
        "bad_capture: "
        f"{audit.bad_records_hard_vetoed}/{audit.bad_records} "
        f"ratio={audit.bad_capture_ratio:.3f}",
        "accept_positive_false_veto: "
        f"{audit.accept_positive_records_hard_vetoed}/"
        f"{audit.accept_positive_records} "
        f"ratio={audit.accept_positive_false_veto_ratio:.3f}",
        f"hard_veto_by_reason: {_inline_counts(audit.hard_veto_by_reason)}",
        f"review_by_reason: {_inline_counts(audit.review_by_reason)}",
        f"hard_veto_by_training_use: {_inline_counts(audit.hard_veto_by_training_use)}",
        f"review_by_training_use: {_inline_counts(audit.review_by_training_use)}",
        f"hard_veto_by_action: {_inline_counts(audit.hard_veto_by_action)}",
        f"review_by_action: {_inline_counts(audit.review_by_action)}",
    ]
    if show_decisions:
        lines.append("decisions:")
        shown = False
        for decision in audit.decisions:
            if not decision.hard_veto and not decision.review_reasons:
                continue
            shown = True
            lines.append(
                f"  {Path(decision.path).name}: step={decision.step} "
                f"t={decision.game_time:.1f} action={decision.candidate_action} "
                f"use={decision.recommended_training_use} "
                f"quality={decision.label_quality} "
                f"hard_veto={str(decision.hard_veto).lower()} "
                f"hard_reasons={_inline_items(decision.hard_veto_reasons)} "
                f"review={_inline_items(decision.review_reasons)}"
            )
        if not shown:
            lines.append("  <none>")
    return "\n".join(lines)


def _inline_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _inline_items(items: list[str]) -> str:
    if not items:
        return "<none>"
    return ", ".join(items)


if __name__ == "__main__":
    raise SystemExit(main())
