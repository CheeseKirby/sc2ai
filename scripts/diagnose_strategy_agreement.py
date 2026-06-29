"""Diagnose offline agreement between strategy teacher and checkpoint policy."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import write_json  # noqa: E402
from rl.strategy_agreement_diagnostics import (  # noqa: E402
    StrategyAgreementBucketSummary,
    StrategyAgreementDiagnostics,
    diagnose_strategy_agreement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose offline agreement between strategy teacher and checkpoint"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or dirs")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Strategy checkpoint path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable agreement diagnostics output path.",
    )
    parser.add_argument(
        "--show-buckets",
        action="store_true",
        help="Print time and state bucket summaries.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file agreement summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = diagnose_strategy_agreement(args.inputs, args.checkpoint)
    print(
        format_strategy_agreement_diagnostics(
            diagnostics,
            show_buckets=args.show_buckets,
            show_files=args.show_files,
        )
    )
    if args.json_output is not None:
        write_json(args.json_output, asdict(diagnostics))
    return 0


def format_strategy_agreement_diagnostics(
    diagnostics: StrategyAgreementDiagnostics,
    *,
    show_buckets: bool = False,
    show_files: bool = False,
) -> str:
    """Return a compact human-readable agreement report."""
    lines = [
        "Strategy agreement diagnostics",
        f"inputs: {', '.join(diagnostics.inputs)}",
        f"checkpoint: {diagnostics.checkpoint_path}",
        f"files: {diagnostics.files}",
        f"rows: {diagnostics.rows}",
        (
            "accuracy: "
            f"stored_vs_teacher_accuracy={diagnostics.stored_vs_teacher_accuracy:.3f} "
            f"checkpoint_vs_teacher_accuracy={diagnostics.checkpoint_vs_teacher_accuracy:.3f} "
            f"checkpoint_vs_stored_accuracy={diagnostics.checkpoint_vs_stored_accuracy:.3f}"
        ),
        "observation_schema_counts:",
    ]
    if diagnostics.observation_schema_counts:
        for schema, count in diagnostics.observation_schema_counts.items():
            lines.append(f"  {schema}: {count}")
    else:
        lines.append("  <none>")
    lines.append(
        f"rows_defaulted_observation_fields: "
        f"{diagnostics.rows_defaulted_observation_fields}"
    )

    _append_counts(lines, "stored_action_counts:", diagnostics.stored_action_counts_by_name)
    _append_counts(lines, "teacher_action_counts:", diagnostics.teacher_action_counts_by_name)
    _append_counts(
        lines,
        "checkpoint_action_counts:",
        diagnostics.checkpoint_action_counts_by_name,
    )
    _append_counts(
        lines,
        "mismatch_counts_by_teacher:",
        diagnostics.mismatch_counts_by_teacher_name,
    )
    _append_confusion(
        lines,
        "confusion_teacher_to_checkpoint:",
        diagnostics.confusion_matrix_teacher_to_checkpoint_by_name,
    )

    if show_buckets:
        lines.append("time_buckets:")
        _append_bucket_summaries(lines, diagnostics.time_buckets)
        lines.append("state_buckets:")
        _append_bucket_summaries(lines, diagnostics.state_buckets)

    if show_files:
        lines.append("files:")
        if diagnostics.file_summaries:
            for summary in diagnostics.file_summaries:
                mismatches = ", ".join(
                    f"{action}={count}"
                    for action, count in summary.mismatch_counts_by_teacher_name.items()
                )
                lines.append(
                    f"  {summary.path}: "
                    f"difficulty={summary.difficulty or '<unknown>'} "
                    f"opponent={summary.opponent_race or '<unknown>'} "
                    f"rows={summary.rows} "
                    f"time={_format_optional_float(summary.first_game_time)}-"
                    f"{_format_optional_float(summary.last_game_time)} "
                    f"stored_vs_teacher={summary.stored_vs_teacher_accuracy:.3f} "
                    f"checkpoint_vs_teacher={summary.checkpoint_vs_teacher_accuracy:.3f} "
                    f"checkpoint_vs_stored={summary.checkpoint_vs_stored_accuracy:.3f} "
                    f"mismatches_by_teacher={mismatches or '<none>'}"
                )
        else:
            lines.append("  <none>")

    return "\n".join(lines)


def _append_bucket_summaries(
    lines: list[str],
    buckets: dict[str, StrategyAgreementBucketSummary],
) -> None:
    if not buckets:
        lines.append("  <none>")
        return
    for name, summary in buckets.items():
        lines.append(
            f"  {name}: rows={summary.rows} "
            f"stored_vs_teacher={summary.stored_vs_teacher_accuracy:.3f} "
            f"checkpoint_vs_teacher={summary.checkpoint_vs_teacher_accuracy:.3f} "
            f"checkpoint_vs_stored={summary.checkpoint_vs_stored_accuracy:.3f}"
        )
        if summary.mismatch_counts_by_teacher_name:
            mismatches = ", ".join(
                f"{action}={count}"
                for action, count in summary.mismatch_counts_by_teacher_name.items()
            )
            lines.append(f"    mismatches_by_teacher: {mismatches}")


def _append_counts(lines: list[str], title: str, counts: dict[str, int]) -> None:
    lines.append(title)
    if not counts:
        lines.append("  <none>")
        return
    for name, count in counts.items():
        lines.append(f"  {name}: {count}")


def _append_confusion(
    lines: list[str],
    title: str,
    confusion: dict[str, dict[str, int]],
) -> None:
    lines.append(title)
    if not confusion:
        lines.append("  <none>")
        return
    for teacher_action, checkpoint_counts in confusion.items():
        counts = ", ".join(
            f"{checkpoint_action}={count}"
            for checkpoint_action, count in checkpoint_counts.items()
        )
        lines.append(f"  {teacher_action}: {counts}")


def _format_optional_float(value: float | None) -> str:
    return "none" if value is None else f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
