"""Summarize evaluation JSONL files produced by scripts/evaluate.py."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EvalSummary:
    policy_name: str
    map_name: str
    difficulty: str
    opponent_race: str
    games: int
    victories: int
    defeats: int
    ties: int
    no_result: int
    failures: int
    win_rate: float
    avg_duration_seconds: float


def load_eval_records(paths: Iterable[str | Path]) -> list[dict]:
    """Load records from one or more evaluation JSONL files."""
    records: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
    return records


def summarize_records(records: Iterable[dict]) -> list[EvalSummary]:
    """Aggregate eval records by map/difficulty/opponent."""
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("policy_name", "rule")),
            str(record.get("map_name", "")),
            str(record.get("difficulty", "")),
            str(record.get("opponent_race", "")),
        )
        groups[key].append(record)

    summaries: list[EvalSummary] = []
    for (policy_name, map_name, difficulty, opponent_race), rows in sorted(
        groups.items()
    ):
        games = len(rows)
        victories = sum(1 for row in rows if row.get("result") == "Result.Victory")
        defeats = sum(1 for row in rows if row.get("result") == "Result.Defeat")
        ties = sum(1 for row in rows if row.get("result") == "Result.Tie")
        no_result = sum(1 for row in rows if not row.get("result"))
        failures = sum(1 for row in rows if int(row.get("return_code", 1)) != 0)
        durations = [float(row.get("duration_seconds", 0.0)) for row in rows]
        summaries.append(
            EvalSummary(
                policy_name=policy_name,
                map_name=map_name,
                difficulty=difficulty,
                opponent_race=opponent_race,
                games=games,
                victories=victories,
                defeats=defeats,
                ties=ties,
                no_result=no_result,
                failures=failures,
                win_rate=victories / games if games else 0.0,
                avg_duration_seconds=sum(durations) / games if games else 0.0,
            )
        )
    return summaries


def format_summary_table(summaries: list[EvalSummary]) -> str:
    """Format summaries as a compact plain-text table."""
    headers = [
        "policy",
        "map",
        "difficulty",
        "opponent",
        "games",
        "wins",
        "losses",
        "ties",
        "no_result",
        "failures",
        "win_rate",
        "avg_s",
    ]
    rows = [
        [
            summary.policy_name,
            summary.map_name,
            summary.difficulty,
            summary.opponent_race,
            str(summary.games),
            str(summary.victories),
            str(summary.defeats),
            str(summary.ties),
            str(summary.no_result),
            str(summary.failures),
            f"{summary.win_rate:.1%}",
            f"{summary.avg_duration_seconds:.1f}",
        ]
        for summary in summaries
    ]
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    lines = [_format_row(headers, widths)]
    lines.append(_format_row(["-" * width for width in widths], widths))
    lines.extend(_format_row(row, widths) for row in rows)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize eval JSONL files")
    parser.add_argument("inputs", nargs="+", type=Path, help="Eval JSONL files")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for machine-readable summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = summarize_records(load_eval_records(args.inputs))
    print(format_summary_table(summaries))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                [asdict(summary) for summary in summaries],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return 0


def _format_row(row: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(row, widths))


if __name__ == "__main__":
    raise SystemExit(main())
