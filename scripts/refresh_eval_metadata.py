"""Refresh an evaluation run's metadata summary from its eval JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_eval import load_eval_records, summarize_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh eval run metadata summary")
    parser.add_argument("run_dir", type=Path, help="Evaluation run directory")
    return parser.parse_args()


def refresh_eval_metadata(run_dir: str | Path) -> dict:
    """Recompute summary fields for a standard evaluation run directory."""
    run = Path(run_dir)
    metadata_path = run / "metadata.json"
    eval_path = run / "artifacts" / "eval.jsonl"
    summary_path = run / "artifacts" / "summary.json"

    records = load_eval_records([eval_path])
    summaries = [asdict(summary) for summary in summarize_records(records)]
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    metadata["summary"] = {
        "output": str(eval_path),
        "summary_json": str(summary_path),
        "games": len(records),
        "failures": sum(1 for row in records if int(row.get("return_code", 1)) != 0),
        "groups": summaries,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    args = parse_args()
    metadata = refresh_eval_metadata(args.run_dir)
    print(
        f"Refreshed {args.run_dir}: "
        f"{metadata['summary']['games']} games, "
        f"{len(metadata['summary']['groups'])} groups"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
