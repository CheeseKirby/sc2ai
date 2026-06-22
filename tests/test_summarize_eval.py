from __future__ import annotations

import json

import pytest

from scripts.summarize_eval import (
    format_summary_table,
    load_eval_records,
    summarize_records,
)


@pytest.mark.unit
def test_summarize_records_groups_results() -> None:
    records = [
        {
            "map_name": "AcropolisLE",
            "difficulty": "Easy",
            "opponent_race": "Protoss",
            "return_code": 0,
            "result": "Result.Victory",
            "duration_seconds": 10.0,
        },
        {
            "map_name": "AcropolisLE",
            "difficulty": "Easy",
            "opponent_race": "Protoss",
            "return_code": 0,
            "result": "Result.Defeat",
            "duration_seconds": 20.0,
        },
        {
            "map_name": "AcropolisLE",
            "difficulty": "Medium",
            "opponent_race": "Zerg",
            "return_code": 1,
            "result": None,
            "duration_seconds": 5.0,
        },
        {
            "map_name": "AcropolisLE",
            "difficulty": "Medium",
            "opponent_race": "Zerg",
            "return_code": 0,
            "result": "Result.Tie",
            "duration_seconds": 15.0,
        },
    ]

    summaries = summarize_records(records)

    assert len(summaries) == 2
    easy = summaries[0]
    medium = summaries[1]
    assert easy.policy_name == "rule"
    assert easy.games == 2
    assert easy.victories == 1
    assert easy.defeats == 1
    assert easy.win_rate == 0.5
    assert easy.avg_duration_seconds == 15.0
    assert medium.no_result == 1
    assert medium.ties == 1
    assert medium.failures == 1


@pytest.mark.unit
def test_load_eval_records_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    rows = [
        {"map_name": "A", "difficulty": "Easy"},
        {"map_name": "B", "difficulty": "Medium"},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert load_eval_records([path]) == rows


@pytest.mark.unit
def test_format_summary_table_contains_win_rate() -> None:
    table = format_summary_table(
        summarize_records(
            [
                {
                    "policy_name": "imitation_v1",
                    "map_name": "AcropolisLE",
                    "difficulty": "Easy",
                    "opponent_race": "Protoss",
                    "return_code": 0,
                    "result": "Result.Victory",
                    "duration_seconds": 10.0,
                }
            ]
        )
    )

    assert "AcropolisLE" in table
    assert "imitation_v1" in table
    assert "ties" in table
    assert "100.0%" in table
