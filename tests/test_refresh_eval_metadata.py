from __future__ import annotations

import json

import pytest

from scripts.refresh_eval_metadata import refresh_eval_metadata


@pytest.mark.unit
def test_refresh_eval_metadata_recomputes_groups(tmp_path) -> None:
    run = tmp_path / "run"
    artifacts = run / "artifacts"
    artifacts.mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps({"status": "complete", "summary": {"groups": []}}),
        encoding="utf-8",
    )
    (artifacts / "eval.jsonl").write_text(
        json.dumps(
            {
                "policy_name": "rule",
                "map_name": "AcropolisLE",
                "difficulty": "Easy",
                "opponent_race": "Terran",
                "return_code": 0,
                "result": "Result.Tie",
                "duration_seconds": 10.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = refresh_eval_metadata(run)

    assert metadata["summary"]["games"] == 1
    assert metadata["summary"]["groups"][0]["ties"] == 1
    assert (artifacts / "summary.json").is_file()

