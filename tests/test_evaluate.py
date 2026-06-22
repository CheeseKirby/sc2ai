from __future__ import annotations

from argparse import Namespace

import pytest

from rl.experiments import read_json
from scripts.evaluate import (
    make_eval_config,
    parse_result,
    policy_name,
    write_eval_summary,
)


@pytest.mark.unit
def test_parse_result_from_safe_launch_summary() -> None:
    output = "2026 INFO === Game result: Result.Victory ==="

    assert parse_result(output) == "Result.Victory"


@pytest.mark.unit
def test_parse_result_from_bot_end_line() -> None:
    output = "[ProtossRuleBot] Game ended. Result = Result.Defeat"

    assert parse_result(output) == "Result.Defeat"


@pytest.mark.unit
def test_parse_result_returns_none_when_missing() -> None:
    assert parse_result("no game result here") is None


@pytest.mark.unit
def test_make_eval_config_is_metadata_friendly() -> None:
    args = Namespace(
        maps=["AcropolisLE"],
        difficulties=["VeryEasy", "Easy"],
        opponents=["Protoss"],
        games_per_combo=2,
        record_decision_interval=16,
        guard_interval=0.02,
        hide_watch_seconds=120.0,
        hide_watch_interval=0.02,
        game_time_limit=300.0,
        army_policy="coverage-teacher",
        army_attack_threshold=10,
        army_retreat_threshold=7,
        retreat_peak_loss_ratio=0.35,
        retreat_min_peak_army=9,
        retreat_min_lost_from_peak=3,
        policy_name=None,
        policy_checkpoint=None,
        policy_device="cpu",
        trajectory_dir=None,
    )

    assert make_eval_config(args) == {
        "maps": ["AcropolisLE"],
        "difficulties": ["VeryEasy", "Easy"],
        "opponents": ["Protoss"],
        "games_per_combo": 2,
        "record_decision_interval": 16,
        "guard_interval": 0.02,
        "hide_watch_seconds": 120.0,
        "hide_watch_interval": 0.02,
        "game_time_limit": 300.0,
        "army_policy": "coverage-teacher",
        "army_attack_threshold": 10,
        "army_retreat_threshold": 7,
        "retreat_peak_loss_ratio": 0.35,
        "retreat_min_peak_army": 9,
        "retreat_min_lost_from_peak": 3,
        "policy_name": "rule",
        "policy_checkpoint": None,
        "policy_device": "cpu",
        "trajectory_dir": None,
    }


@pytest.mark.unit
def test_policy_name_defaults_to_rule_or_checkpoint_stem(tmp_path) -> None:
    assert policy_name(Namespace(policy_name=None, policy_checkpoint=None)) == "rule"
    assert (
        policy_name(
            Namespace(policy_name=None, policy_checkpoint=tmp_path / "policy.pt")
        )
        == "policy"
    )
    assert (
        policy_name(Namespace(policy_name="imitation_v1", policy_checkpoint=None))
        == "imitation_v1"
    )


@pytest.mark.unit
def test_write_eval_summary_creates_artifact_and_metadata_payload(tmp_path) -> None:
    records = [
        {
            "policy_name": "imitation_v1",
            "map_name": "AcropolisLE",
            "difficulty": "Easy",
            "opponent_race": "Protoss",
            "return_code": 0,
            "result": "Result.Victory",
            "duration_seconds": 12.0,
        },
        {
            "policy_name": "imitation_v1",
            "map_name": "AcropolisLE",
            "difficulty": "Easy",
            "opponent_race": "Protoss",
            "return_code": 0,
            "result": "Result.Tie",
            "duration_seconds": 18.0,
        },
    ]
    output = tmp_path / "eval.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = write_eval_summary(records, output=output, summary_path=summary_path)

    assert summary["output"] == str(output)
    assert summary["summary_json"] == str(summary_path)
    assert summary["games"] == 2
    assert summary["failures"] == 0
    assert summary["groups"][0]["victories"] == 1
    assert summary["groups"][0]["ties"] == 1
    assert read_json(summary_path) == summary["groups"]
