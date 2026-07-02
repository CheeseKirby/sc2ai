from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rl.experiments import read_json
from scripts.evaluate import (
    make_eval_config,
    parse_result,
    parse_args,
    policy_name,
    run_one_game,
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
def test_parse_args_defaults_to_random_ai_build(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["evaluate.py"])

    args = parse_args()

    assert args.ai_builds == ["RandomBuild"]
    assert args.strategy_tactic_mode == "off"
    assert args.strategy_teacher_profile == "standard"


@pytest.mark.unit
def test_make_eval_config_is_metadata_friendly() -> None:
    args = Namespace(
        maps=["AcropolisLE"],
        difficulties=["VeryEasy", "Easy"],
        opponents=["Protoss"],
        ai_builds=["Rush", "Macro"],
        games_per_combo=2,
        record_decision_interval=16,
        guard_interval=0.02,
        hide_watch_seconds=120.0,
        hide_watch_interval=0.02,
        game_time_limit=300.0,
        army_policy="coverage-teacher",
        strategy_policy="coverage-teacher",
        strategy_tactic_mode="rule",
        strategy_teacher_profile="pre-collapse-recovery",
        army_attack_threshold=10,
        army_retreat_threshold=7,
        retreat_peak_loss_ratio=0.35,
        retreat_min_peak_army=9,
        retreat_min_lost_from_peak=3,
        policy_name=None,
        policy_checkpoint=None,
        policy_device="cpu",
        strategy_checkpoint=None,
        strategy_device="cpu",
        strategy_action_critic_checkpoint=None,
        strategy_action_critic_threshold=0.5,
        strategy_action_critic_fallback_policy="lowest-risk",
        trajectory_dir=None,
        strategy_trajectory_dir=None,
    )

    assert make_eval_config(args) == {
        "maps": ["AcropolisLE"],
        "difficulties": ["VeryEasy", "Easy"],
        "opponents": ["Protoss"],
        "ai_builds": ["Rush", "Macro"],
        "games_per_combo": 2,
        "record_decision_interval": 16,
        "guard_interval": 0.02,
        "hide_watch_seconds": 120.0,
        "hide_watch_interval": 0.02,
        "game_time_limit": 300.0,
        "army_policy": "coverage-teacher",
        "strategy_policy": "coverage-teacher",
        "strategy_tactic_mode": "rule",
        "strategy_teacher_profile": "pre-collapse-recovery",
        "army_attack_threshold": 10,
        "army_retreat_threshold": 7,
        "retreat_peak_loss_ratio": 0.35,
        "retreat_min_peak_army": 9,
        "retreat_min_lost_from_peak": 3,
        "policy_name": "rule",
        "policy_checkpoint": None,
        "policy_device": "cpu",
        "strategy_checkpoint": None,
        "strategy_device": "cpu",
        "strategy_action_critic_checkpoint": None,
        "strategy_action_critic_threshold": 0.5,
        "strategy_action_critic_fallback_policy": "lowest-risk",
        "trajectory_dir": None,
        "strategy_trajectory_dir": None,
    }


@pytest.mark.unit
def test_policy_name_defaults_to_rule_or_checkpoint_stem(tmp_path) -> None:
    assert (
        policy_name(
            Namespace(
                policy_name=None,
                policy_checkpoint=None,
                strategy_checkpoint=None,
            )
        )
        == "rule"
    )
    assert (
        policy_name(
            Namespace(
                policy_name=None,
                policy_checkpoint=tmp_path / "policy.pt",
                strategy_checkpoint=None,
            )
        )
        == "policy"
    )
    assert (
        policy_name(
            Namespace(
                policy_name="imitation_v1",
                policy_checkpoint=None,
                strategy_checkpoint=None,
            )
        )
        == "imitation_v1"
    )
    assert (
        policy_name(
            Namespace(
                policy_name=None,
                policy_checkpoint=None,
                strategy_checkpoint=tmp_path / "strategy_policy.pt",
            )
        )
        == "strategy_policy"
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


@pytest.mark.unit
def test_run_one_game_forwards_strategy_trajectory_path(tmp_path) -> None:
    army_dir = tmp_path / "army"
    strategy_dir = tmp_path / "strategy"
    with patch("scripts.evaluate.subprocess.run") as run_mock:
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout="2026 INFO === Game result: Result.Victory ===",
        )

        record = run_one_game(
            map_name="AcropolisLE",
            difficulty="Easy",
            opponent_race="Protoss",
            opponent_ai_build="Rush",
            game_index=1,
            trajectory_dir=army_dir,
            strategy_trajectory_dir=strategy_dir,
            record_decision_interval=16,
            guard_interval=0.02,
            hide_watch_seconds=120.0,
            hide_watch_interval=0.02,
            game_time_limit=300.0,
            army_policy="rule",
            strategy_policy="rule",
            strategy_tactic_mode="off",
            strategy_teacher_profile="standard",
            army_attack_threshold=None,
            army_retreat_threshold=None,
            retreat_peak_loss_ratio=None,
            retreat_min_peak_army=None,
            retreat_min_lost_from_peak=None,
            policy_name="rule",
            policy_checkpoint=None,
            policy_device="cpu",
            strategy_checkpoint=None,
            strategy_device="cpu",
            strategy_action_critic_checkpoint=None,
            strategy_action_critic_threshold=0.5,
            strategy_action_critic_fallback_policy="lowest-risk",
            llm_provider=None,
            llm_model=None,
            llm_base_url=None,
            llm_api_key_env=None,
            llm_timeout=None,
            llm_decision_interval=None,
            llm_temperature=None,
            llm_max_output_tokens=None,
            llm_allow_no_api_key=False,
            llm_log_dir=None,
        )

    command = run_mock.call_args.args[0]
    assert "--trajectory-path" in command
    assert "--strategy-trajectory-path" in command
    assert "--ai-build" in command
    assert command[command.index("--ai-build") + 1] == "Rush"
    strategy_index = command.index("--strategy-trajectory-path") + 1
    assert str(strategy_dir) in command[strategy_index]
    assert record.trajectory_path is not None
    assert record.strategy_trajectory_path is not None
    assert "Rush" in record.trajectory_path
    assert "strategy" in record.strategy_trajectory_path
    assert "Rush" in record.strategy_trajectory_path
    assert record.opponent_ai_build == "Rush"


@pytest.mark.unit
def test_run_one_game_forwards_strategy_policy(tmp_path) -> None:
    with patch("scripts.evaluate.subprocess.run") as run_mock:
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout="2026 INFO === Game result: Result.Victory ===",
        )

        run_one_game(
            map_name="AcropolisLE",
            difficulty="Easy",
            opponent_race="Protoss",
            opponent_ai_build="RandomBuild",
            game_index=1,
            trajectory_dir=None,
            strategy_trajectory_dir=None,
            record_decision_interval=16,
            guard_interval=0.02,
            hide_watch_seconds=120.0,
            hide_watch_interval=0.02,
            game_time_limit=None,
            army_policy="rule",
            strategy_policy="coverage-teacher",
            strategy_tactic_mode="rule",
            strategy_teacher_profile="standard",
            army_attack_threshold=None,
            army_retreat_threshold=None,
            retreat_peak_loss_ratio=None,
            retreat_min_peak_army=None,
            retreat_min_lost_from_peak=None,
            policy_name="rule",
            policy_checkpoint=None,
            policy_device="cpu",
            strategy_checkpoint=None,
            strategy_device="cpu",
            strategy_action_critic_checkpoint=None,
            strategy_action_critic_threshold=0.5,
            strategy_action_critic_fallback_policy="lowest-risk",
            llm_provider=None,
            llm_model=None,
            llm_base_url=None,
            llm_api_key_env=None,
            llm_timeout=None,
            llm_decision_interval=None,
            llm_temperature=None,
            llm_max_output_tokens=None,
            llm_allow_no_api_key=False,
            llm_log_dir=None,
        )

    command = run_mock.call_args.args[0]
    assert "--strategy-policy" in command
    strategy_index = command.index("--strategy-policy") + 1
    assert command[strategy_index] == "coverage-teacher"
    assert "--strategy-tactic-mode" in command
    tactic_index = command.index("--strategy-tactic-mode") + 1
    assert command[tactic_index] == "rule"


@pytest.mark.unit
def test_run_one_game_forwards_strategy_teacher_profile(tmp_path) -> None:
    with patch("scripts.evaluate.subprocess.run") as run_mock:
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout="2026 INFO === Game result: Result.Victory ===",
        )

        record = run_one_game(
            map_name="ThunderbirdLE",
            difficulty="Hard",
            opponent_race="Terran",
            opponent_ai_build="Power",
            game_index=1,
            trajectory_dir=None,
            strategy_trajectory_dir=None,
            record_decision_interval=16,
            guard_interval=0.02,
            hide_watch_seconds=120.0,
            hide_watch_interval=0.02,
            game_time_limit=None,
            army_policy="rule",
            strategy_policy="coverage-teacher",
            strategy_tactic_mode="off",
            strategy_teacher_profile="pre-collapse-recovery",
            army_attack_threshold=None,
            army_retreat_threshold=None,
            retreat_peak_loss_ratio=None,
            retreat_min_peak_army=None,
            retreat_min_lost_from_peak=None,
            policy_name="rule",
            policy_checkpoint=None,
            policy_device="cpu",
            strategy_checkpoint=None,
            strategy_device="cpu",
            strategy_action_critic_checkpoint=None,
            strategy_action_critic_threshold=0.5,
            strategy_action_critic_fallback_policy="lowest-risk",
            llm_provider=None,
            llm_model=None,
            llm_base_url=None,
            llm_api_key_env=None,
            llm_timeout=None,
            llm_decision_interval=None,
            llm_temperature=None,
            llm_max_output_tokens=None,
            llm_allow_no_api_key=False,
            llm_log_dir=None,
        )

    command = run_mock.call_args.args[0]
    assert "--strategy-teacher-profile" in command
    profile_index = command.index("--strategy-teacher-profile") + 1
    assert command[profile_index] == "pre-collapse-recovery"
    assert record.strategy_teacher_profile == "pre-collapse-recovery"


@pytest.mark.unit
def test_run_one_game_forwards_strategy_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "strategy.pt"
    critic_checkpoint = tmp_path / "critic.pt"
    with patch("scripts.evaluate.subprocess.run") as run_mock:
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout="2026 INFO === Game result: Result.Victory ===",
        )

        record = run_one_game(
            map_name="AcropolisLE",
            difficulty="Easy",
            opponent_race="Protoss",
            opponent_ai_build="RandomBuild",
            game_index=1,
            trajectory_dir=None,
            strategy_trajectory_dir=None,
            record_decision_interval=16,
            guard_interval=0.02,
            hide_watch_seconds=120.0,
            hide_watch_interval=0.02,
            game_time_limit=None,
            army_policy="rule",
            strategy_policy="checkpoint",
            strategy_tactic_mode="off",
            strategy_teacher_profile="standard",
            army_attack_threshold=None,
            army_retreat_threshold=None,
            retreat_peak_loss_ratio=None,
            retreat_min_peak_army=None,
            retreat_min_lost_from_peak=None,
            policy_name="strategy_policy",
            policy_checkpoint=None,
            policy_device="cpu",
            strategy_checkpoint=checkpoint,
            strategy_device="cpu",
            strategy_action_critic_checkpoint=critic_checkpoint,
            strategy_action_critic_threshold=0.95,
            strategy_action_critic_fallback_policy="first-executable",
            llm_provider=None,
            llm_model=None,
            llm_base_url=None,
            llm_api_key_env=None,
            llm_timeout=None,
            llm_decision_interval=None,
            llm_temperature=None,
            llm_max_output_tokens=None,
            llm_allow_no_api_key=False,
            llm_log_dir=None,
        )

    command = run_mock.call_args.args[0]
    assert "--strategy-policy" in command
    assert command[command.index("--strategy-policy") + 1] == "checkpoint"
    assert "--strategy-checkpoint" in command
    assert command[command.index("--strategy-checkpoint") + 1] == str(checkpoint)
    assert "--strategy-device" in command
    assert "--strategy-action-critic-checkpoint" in command
    assert (
        command[command.index("--strategy-action-critic-checkpoint") + 1]
        == str(critic_checkpoint)
    )
    assert "--strategy-action-critic-threshold" in command
    assert command[command.index("--strategy-action-critic-threshold") + 1] == "0.95"
    assert "--strategy-action-critic-fallback-policy" in command
    assert (
        command[command.index("--strategy-action-critic-fallback-policy") + 1]
        == "first-executable"
    )
    assert record.policy_name == "strategy_policy"
    assert record.strategy_policy == "checkpoint"
    assert record.strategy_tactic_mode == "off"
    assert record.strategy_checkpoint == str(checkpoint)
    assert record.strategy_action_critic_checkpoint == str(critic_checkpoint)
    assert record.strategy_action_critic_threshold == 0.95
    assert record.strategy_action_critic_fallback_policy == "first-executable"
