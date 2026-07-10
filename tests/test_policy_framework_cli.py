from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import run
from scripts import evaluate


@pytest.mark.unit
@pytest.mark.parametrize("strategy_policy", ["ppo", "llm"])
def test_run_parse_args_accepts_new_strategy_policy_modes(
    monkeypatch,
    strategy_policy: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--strategy-policy", strategy_policy],
    )

    args = run.parse_args()

    assert args.strategy_policy == strategy_policy


@pytest.mark.unit
@pytest.mark.parametrize("strategy_policy", ["ppo", "llm"])
def test_evaluate_parse_args_accepts_new_strategy_policy_modes(
    monkeypatch,
    strategy_policy: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--strategy-policy", strategy_policy],
    )

    args = evaluate.parse_args()

    assert args.strategy_policy == strategy_policy


def _run_one_game(
    *,
    strategy_policy: str,
    strategy_checkpoint: Path | None,
    llm_model: str | None = None,
) -> tuple[object, list[str]]:
    with patch("scripts.evaluate.subprocess.run") as run_mock:
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout="2026 INFO === Game result: Result.Victory ===",
        )
        record = evaluate.run_one_game(
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
            strategy_policy=strategy_policy,
            strategy_tactic_mode="off",
            strategy_teacher_profile="standard",
            army_attack_threshold=None,
            army_retreat_threshold=None,
            retreat_peak_loss_ratio=None,
            retreat_min_peak_army=None,
            retreat_min_lost_from_peak=None,
            policy_name="framework_policy",
            policy_checkpoint=None,
            policy_device="cpu",
            strategy_checkpoint=strategy_checkpoint,
            strategy_device="cpu",
            strategy_action_critic_checkpoint=None,
            strategy_action_critic_threshold=0.5,
            strategy_action_critic_fallback_policy="lowest-risk",
            llm_provider="openai-responses" if llm_model else None,
            llm_model=llm_model,
            llm_base_url=None,
            llm_api_key_env=None,
            llm_timeout=None,
            llm_decision_interval=None,
            llm_temperature=None,
            llm_max_output_tokens=None,
            llm_allow_no_api_key=False,
            llm_log_dir=None,
        )
    return record, run_mock.call_args.args[0]


@pytest.mark.unit
def test_evaluate_forwards_ppo_strategy_checkpoint() -> None:
    checkpoint = Path("runs/example/checkpoints/strategy_ppo.zip")

    record, command = _run_one_game(
        strategy_policy="ppo",
        strategy_checkpoint=checkpoint,
    )

    assert command[command.index("--strategy-policy") + 1] == "ppo"
    assert command[command.index("--strategy-checkpoint") + 1] == str(checkpoint)
    assert "--strategy-device" in command
    assert record.strategy_policy == "ppo"


@pytest.mark.unit
def test_evaluate_forwards_llm_config_for_strategy_llm() -> None:
    record, command = _run_one_game(
        strategy_policy="llm",
        strategy_checkpoint=None,
        llm_model="test-strategy-model",
    )

    assert command[command.index("--strategy-policy") + 1] == "llm"
    assert command[command.index("--llm-model") + 1] == "test-strategy-model"
    assert record.strategy_policy == "llm"
