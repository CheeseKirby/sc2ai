from __future__ import annotations

import json

import pytest
import torch

from bot.managers.rl_strategy_policy import RLStrategyPolicy as _RLStrategyPolicy
from rl.models import PolicyModelSpec, build_policy_model
from rl.strategy_actions import StrategyAction
from rl.strategy_agreement_diagnostics import diagnose_strategy_agreement
from rl.strategy_checkpoints import save_strategy_policy_checkpoint
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.diagnose_strategy_agreement import format_strategy_agreement_diagnostics

_ = _RLStrategyPolicy


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "game_time": 120.0,
            "minerals": 500.0,
            "vespene": 100.0,
            "supply_left": 10.0,
            "workers": 22.0,
            "own_bases": 2.0,
            "ready_gateways": 2.0,
            "has_cybernetics_core": 1.0,
            "army_count": 12.0,
            "worker_saturation_ratio": 1.0,
        }
    )
    observation.update(overrides)
    return observation


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _constant_strategy_checkpoint(tmp_path, action: StrategyAction):
    model = build_policy_model(
        PolicyModelSpec(
            observation_dim=len(STRATEGY_OBSERVATION_FIELDS),
            action_dim=len(StrategyAction),
            hidden_sizes=(),
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias[int(action)] = 10.0
    checkpoint = tmp_path / f"{action.name.lower()}.pt"
    save_strategy_policy_checkpoint(checkpoint, model)
    return checkpoint


@pytest.mark.unit
def test_strategy_agreement_diagnostics_reports_time_and_state_buckets(tmp_path) -> None:
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.STAY_COURSE)
    trajectory = tmp_path / "teacher.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 64,
                "difficulty": "Hard",
                "opponent_race": "Terran",
                "strategy_observation": _observation(game_time=120.0),
                "strategy_action": int(StrategyAction.ADD_GATEWAYS),
                "done": False,
            },
            {
                "step": 128,
                "difficulty": "Hard",
                "opponent_race": "Terran",
                "strategy_observation": _observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    ready_static_defense=0.0,
                ),
                "strategy_action": int(StrategyAction.BUILD_STATIC_DEFENSE),
                "done": False,
            },
            {
                "step": 192,
                "difficulty": "Hard",
                "opponent_race": "Terran",
                "strategy_observation": _observation(game_time=260.0),
                "strategy_action": int(StrategyAction.STAY_COURSE),
                "done": True,
                "result": "Result.Defeat",
            },
        ],
    )

    diagnostics = diagnose_strategy_agreement(
        trajectory,
        checkpoint,
    )

    assert diagnostics.rows == 2
    assert diagnostics.stored_vs_teacher_accuracy == 1.0
    assert diagnostics.checkpoint_vs_teacher_accuracy == 0.0
    assert diagnostics.teacher_action_counts_by_name == {
        "ADD_GATEWAYS": 1,
        "BUILD_STATIC_DEFENSE": 1,
    }
    assert diagnostics.checkpoint_action_counts_by_name == {"STAY_COURSE": 2}
    assert diagnostics.confusion_matrix_teacher_to_checkpoint_by_name == {
        "ADD_GATEWAYS": {"STAY_COURSE": 1},
        "BUILD_STATIC_DEFENSE": {"STAY_COURSE": 1},
    }
    assert diagnostics.time_buckets["0-180"].rows == 1
    assert diagnostics.time_buckets["180-360"].rows == 1
    assert diagnostics.state_buckets["gateway_scaling_needed"].rows == 2
    assert diagnostics.state_buckets["base_under_threat"].rows == 1
    assert diagnostics.state_buckets["difficulty:Hard"].rows == 2
    assert diagnostics.state_buckets["opponent:Terran"].rows == 2
    assert len(diagnostics.file_summaries) == 1
    assert diagnostics.file_summaries[0].path == str(trajectory)
    assert diagnostics.file_summaries[0].checkpoint_vs_teacher_accuracy == 0.0


@pytest.mark.unit
def test_format_strategy_agreement_diagnostics_includes_key_sections(tmp_path) -> None:
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.STAY_COURSE)
    trajectory = tmp_path / "teacher.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 64,
                "strategy_observation": _observation(game_time=120.0),
                "strategy_action": int(StrategyAction.ADD_GATEWAYS),
                "done": False,
            },
        ],
    )

    report = format_strategy_agreement_diagnostics(
        diagnose_strategy_agreement(trajectory, checkpoint),
        show_buckets=True,
        show_files=True,
    )

    assert "Strategy agreement diagnostics" in report
    assert "checkpoint_vs_teacher_accuracy=0.000" in report
    assert "teacher_action_counts:" in report
    assert "confusion_teacher_to_checkpoint:" in report
    assert "time_buckets:" in report
    assert "state_buckets:" in report
    assert "files:" in report
