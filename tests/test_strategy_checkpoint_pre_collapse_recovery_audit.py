from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch

from bot.managers.rl_strategy_policy import RLStrategyPolicy as _RLStrategyPolicy
from rl.models import PolicyModelSpec, build_policy_model
from rl.strategy_actions import StrategyAction
from rl.strategy_checkpoint_pre_collapse_recovery_audit import (
    audit_strategy_checkpoint_pre_collapse_recovery,
)
from rl.strategy_checkpoints import save_strategy_policy_checkpoint
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.audit_strategy_checkpoint_pre_collapse_recovery import (
    format_strategy_checkpoint_pre_collapse_recovery_audit,
    main as checkpoint_pre_collapse_main,
)

_ = _RLStrategyPolicy


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "game_time": 100.0,
            "minerals": 200.0,
            "vespene": 150.0,
            "supply_left": 8.0,
            "workers": 24.0,
            "own_bases": 1.0,
            "ready_gateways": 1.0,
            "pending_gateways": 0.0,
            "gateway_idle_count": 1.0,
            "ready_robo": 0.0,
            "pending_robo": 0.0,
            "ready_forge": 1.0,
            "ready_static_defense": 0.0,
            "pending_static_defense": 0.0,
            "has_cybernetics_core": 1.0,
            "army_count": 8.0,
            "worker_saturation_ratio": 0.9,
        }
    )
    observation.update(overrides)
    return observation


def _row(**overrides) -> dict:
    row = {
        "step": 64,
        "map_name": "AcropolisLE",
        "difficulty": "Hard",
        "opponent_race": "Terran",
        "opponent_ai_build": "Power",
        "strategy_observation": _observation(),
        "strategy_action": int(StrategyAction.STAY_COURSE),
        "strategy_action_name": "STAY_COURSE",
        "done": False,
    }
    row.update(overrides)
    return row


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _pre_collapse_fixture(path) -> None:
    _write_jsonl(
        path,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.TECH_ROBO),
                strategy_action_name="TECH_ROBO",
                strategy_observation=_observation(
                    game_time=80.0,
                    minerals=200.0,
                    vespene=150.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                ),
                strategy_execution_attempted=True,
                strategy_execution_effect="build_structure",
                strategy_execution_unit_type="ROBOTICSFACILITY",
            ),
            _row(
                step=96,
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=50.0,
                    vespene=130.0,
                    pending_robo=1.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    minerals=25.0,
                    vespene=0.0,
                    supply_left=0.0,
                    own_bases=1.0,
                    ready_gateways=4.0,
                    pending_gateways=0.0,
                    gateway_idle_count=0.0,
                    has_cybernetics_core=0.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    ready_forge=0.0,
                    ready_static_defense=2.0,
                    pending_static_defense=0.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(
                done=True,
                result="Result.Defeat",
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
            ),
        ],
    )


@pytest.mark.unit
def test_checkpoint_pre_collapse_audit_holds_when_checkpoint_misses_recovery(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _pre_collapse_fixture(trajectory)
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.STAY_COURSE)

    audit = audit_strategy_checkpoint_pre_collapse_recovery(
        trajectory,
        checkpoint,
        prediction_mode="executable-mask",
    )

    assert audit.recommendation == "hold"
    assert audit.blocking_reasons == [
        "missed_checkpoint_pre_collapse_recovery_rows",
        "missed_checkpoint_pre_collapse_recovery_rate",
        "missed_accept_positive_recovery_rows",
        "missed_accept_positive_recovery_rate",
    ]
    assert audit.target_rows == 1
    assert audit.rows_with_pre_collapse_recovery_window == 1
    assert audit.rows_with_checkpoint_pre_collapse_executable_recovery == 0
    assert audit.missed_checkpoint_pre_collapse_recovery_rows == 1
    assert audit.missed_checkpoint_pre_collapse_recovery_rate == 1.0
    assert audit.accept_positive_recovery_rows == 1
    assert audit.accept_positive_recovery_matches == 0
    assert audit.missed_accept_positive_recovery_rows == 1
    assert audit.accept_positive_recovery_recorded_counts_by_action == {
        "TECH_ROBO": 1
    }
    assert audit.accept_positive_recovery_predicted_counts_by_action == {
        "STAY_COURSE": 1
    }

    target = audit.targets[0]
    assert target.recorded_training_use == "action_space_exhausted"
    assert target.recovery_windows[0].predicted_action == "STAY_COURSE"
    assert target.recovery_windows[0].predicted_executable_recovery_action is None
    assert asdict(audit)["targets"][0]["missed_checkpoint_pre_collapse_recovery"]


@pytest.mark.unit
def test_checkpoint_pre_collapse_audit_ready_when_checkpoint_predicts_recovery(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _pre_collapse_fixture(trajectory)
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.TECH_ROBO)

    audit = audit_strategy_checkpoint_pre_collapse_recovery(
        trajectory,
        checkpoint,
        prediction_mode="executable-mask",
    )

    assert audit.recommendation == "ready"
    assert audit.blocking_reasons == []
    assert audit.rows_with_checkpoint_pre_collapse_executable_recovery == 1
    assert audit.missed_checkpoint_pre_collapse_recovery_rows == 0
    assert audit.accept_positive_recovery_matches == 1
    assert audit.checkpoint_predicted_executable_recovery_counts_by_action == {
        "TECH_ROBO": 1
    }
    assert audit.targets[0].last_checkpoint_executable_recovery_action == "TECH_ROBO"
    assert audit.accept_positive_recovery_decisions[0].prediction_matches_recorded


@pytest.mark.unit
def test_checkpoint_pre_collapse_audit_validates_arguments(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _pre_collapse_fixture(trajectory)
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.TECH_ROBO)

    with pytest.raises(ValueError, match="lookback_seconds"):
        audit_strategy_checkpoint_pre_collapse_recovery(
            trajectory,
            checkpoint,
            lookback_seconds=0,
        )
    with pytest.raises(ValueError, match="max_targets"):
        audit_strategy_checkpoint_pre_collapse_recovery(
            trajectory,
            checkpoint,
            max_targets=-1,
        )
    with pytest.raises(ValueError, match="max_windows_per_target"):
        audit_strategy_checkpoint_pre_collapse_recovery(
            trajectory,
            checkpoint,
            max_windows_per_target=-1,
        )
    with pytest.raises(ValueError, match="max_accept_positive_decisions"):
        audit_strategy_checkpoint_pre_collapse_recovery(
            trajectory,
            checkpoint,
            max_accept_positive_decisions=-1,
        )
    with pytest.raises(ValueError, match="max_missed_accept_positive_recovery_rate"):
        audit_strategy_checkpoint_pre_collapse_recovery(
            trajectory,
            checkpoint,
            max_missed_accept_positive_recovery_rate=1.1,
        )


@pytest.mark.unit
def test_format_checkpoint_pre_collapse_audit_includes_slices(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _pre_collapse_fixture(trajectory)
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.STAY_COURSE)

    report = format_strategy_checkpoint_pre_collapse_recovery_audit(
        audit_strategy_checkpoint_pre_collapse_recovery(trajectory, checkpoint)
    )

    assert "Strategy checkpoint pre-collapse recovery audit" in report
    assert "recommendation: hold" in report
    assert "missed_checkpoint_pre_collapse_recovery_rows: 1/1" in report
    assert "accept_positive_recovery_match: 0/1" in report
    assert "targets:" in report
    assert "accept_positive_recovery:" in report


@pytest.mark.unit
def test_checkpoint_pre_collapse_cli_fail_on_hold_respects_thresholds(
    tmp_path,
    monkeypatch,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _pre_collapse_fixture(trajectory)
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.STAY_COURSE)

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_strategy_checkpoint_pre_collapse_recovery.py",
            str(trajectory),
            "--checkpoint",
            str(checkpoint),
            "--fail-on-hold",
        ],
    )
    assert checkpoint_pre_collapse_main() == 1

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_strategy_checkpoint_pre_collapse_recovery.py",
            str(trajectory),
            "--checkpoint",
            str(checkpoint),
            "--fail-on-hold",
            "--max-missed-checkpoint-pre-collapse-recovery-rows",
            "1",
            "--max-missed-checkpoint-pre-collapse-recovery-rate",
            "1.0",
            "--max-missed-accept-positive-recovery-rows",
            "1",
            "--max-missed-accept-positive-recovery-rate",
            "1.0",
        ],
    )
    assert checkpoint_pre_collapse_main() == 0
