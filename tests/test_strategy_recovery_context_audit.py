from __future__ import annotations

import json

import pytest
import torch

from bot.managers.rl_strategy_policy import RLStrategyPolicy as _RLStrategyPolicy
from rl.models import PolicyModelSpec, build_policy_model
from rl.strategy_actions import StrategyAction
from rl.strategy_checkpoints import save_strategy_policy_checkpoint
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from rl.strategy_recovery_context_audit import audit_strategy_recovery_context
from scripts.audit_strategy_recovery_context import (
    format_strategy_recovery_context_audit,
    main as recovery_context_main,
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
            "ready_gateways": 2.0,
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


def _recovery_context_fixture(path) -> None:
    _write_jsonl(
        path,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.TECH_ROBO),
                strategy_action_name="TECH_ROBO",
                strategy_observation=_observation(
                    game_time=120.0,
                    minerals=200.0,
                    vespene=150.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                ),
            ),
            _row(
                step=96,
                strategy_observation=_observation(
                    game_time=136.0,
                    pending_robo=1.0,
                ),
            ),
            _row(
                step=128,
                strategy_action=int(StrategyAction.TECH_ROBO),
                strategy_action_name="TECH_ROBO",
                strategy_observation=_observation(
                    game_time=560.0,
                    minerals=200.0,
                    vespene=600.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    base_under_threat=0.0,
                ),
            ),
            _row(
                step=160,
                strategy_observation=_observation(
                    game_time=576.0,
                    pending_robo=1.0,
                ),
            ),
            _row(
                step=192,
                strategy_action=int(StrategyAction.BUILD_STATIC_DEFENSE),
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=620.0,
                    minerals=150.0,
                    ready_forge=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    base_under_threat=0.0,
                ),
            ),
            _row(
                step=224,
                strategy_observation=_observation(
                    game_time=636.0,
                    ready_static_defense=1.0,
                ),
            ),
            _row(
                done=True,
                result="Result.Tie",
                strategy_observation=_observation(game_time=700.0),
            ),
        ],
    )


@pytest.mark.unit
def test_recovery_context_audit_reports_context_confusion(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _recovery_context_fixture(trajectory)
    checkpoint = _constant_strategy_checkpoint(
        tmp_path,
        StrategyAction.BUILD_STATIC_DEFENSE,
    )

    audit = audit_strategy_recovery_context(
        trajectory,
        checkpoint,
        prediction_mode="executable-mask",
    )

    assert audit.recommendation == "hold"
    assert audit.blocking_reasons == [
        "context_missed_accept_positive_recovery_rows",
        "context_missed_accept_positive_recovery_rate",
        "context_cross_action_confusion_rows",
        "context_cross_action_confusion_rate",
    ]
    assert audit.accept_positive_recovery_rows == 3
    assert audit.accept_positive_recovery_matches == 1
    assert audit.context_matched_accept_positive_recovery_rows == 2
    assert audit.context_matched_accept_positive_recovery_matches == 1
    assert audit.context_missed_accept_positive_recovery_rows == 1
    assert audit.context_skipped_accept_positive_recovery_rows == 1
    assert audit.cross_action_confusion_rows == 2
    assert audit.context_matched_cross_action_confusion_rows == 1
    assert audit.recorded_counts_by_action == {
        "BUILD_STATIC_DEFENSE": 1,
        "TECH_ROBO": 2,
    }
    assert audit.predicted_counts_by_action == {"BUILD_STATIC_DEFENSE": 3}
    assert audit.context_matched_recorded_counts_by_action == {
        "BUILD_STATIC_DEFENSE": 1,
        "TECH_ROBO": 1,
    }
    assert audit.context_matched_confusion_counts_by_recorded_then_predicted == {
        "BUILD_STATIC_DEFENSE": {"BUILD_STATIC_DEFENSE": 1},
        "TECH_ROBO": {"BUILD_STATIC_DEFENSE": 1},
    }

    tech_summary = next(
        summary
        for summary in audit.action_summaries
        if summary.recorded_action == "TECH_ROBO"
    )
    assert tech_summary.rows == 2
    assert tech_summary.context_matched_rows == 1
    assert tech_summary.context_matched_matches == 0
    assert tech_summary.context_matched_cross_action_confusion_rows == 1


@pytest.mark.unit
def test_format_recovery_context_audit_includes_confusion(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _recovery_context_fixture(trajectory)
    checkpoint = _constant_strategy_checkpoint(
        tmp_path,
        StrategyAction.BUILD_STATIC_DEFENSE,
    )

    report = format_strategy_recovery_context_audit(
        audit_strategy_recovery_context(trajectory, checkpoint)
    )

    assert "Strategy recovery context audit" in report
    assert "recommendation: hold" in report
    assert "context_matched_accept_positive_recovery_match: 1/2" in report
    assert "context_matched_cross_action_confusion_rows: 1/2" in report
    assert "context_matched_confusion:" in report
    assert "TECH_ROBO: BUILD_STATIC_DEFENSE=1" in report


@pytest.mark.unit
def test_recovery_context_audit_validates_arguments(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _recovery_context_fixture(trajectory)
    checkpoint = _constant_strategy_checkpoint(
        tmp_path,
        StrategyAction.BUILD_STATIC_DEFENSE,
    )

    with pytest.raises(ValueError, match="context_filter"):
        audit_strategy_recovery_context(
            trajectory,
            checkpoint,
            context_filter="made-up",
        )
    with pytest.raises(ValueError, match="max_decisions"):
        audit_strategy_recovery_context(
            trajectory,
            checkpoint,
            max_decisions=-1,
        )
    with pytest.raises(ValueError, match="max_context_missed_rate"):
        audit_strategy_recovery_context(
            trajectory,
            checkpoint,
            max_context_missed_rate=1.1,
        )


@pytest.mark.unit
def test_recovery_context_cli_fail_on_hold_respects_thresholds(
    tmp_path,
    monkeypatch,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _recovery_context_fixture(trajectory)
    checkpoint = _constant_strategy_checkpoint(
        tmp_path,
        StrategyAction.BUILD_STATIC_DEFENSE,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_strategy_recovery_context.py",
            str(trajectory),
            "--checkpoint",
            str(checkpoint),
            "--fail-on-hold",
        ],
    )
    assert recovery_context_main() == 1

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_strategy_recovery_context.py",
            str(trajectory),
            "--checkpoint",
            str(checkpoint),
            "--fail-on-hold",
            "--max-context-missed-rows",
            "1",
            "--max-context-missed-rate",
            "1.0",
            "--max-context-cross-action-rows",
            "1",
            "--max-context-cross-action-rate",
            "1.0",
        ],
    )
    assert recovery_context_main() == 0
