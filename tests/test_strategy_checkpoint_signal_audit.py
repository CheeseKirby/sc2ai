from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch

from bot.managers.rl_strategy_policy import RLStrategyPolicy as _RLStrategyPolicy
from rl.models import PolicyModelSpec, build_policy_model
from rl.strategy_action_critic import (
    ACTION_CRITIC_FEATURE_FIELDS,
    ActionCriticModelSpec,
    StrategyActionCriticNetwork,
    save_strategy_action_critic_checkpoint,
)
from rl.strategy_actions import StrategyAction
from rl.strategy_checkpoint_signal_audit import audit_strategy_checkpoint_signals
from rl.strategy_checkpoints import save_strategy_policy_checkpoint
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.audit_strategy_checkpoint_signals import (
    format_strategy_checkpoint_signal_audit,
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


def _strategy_checkpoint_with_biases(tmp_path, biases: dict[StrategyAction, float]):
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
        final = model.net[-1]
        for action, bias in biases.items():
            final.bias[int(action)] = bias
    checkpoint = tmp_path / "biased.pt"
    save_strategy_policy_checkpoint(checkpoint, model)
    return checkpoint


def _action_critic_checkpoint(
    tmp_path,
    *,
    unsafe_action: StrategyAction,
    safe_bias: float = -10.0,
    unsafe_weight: float = 20.0,
):
    model = StrategyActionCriticNetwork(
        ActionCriticModelSpec(
            feature_dim=len(ACTION_CRITIC_FEATURE_FIELDS),
            hidden_sizes=(),
        )
    )
    with torch.no_grad():
        final = model.net[-1]
        final.weight.zero_()
        final.bias.fill_(safe_bias)
        action_index = ACTION_CRITIC_FEATURE_FIELDS.index(f"action:{unsafe_action.name}")
        final.weight[0, action_index] = unsafe_weight
    checkpoint = tmp_path / "action_critic.pt"
    save_strategy_action_critic_checkpoint(checkpoint, model)
    return checkpoint


def _action_critic_checkpoint_with_action_logits(
    tmp_path,
    *,
    action_logits: dict[StrategyAction, float],
    default_logit: float = 2.0,
):
    model = StrategyActionCriticNetwork(
        ActionCriticModelSpec(
            feature_dim=len(ACTION_CRITIC_FEATURE_FIELDS),
            hidden_sizes=(),
        )
    )
    with torch.no_grad():
        final = model.net[-1]
        final.weight.zero_()
        final.bias.fill_(default_logit)
        for action, logit in action_logits.items():
            action_index = ACTION_CRITIC_FEATURE_FIELDS.index(f"action:{action.name}")
            final.weight[0, action_index] = logit - default_logit
    checkpoint = tmp_path / "action_critic_logits.pt"
    save_strategy_action_critic_checkpoint(checkpoint, model)
    return checkpoint


@pytest.mark.unit
def test_checkpoint_signal_audit_counts_predictions_and_executability(tmp_path) -> None:
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.ADD_GATEWAYS)
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.ADD_GATEWAYS),
                strategy_action_name="ADD_GATEWAYS",
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=130.0,
                    ready_gateways=1.0,
                    pending_gateways=1.0,
                    minerals=200.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=220.0,
                    ready_gateways=4.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
            ),
            _row(done=True, result="Result.Victory"),
        ],
    )

    audit = audit_strategy_checkpoint_signals(trajectory, checkpoint)

    assert audit.rows == 3
    assert audit.predicted_action_counts_by_name == {"ADD_GATEWAYS": 3}
    assert audit.recorded_training_use_counts["accept_positive"] == 1
    assert audit.predicted_non_executable_rows == 1
    assert audit.predicted_non_executable_ratio == pytest.approx(1 / 3)
    assert audit.predicted_blocker_counts == {"target_gateways_reached": 1}
    assert audit.accept_positive_rows == 1
    assert audit.accept_positive_prediction_matches == 1
    assert audit.accept_positive_prediction_match_ratio == 1.0
    assert audit.predicted_non_executable_by_recorded_training_use == {
        "drop_ambiguous": 1
    }
    assert "predicted_non_executable_ratio_high" in audit.blocking_reasons
    assert asdict(audit)["decisions"][0]["predicted_action"] == "ADD_GATEWAYS"


@pytest.mark.unit
def test_checkpoint_signal_audit_can_apply_executable_mask(tmp_path) -> None:
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.ADD_GATEWAYS)
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_gateways=4.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
            ),
        ],
    )

    audit = audit_strategy_checkpoint_signals(
        trajectory,
        checkpoint,
        prediction_mode="executable-mask",
    )

    assert audit.prediction_mode == "executable-mask"
    assert audit.raw_predicted_action_counts_by_name == {"ADD_GATEWAYS": 1}
    assert audit.predicted_action_counts_by_name == {"STAY_COURSE": 1}
    assert audit.predicted_non_executable_rows == 0
    assert audit.raw_predicted_non_executable_rows == 1
    assert audit.masked_prediction_changes == 1
    assert audit.decisions[0].raw_predicted_action == "ADD_GATEWAYS"
    assert audit.decisions[0].predicted_action == "STAY_COURSE"
    assert audit.decisions[0].prediction_was_masked is True


@pytest.mark.unit
def test_checkpoint_signal_audit_can_apply_signal_risk_mask(tmp_path) -> None:
    checkpoint = _strategy_checkpoint_with_biases(
        tmp_path,
        {
            StrategyAction.PRODUCE_ARMY: 10.0,
            StrategyAction.STAY_COURSE: 9.0,
        },
    )
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.PRODUCE_ARMY),
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=10.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    audit = audit_strategy_checkpoint_signals(
        trajectory,
        checkpoint,
        prediction_mode="signal-risk-mask",
        critic_min_samples=1,
    )

    assert audit.prediction_mode == "signal-risk-mask"
    assert audit.raw_predicted_action_counts_by_name == {"PRODUCE_ARMY": 2}
    assert audit.predicted_action_counts_by_name == {"ADD_GATEWAYS": 2}
    assert audit.veto_negative_prediction_matches == 0
    assert audit.critic_vetoed_candidates >= 2
    assert audit.critic_veto_reason_counts == {
        "risk:action=PRODUCE_ARMY,context=other,threat=ground_threat": 2,
        "risk:action=STAY_COURSE,context=other,threat=ground_threat": 2,
    }
    assert audit.decisions[0].raw_predicted_action == "PRODUCE_ARMY"
    assert audit.decisions[0].predicted_action == "ADD_GATEWAYS"
    assert audit.decisions[0].prediction_was_masked is True
    assert audit.decisions[0].critic_vetoed_actions == [
        "PRODUCE_ARMY",
        "STAY_COURSE",
    ]


@pytest.mark.unit
def test_checkpoint_signal_audit_can_apply_action_critic_mask(tmp_path) -> None:
    checkpoint = _strategy_checkpoint_with_biases(
        tmp_path,
        {
            StrategyAction.PRODUCE_ARMY: 10.0,
            StrategyAction.STAY_COURSE: 9.0,
        },
    )
    critic_checkpoint = _action_critic_checkpoint(
        tmp_path,
        unsafe_action=StrategyAction.PRODUCE_ARMY,
    )
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.PRODUCE_ARMY),
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=10.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=0.0,
                    base_under_ground_threat=0.0,
                    army_count=6.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    audit = audit_strategy_checkpoint_signals(
        trajectory,
        checkpoint,
        prediction_mode="action-critic-mask",
        action_critic_checkpoint_path=critic_checkpoint,
        action_critic_threshold=0.5,
    )

    assert audit.prediction_mode == "action-critic-mask"
    assert audit.action_critic_checkpoint_path == str(critic_checkpoint)
    assert audit.action_critic_threshold == 0.5
    assert audit.raw_predicted_action_counts_by_name == {"PRODUCE_ARMY": 2}
    assert audit.predicted_action_counts_by_name == {"STAY_COURSE": 2}
    assert audit.veto_negative_prediction_matches == 0
    assert audit.critic_vetoed_candidates == 2
    assert audit.critic_veto_reason_counts == {
        "action_critic_unsafe_probability_high": 2
    }
    assert audit.critic_veto_action_counts == {"PRODUCE_ARMY": 2}
    assert audit.action_critic_vetoed_probability_max is not None
    assert audit.action_critic_vetoed_probability_max > 0.99
    assert audit.action_critic_selected_unsafe_probability_max is not None
    assert audit.action_critic_selected_unsafe_probability_max < 0.01
    assert audit.action_critic_fallback_rows == 0
    assert audit.decisions[0].raw_predicted_action == "PRODUCE_ARMY"
    assert audit.decisions[0].predicted_action == "STAY_COURSE"
    assert audit.decisions[0].prediction_was_masked is True
    assert audit.decisions[0].critic_vetoed_actions == ["PRODUCE_ARMY"]
    assert audit.decisions[0].action_critic_candidate_actions == [
        "PRODUCE_ARMY",
        "STAY_COURSE",
    ]
    assert len(audit.decisions[0].action_critic_candidate_unsafe_probabilities) == 2
    assert audit.decisions[0].action_critic_candidate_unsafe_probabilities[0] > 0.99
    assert audit.decisions[0].action_critic_candidate_unsafe_probabilities[1] < 0.01
    assert audit.decisions[0].action_critic_vetoed_probabilities[0] > 0.99
    assert audit.decisions[0].action_critic_selected_unsafe_probability < 0.01
    assert audit.decisions[0].action_critic_fallback_selected is False
    assert audit.decisions[0].action_critic_fallback_policy_used is None


@pytest.mark.unit
def test_action_critic_safe_stay_course_fallback_does_not_block_bad_row(
    tmp_path,
) -> None:
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.STAY_COURSE)
    critic_checkpoint = _action_critic_checkpoint(
        tmp_path,
        unsafe_action=StrategyAction.STAY_COURSE,
        safe_bias=10.0,
        unsafe_weight=0.0,
    )
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.BUILD_STATIC_DEFENSE),
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    has_cybernetics_core=0.0,
                    ready_forge=0.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    minerals=200.0,
                ),
            ),
            _row(done=True, result="Result.Tie"),
        ],
    )

    audit = audit_strategy_checkpoint_signals(
        trajectory,
        checkpoint,
        prediction_mode="action-critic-mask",
        action_critic_checkpoint_path=critic_checkpoint,
        action_critic_threshold=0.5,
        action_critic_fallback_policy="first-executable",
    )

    assert audit.signal_healthy is True
    assert audit.blocking_reasons == []
    assert audit.action_critic_fallback_rows == 1
    assert audit.action_critic_safe_fallback_rows == 1
    assert audit.action_critic_unsafe_fallback_rows == 0
    assert audit.decisions[0].recorded_training_use == "drop_non_executable"
    assert audit.decisions[0].predicted_action == "STAY_COURSE"
    assert audit.decisions[0].action_critic_fallback_selected is True


@pytest.mark.unit
def test_action_critic_stay_course_fallback_blocks_accept_positive_row(
    tmp_path,
) -> None:
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.STAY_COURSE)
    critic_checkpoint = _action_critic_checkpoint(
        tmp_path,
        unsafe_action=StrategyAction.STAY_COURSE,
        safe_bias=10.0,
        unsafe_weight=0.0,
    )
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.ADD_GATEWAYS),
                strategy_action_name="ADD_GATEWAYS",
                strategy_observation=_observation(
                    game_time=100.0,
                    own_bases=2.0,
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=130.0,
                    own_bases=2.0,
                    ready_gateways=1.0,
                    pending_gateways=1.0,
                    minerals=200.0,
                ),
            ),
            _row(done=True, result="Result.Victory"),
        ],
    )

    audit = audit_strategy_checkpoint_signals(
        trajectory,
        checkpoint,
        prediction_mode="action-critic-mask",
        action_critic_checkpoint_path=critic_checkpoint,
        action_critic_threshold=0.5,
        action_critic_fallback_policy="first-executable",
    )

    assert audit.signal_healthy is False
    assert "action_critic_all_executable_candidates_vetoed" in audit.blocking_reasons
    assert audit.action_critic_fallback_rows >= 1
    assert audit.action_critic_safe_fallback_rows == 0
    assert audit.action_critic_unsafe_fallback_rows >= 1
    assert audit.decisions[0].recorded_training_use == "accept_positive"
    assert audit.decisions[0].predicted_action == "STAY_COURSE"


@pytest.mark.unit
def test_action_critic_mask_can_use_signal_risk_fallback(tmp_path) -> None:
    checkpoint = _strategy_checkpoint_with_biases(
        tmp_path,
        {
            StrategyAction.PRODUCE_ARMY: 10.0,
            StrategyAction.STAY_COURSE: 9.0,
        },
    )
    critic_checkpoint = _action_critic_checkpoint(
        tmp_path,
        unsafe_action=StrategyAction.PRODUCE_ARMY,
        safe_bias=10.0,
        unsafe_weight=0.0,
    )
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.PRODUCE_ARMY),
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=10.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    audit = audit_strategy_checkpoint_signals(
        trajectory,
        checkpoint,
        prediction_mode="action-critic-mask",
        action_critic_checkpoint_path=critic_checkpoint,
        action_critic_threshold=0.5,
        action_critic_fallback_policy="signal-risk",
        critic_min_samples=1,
    )

    assert audit.action_critic_fallback_policy == "signal-risk"
    assert audit.predicted_action_counts_by_name == {"ADD_GATEWAYS": 2}
    assert audit.veto_negative_prediction_matches == 0
    assert audit.action_critic_fallback_rows == 2
    assert audit.action_critic_fallback_policy_counts == {"signal-risk": 2}
    assert audit.decisions[0].action_critic_fallback_selected is True
    assert audit.decisions[0].action_critic_fallback_policy_used == "signal-risk"
    assert audit.decisions[0].action_critic_selected_unsafe_probability > 0.99


@pytest.mark.unit
def test_action_critic_mask_can_use_rule_safe_fallback_under_threat(tmp_path) -> None:
    checkpoint = _strategy_checkpoint_with_biases(
        tmp_path,
        {
            StrategyAction.STAY_COURSE: 10.0,
            StrategyAction.BUILD_STATIC_DEFENSE: 9.0,
            StrategyAction.PRODUCE_ARMY: 8.0,
        },
    )
    critic_checkpoint = _action_critic_checkpoint(
        tmp_path,
        unsafe_action=StrategyAction.STAY_COURSE,
        safe_bias=10.0,
        unsafe_weight=0.0,
    )
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.STAY_COURSE),
                strategy_action_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    minerals=200.0,
                    has_cybernetics_core=1.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    army_count=8.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=180.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    audit = audit_strategy_checkpoint_signals(
        trajectory,
        checkpoint,
        prediction_mode="action-critic-mask",
        action_critic_checkpoint_path=critic_checkpoint,
        action_critic_threshold=0.5,
        action_critic_fallback_policy="rule-safe",
    )

    assert audit.action_critic_fallback_policy == "rule-safe"
    assert audit.predicted_action_counts_by_name == {"BUILD_STATIC_DEFENSE": 1}
    assert audit.veto_negative_rows == 1
    assert audit.veto_negative_prediction_matches == 0
    assert audit.action_critic_fallback_rows == 1
    assert audit.action_critic_fallback_policy_counts == {"rule-safe": 1}
    assert audit.decisions[0].raw_predicted_action == "STAY_COURSE"
    assert audit.decisions[0].predicted_action == "BUILD_STATIC_DEFENSE"
    assert audit.decisions[0].prediction_was_masked is True
    assert audit.decisions[0].action_critic_fallback_selected is True
    assert audit.decisions[0].action_critic_fallback_policy_used == "rule-safe"


@pytest.mark.unit
def test_action_critic_mask_rule_risk_fallback_skips_signal_risk_veto(tmp_path) -> None:
    checkpoint = _strategy_checkpoint_with_biases(
        tmp_path,
        {
            StrategyAction.STAY_COURSE: 10.0,
            StrategyAction.BUILD_STATIC_DEFENSE: 9.0,
            StrategyAction.PRODUCE_ARMY: 8.0,
        },
    )
    critic_checkpoint = _action_critic_checkpoint(
        tmp_path,
        unsafe_action=StrategyAction.STAY_COURSE,
        safe_bias=10.0,
        unsafe_weight=0.0,
    )
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.BUILD_STATIC_DEFENSE),
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    minerals=200.0,
                    has_cybernetics_core=1.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    army_count=8.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=180.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    audit = audit_strategy_checkpoint_signals(
        trajectory,
        checkpoint,
        prediction_mode="action-critic-mask",
        action_critic_checkpoint_path=critic_checkpoint,
        action_critic_threshold=0.5,
        action_critic_fallback_policy="rule-risk",
        critic_min_samples=1,
    )

    assert audit.action_critic_fallback_policy == "rule-risk"
    assert audit.predicted_action_counts_by_name == {"PRODUCE_ARMY": 1}
    assert audit.veto_negative_rows == 1
    assert audit.veto_negative_prediction_matches == 0
    assert audit.action_critic_fallback_rows == 1
    assert audit.action_critic_fallback_policy_counts == {"rule-risk": 1}
    assert audit.decisions[0].predicted_action == "PRODUCE_ARMY"
    assert audit.decisions[0].action_critic_fallback_policy_used == "rule-risk"


@pytest.mark.unit
def test_action_critic_mask_threat_risk_fallback_avoids_stay_under_threat(tmp_path) -> None:
    checkpoint = _strategy_checkpoint_with_biases(
        tmp_path,
        {
            StrategyAction.STAY_COURSE: 10.0,
            StrategyAction.BUILD_STATIC_DEFENSE: 9.0,
            StrategyAction.PRODUCE_ARMY: 8.0,
        },
    )
    critic_checkpoint = _action_critic_checkpoint_with_action_logits(
        tmp_path,
        action_logits={
            StrategyAction.STAY_COURSE: 0.5,
            StrategyAction.BUILD_STATIC_DEFENSE: 0.75,
            StrategyAction.PRODUCE_ARMY: 1.0,
        },
    )
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.STAY_COURSE),
                strategy_action_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    minerals=200.0,
                    has_cybernetics_core=1.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    army_count=8.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=180.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    audit = audit_strategy_checkpoint_signals(
        trajectory,
        checkpoint,
        prediction_mode="action-critic-mask",
        action_critic_checkpoint_path=critic_checkpoint,
        action_critic_threshold=0.4,
        action_critic_fallback_policy="threat-risk",
        critic_min_samples=1,
    )

    assert audit.action_critic_fallback_policy == "threat-risk"
    assert audit.predicted_action_counts_by_name == {"BUILD_STATIC_DEFENSE": 1}
    assert audit.veto_negative_rows == 1
    assert audit.veto_negative_prediction_matches == 0
    assert audit.action_critic_fallback_rows == 1
    assert audit.action_critic_fallback_policy_counts == {"threat-risk": 1}
    assert audit.decisions[0].raw_predicted_action == "STAY_COURSE"
    assert audit.decisions[0].predicted_action == "BUILD_STATIC_DEFENSE"
    assert audit.decisions[0].action_critic_fallback_policy_used == "threat-risk"


@pytest.mark.unit
def test_action_critic_mask_mixed_threat_fallback_avoids_stay_only_for_mixed_threat(
    tmp_path,
) -> None:
    checkpoint = _strategy_checkpoint_with_biases(
        tmp_path,
        {
            StrategyAction.STAY_COURSE: 10.0,
            StrategyAction.BUILD_STATIC_DEFENSE: 9.0,
            StrategyAction.PRODUCE_ARMY: 8.0,
        },
    )
    critic_checkpoint = _action_critic_checkpoint_with_action_logits(
        tmp_path,
        action_logits={
            StrategyAction.STAY_COURSE: 0.5,
            StrategyAction.BUILD_STATIC_DEFENSE: 0.75,
            StrategyAction.PRODUCE_ARMY: 1.0,
        },
    )
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.STAY_COURSE),
                strategy_action_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    minerals=200.0,
                    has_cybernetics_core=1.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    army_count=8.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    minerals=200.0,
                    has_cybernetics_core=1.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    army_count=8.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=180.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    audit = audit_strategy_checkpoint_signals(
        trajectory,
        checkpoint,
        prediction_mode="action-critic-mask",
        action_critic_checkpoint_path=critic_checkpoint,
        action_critic_threshold=0.4,
        action_critic_fallback_policy="mixed-threat-risk",
    )

    assert audit.action_critic_fallback_policy == "mixed-threat-risk"
    assert audit.predicted_action_counts_by_name == {
        "BUILD_STATIC_DEFENSE": 1,
        "STAY_COURSE": 1,
    }
    assert audit.veto_negative_rows == 2
    assert audit.veto_negative_prediction_matches == 1
    assert audit.action_critic_fallback_rows == 2
    assert audit.action_critic_fallback_policy_counts == {"mixed-threat-risk": 2}
    assert audit.decisions[0].threat_state == "air_and_ground_threat"
    assert audit.decisions[0].predicted_action == "BUILD_STATIC_DEFENSE"
    assert audit.decisions[0].action_critic_fallback_policy_used == "mixed-threat-risk"
    assert audit.decisions[0].action_critic_candidate_actions[:3] == [
        "STAY_COURSE",
        "BUILD_STATIC_DEFENSE",
        "PRODUCE_ARMY",
    ]
    assert "ADD_GATEWAYS" in audit.decisions[0].action_critic_candidate_actions
    assert audit.decisions[1].threat_state == "ground_threat"
    assert audit.decisions[1].predicted_action == "STAY_COURSE"


@pytest.mark.unit
def test_action_critic_mask_requires_checkpoint(tmp_path) -> None:
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.STAY_COURSE)
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(trajectory, [_row()])

    with pytest.raises(ValueError, match="action_critic_checkpoint_path"):
        audit_strategy_checkpoint_signals(
            trajectory,
            checkpoint,
            prediction_mode="action-critic-mask",
        )


@pytest.mark.unit
def test_checkpoint_signal_audit_flags_repeated_veto_negative_labels(tmp_path) -> None:
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.PRODUCE_ARMY)
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.PRODUCE_ARMY),
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=10.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                    supply_left=8.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    audit = audit_strategy_checkpoint_signals(trajectory, checkpoint)

    assert audit.veto_negative_rows == 2
    assert audit.veto_negative_prediction_matches == 1
    assert audit.bad_recorded_rows == 2
    assert audit.bad_recorded_prediction_matches == 1
    assert audit.predicted_action_counts_by_name == {"PRODUCE_ARMY": 2}
    assert "predicted_matches_veto_negative_labels" in audit.blocking_reasons


@pytest.mark.unit
def test_checkpoint_signal_audit_flags_action_space_exhausted_matches(tmp_path) -> None:
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.STAY_COURSE)
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.STAY_COURSE),
                strategy_action_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=25.0,
                    vespene=0.0,
                    supply_left=0.0,
                    own_bases=1.0,
                    ready_gateways=4.0,
                    pending_gateways=0.0,
                    has_cybernetics_core=0.0,
                    ready_forge=0.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    audit = audit_strategy_checkpoint_signals(trajectory, checkpoint)

    assert audit.action_space_exhausted_rows == 1
    assert audit.action_space_exhausted_prediction_matches == 1
    assert audit.veto_negative_rows == 0
    assert audit.decisions[0].recorded_training_use == "action_space_exhausted"
    assert audit.decisions[0].prediction_matches_action_space_exhausted is True
    assert "predicted_matches_action_space_exhausted_labels" in audit.blocking_reasons


@pytest.mark.unit
def test_format_checkpoint_signal_audit_includes_decision_lines(tmp_path) -> None:
    checkpoint = _constant_strategy_checkpoint(tmp_path, StrategyAction.ADD_GATEWAYS)
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_gateways=4.0,
                    minerals=200.0,
                ),
            ),
        ],
    )

    report = format_strategy_checkpoint_signal_audit(
        audit_strategy_checkpoint_signals(trajectory, checkpoint),
        show_decisions=True,
    )

    assert "Strategy checkpoint signal audit" in report
    assert "predicted_non_executable:" in report
    assert "raw_predicted_non_executable:" in report
    assert "masked_prediction_changes:" in report
    assert "critic_vetoed_candidates:" in report
    assert "action_space_exhausted_match:" in report
    assert "action_critic_fallback_rows:" in report
    assert "action_critic_safe_fallback_rows:" in report
    assert "action_critic_unsafe_fallback_rows:" in report
    assert "action_critic_fallback_policy:" in report
    assert "action_critic_fallback_policies:" in report
    assert "predicted_action_counts:" in report
    assert "raw_predicted_action_counts:" in report
    assert "decisions:" in report
    assert "predicted=ADD_GATEWAYS" in report
    assert "raw=ADD_GATEWAYS" in report
    assert "blocker=target_gateways_reached" in report
    assert "action_critic_candidates=" in report
    assert "action_critic_candidate_unsafe=" in report
