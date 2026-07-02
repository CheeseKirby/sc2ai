from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl.strategy_action_critic_threshold_sweep import (
    action_critic_threshold_trial_from_audit,
    sweep_strategy_action_critic_thresholds,
)
from scripts.sweep_strategy_action_critic_thresholds import (
    format_strategy_action_critic_threshold_sweep,
)


def _audit(**overrides):
    payload = {
        "inputs": ["data/example"],
        "rows": 10,
        "signal_healthy": False,
        "blocking_reasons": ["action_critic_all_executable_candidates_vetoed"],
        "warnings": [],
        "prediction_matches_recorded": 6,
        "prediction_match_ratio": 0.6,
        "accept_positive_prediction_matches": 3,
        "accept_positive_rows": 4,
        "veto_negative_prediction_matches": 0,
        "veto_negative_rows": 2,
        "drop_non_executable_prediction_matches": 0,
        "drop_non_executable_rows": 1,
        "predicted_non_executable_rows": 0,
        "predicted_non_executable_ratio": 0.0,
        "action_critic_fallback_rows": 5,
        "action_critic_unsafe_fallback_rows": 5,
        "action_critic_selected_unsafe_probability_avg": 0.41,
        "action_critic_selected_unsafe_probability_max": 0.72,
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_action_critic_threshold_trial_from_audit_ranks_hard_safety_first() -> None:
    safer = action_critic_threshold_trial_from_audit(
        _audit(
            action_critic_fallback_rows=10,
            action_critic_unsafe_fallback_rows=10,
        ),
        threshold=0.4,
        fallback_policy="lowest-risk",
    )
    worse = action_critic_threshold_trial_from_audit(
        _audit(
            signal_healthy=True,
            blocking_reasons=[],
            veto_negative_prediction_matches=1,
            action_critic_fallback_rows=0,
        ),
        threshold=0.8,
        fallback_policy="lowest-risk",
    )

    assert safer.rank_score < worse.rank_score


@pytest.mark.unit
def test_action_critic_threshold_trial_treats_safe_fallback_as_non_blocking() -> None:
    safe = action_critic_threshold_trial_from_audit(
        _audit(
            signal_healthy=True,
            blocking_reasons=[],
            action_critic_fallback_rows=10,
            action_critic_unsafe_fallback_rows=0,
        ),
        threshold=0.4,
        fallback_policy="first-executable",
    )
    unsafe = action_critic_threshold_trial_from_audit(
        _audit(action_critic_fallback_rows=1, action_critic_unsafe_fallback_rows=1),
        threshold=0.4,
        fallback_policy="first-executable",
    )

    assert safe.signal_healthy is True
    assert safe.action_critic_fallback_rows == 10
    assert safe.action_critic_unsafe_fallback_rows == 0
    assert safe.rank_score < unsafe.rank_score


@pytest.mark.unit
def test_sweep_strategy_action_critic_thresholds_selects_best_trial(monkeypatch) -> None:
    def fake_audit(*args, **kwargs):
        threshold = kwargs["action_critic_threshold"]
        if threshold == 0.4:
            return SimpleNamespace(**_audit(action_critic_fallback_rows=5))
        return SimpleNamespace(
            **_audit(
                signal_healthy=True,
                blocking_reasons=[],
                action_critic_fallback_rows=0,
                prediction_match_ratio=0.5,
            )
        )

    monkeypatch.setattr(
        "rl.strategy_action_critic_threshold_sweep.audit_strategy_checkpoint_signals",
        fake_audit,
    )

    sweep = sweep_strategy_action_critic_thresholds(
        "data/example",
        "policy.pt",
        "critic.pt",
        thresholds=[0.4, 0.8],
        fallback_policies=["lowest-risk"],
    )

    assert sweep.recommendation == "promotion_candidate"
    assert sweep.selected_trial is not None
    assert sweep.selected_trial.threshold == 0.8
    assert sweep.selected_trial.action_critic_fallback_rows == 0


@pytest.mark.unit
def test_sweep_strategy_action_critic_thresholds_validates_inputs() -> None:
    with pytest.raises(ValueError, match="at least one threshold"):
        sweep_strategy_action_critic_thresholds(
            "data/example",
            "policy.pt",
            "critic.pt",
            thresholds=[],
        )

    with pytest.raises(ValueError, match="unknown fallback"):
        sweep_strategy_action_critic_thresholds(
            "data/example",
            "policy.pt",
            "critic.pt",
            thresholds=[0.4],
            fallback_policies=["missing"],
        )


@pytest.mark.unit
def test_format_strategy_action_critic_threshold_sweep_includes_trials(monkeypatch) -> None:
    def fake_audit(*args, **kwargs):
        return SimpleNamespace(**_audit())

    monkeypatch.setattr(
        "rl.strategy_action_critic_threshold_sweep.audit_strategy_checkpoint_signals",
        fake_audit,
    )

    report = format_strategy_action_critic_threshold_sweep(
        sweep_strategy_action_critic_thresholds(
            "data/example",
            "policy.pt",
            "critic.pt",
            thresholds=[0.4],
        )
    )

    assert "Strategy action critic threshold sweep" in report
    assert "threshold=0.400" in report
    assert "fallback_rows=5" in report
    assert "unsafe_fallback_rows=5" in report
    assert "veto=0/2" in report
