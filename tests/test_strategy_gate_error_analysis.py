from __future__ import annotations

from dataclasses import asdict

import pytest

from rl.experiments import write_json
from rl.strategy_gate_error_analysis import analyze_strategy_gate_errors
from scripts.analyze_strategy_gate_errors import format_strategy_gate_error_analysis


def _decision(**overrides) -> dict:
    payload = {
        "path": "data/trajectory/example.jsonl",
        "step": 64,
        "game_time": 120.0,
        "recorded_action": "STAY_COURSE",
        "raw_predicted_action": "EXPAND",
        "predicted_action": "STAY_COURSE",
        "recorded_training_use": "drop_ambiguous",
        "recorded_label_quality": "unknown",
        "context": "other",
        "threat_state": "no_threat",
        "prediction_matches_accept_positive": False,
        "prediction_matches_veto_negative": False,
        "prediction_matches_drop_non_executable": False,
        "prediction_matches_action_space_exhausted": False,
        "predicted_immediate_executable": True,
        "predicted_blocker": None,
        "action_critic_fallback_selected": False,
        "action_critic_candidate_actions": [],
        "action_critic_selected_unsafe_probability": None,
        "critic_vetoed_actions": [],
        "critic_veto_reasons": [],
    }
    payload.update(overrides)
    return payload


def _audit(**overrides) -> dict:
    payload = {
        "checkpoint_path": "runs/example/checkpoints/policy.pt",
        "warnings": [],
        "decisions": [
            _decision(),
            _decision(
                step=128,
                recorded_training_use="veto_negative",
                recorded_label_quality="bad",
                prediction_matches_veto_negative=True,
                action_critic_fallback_selected=True,
                action_critic_candidate_actions=["STAY_COURSE"],
                action_critic_selected_unsafe_probability=0.91,
                critic_vetoed_actions=["STAY_COURSE"],
                critic_veto_reasons=["action_critic_unsafe_probability_high"],
                threat_state="air_and_ground_threat",
            ),
            _decision(
                step=192,
                predicted_action="EXPAND",
                recorded_training_use="accept_positive",
                recorded_label_quality="good",
                prediction_matches_accept_positive=True,
                action_critic_fallback_selected=True,
                action_critic_candidate_actions=["EXPAND", "STAY_COURSE"],
                action_critic_selected_unsafe_probability=0.63,
                critic_vetoed_actions=["EXPAND"],
                critic_veto_reasons=["action_critic_unsafe_probability_high"],
            ),
            _decision(
                step=256,
                predicted_action="TECH_ROBO",
                predicted_immediate_executable=False,
                predicted_blocker="cannot_afford_robo",
            ),
            _decision(
                step=320,
                recorded_training_use="action_space_exhausted",
                recorded_label_quality="bad",
                prediction_matches_action_space_exhausted=True,
                action_critic_fallback_selected=True,
                action_critic_candidate_actions=["STAY_COURSE"],
                action_critic_selected_unsafe_probability=0.88,
                threat_state="air_and_ground_threat",
            ),
        ],
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_strategy_gate_error_analysis_clusters_gate_blockers(tmp_path) -> None:
    audit_path = tmp_path / "audit.json"
    write_json(audit_path, _audit(warnings=["missing_signal_rows:1"]))

    analysis = analyze_strategy_gate_errors([audit_path])

    assert analysis.audits == 1
    assert analysis.rows == 5
    assert analysis.issue_rows == 4
    assert analysis.veto_negative_matches == 1
    assert analysis.action_space_exhausted_matches == 1
    assert analysis.predicted_non_executable_rows == 1
    assert analysis.action_critic_fallback_rows == 3
    assert analysis.fallback_and_veto_negative_rows == 1
    assert analysis.fallback_and_accept_positive_rows == 1
    assert analysis.fallback_by_predicted_action == {"EXPAND": 1, "STAY_COURSE": 2}
    assert analysis.fallback_by_candidate_action_count == {"1": 2, "2": 1}
    assert analysis.fallback_single_candidate_action == {"STAY_COURSE": 2}
    assert analysis.veto_match_by_candidate_action_set == {"STAY_COURSE": 1}
    assert analysis.action_space_match_by_candidate_action_set == {"STAY_COURSE": 1}
    assert analysis.veto_match_by_predicted_action == {"STAY_COURSE": 1}
    assert analysis.veto_match_by_threat_state == {"air_and_ground_threat": 1}
    assert analysis.non_executable_by_blocker == {"cannot_afford_robo": 1}
    assert analysis.critic_veto_action_counts == {"EXPAND": 1, "STAY_COURSE": 1}
    assert analysis.warning_counts == {"missing_signal_rows:1": 1}
    assert asdict(analysis)["examples"][0]["issues"] == [
        "predicted_matches_veto_negative_labels",
        "action_critic_fallback",
    ]


@pytest.mark.unit
def test_strategy_gate_error_analysis_handles_clean_audit(tmp_path) -> None:
    audit_path = tmp_path / "clean.json"
    write_json(audit_path, _audit(decisions=[_decision()]))

    analysis = analyze_strategy_gate_errors([audit_path])

    assert analysis.rows == 1
    assert analysis.issue_rows == 0
    assert analysis.issue_counts == {}
    assert analysis.examples == []


@pytest.mark.unit
def test_strategy_gate_error_analysis_rejects_negative_examples(tmp_path) -> None:
    audit_path = tmp_path / "audit.json"
    write_json(audit_path, _audit())

    with pytest.raises(ValueError, match="max_examples"):
        analyze_strategy_gate_errors([audit_path], max_examples=-1)


@pytest.mark.unit
def test_format_strategy_gate_error_analysis_includes_clusters(tmp_path) -> None:
    audit_path = tmp_path / "audit.json"
    write_json(audit_path, _audit())

    report = format_strategy_gate_error_analysis(
        analyze_strategy_gate_errors([audit_path])
    )

    assert "Strategy gate error analysis" in report
    assert "veto_negative_matches: 1" in report
    assert "fallback_by_predicted_action: EXPAND=1, STAY_COURSE=2" in report
    assert "action_space_exhausted_matches: 1" in report
    assert "fallback_single_candidate_action: STAY_COURSE=2" in report
    assert "non_executable_by_blocker: cannot_afford_robo=1" in report
    assert "examples:" in report
