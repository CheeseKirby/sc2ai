from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl.experiments import write_json
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from rl.strategy_veto_negative_analysis import analyze_strategy_veto_negatives
from scripts.analyze_strategy_veto_negatives import (
    format_strategy_veto_negative_analysis,
)


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
        "strategy_action": 0,
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


def _audit_decision(path, **overrides) -> dict:
    payload = {
        "path": str(path),
        "step": 64,
        "game_time": 100.0,
        "recorded_action": "STAY_COURSE",
        "predicted_action": "STAY_COURSE",
        "context": "other",
        "threat_state": "air_and_ground_threat",
        "prediction_matches_veto_negative": True,
        "action_critic_fallback_selected": False,
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_strategy_veto_negative_analysis_clusters_veto_rows(tmp_path) -> None:
    stay_path = tmp_path / "stay.jsonl"
    produce_path = tmp_path / "produce.jsonl"
    _write_jsonl(
        stay_path,
        [
            _row(
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    army_count=8.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=180.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )
    _write_jsonl(
        produce_path,
        [
            _row(
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                    supply_left=8.0,
                    army_count=10.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=180.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    army_count=5.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    analysis = analyze_strategy_veto_negatives([stay_path, produce_path])

    assert analysis.files == 2
    assert analysis.veto_negative_records == 2
    assert analysis.by_action == {"PRODUCE_ARMY": 1, "STAY_COURSE": 1}
    assert analysis.by_threat_state == {
        "air_and_ground_threat": 1,
        "ground_threat": 1,
    }
    assert analysis.by_reason["threat_persisted"] == 2
    assert analysis.negative_events_by_window["120s"]["threat_persisted"] == 2
    assert analysis.start_metric_buckets["army_count"] == {"10-19": 1, "5-9": 1}
    assert analysis.last_window_metric_buckets["army_count_delta"] == {
        "-5..-1": 1,
        "0": 1,
    }
    assert asdict(analysis)["examples"][0]["candidate_action"] == "PRODUCE_ARMY"


@pytest.mark.unit
def test_strategy_veto_negative_analysis_joins_audit_matches(tmp_path) -> None:
    trajectory = tmp_path / "stay.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=180.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )
    audit_path = tmp_path / "audit.json"
    write_json(
        audit_path,
        {
            "decisions": [
                _audit_decision(
                    trajectory,
                    action_critic_fallback_selected=True,
                ),
                _audit_decision(
                    trajectory,
                    step=128,
                    prediction_matches_veto_negative=False,
                ),
            ],
        },
    )

    analysis = analyze_strategy_veto_negatives(
        trajectory,
        audit_paths=[audit_path],
    )

    assert analysis.audit_decisions == 2
    assert analysis.matched_by_audit_decisions == 1
    assert analysis.matched_by_any_audit_records == 1
    assert analysis.matched_by_audit_action == {"STAY_COURSE": 1}
    assert analysis.matched_by_audit_fallback_selected == {"true": 1}
    assert analysis.examples[0].matched_audits == [str(audit_path)]
    assert analysis.examples[0].audit_fallback_selected is True


@pytest.mark.unit
def test_strategy_veto_negative_analysis_rejects_negative_examples(tmp_path) -> None:
    trajectory = tmp_path / "empty.jsonl"
    _write_jsonl(trajectory, [_row(done=True, result="Result.Victory")])

    with pytest.raises(ValueError, match="max_examples"):
        analyze_strategy_veto_negatives(trajectory, max_examples=-1)


@pytest.mark.unit
def test_format_strategy_veto_negative_analysis_includes_clusters(tmp_path) -> None:
    trajectory = tmp_path / "stay.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=180.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    report = format_strategy_veto_negative_analysis(
        analyze_strategy_veto_negatives(trajectory)
    )

    assert "Strategy veto-negative analysis" in report
    assert "veto_negative_records: 1" in report
    assert "by_action: STAY_COURSE=1" in report
    assert "negative_events_by_window:" in report
    assert "examples:" in report
