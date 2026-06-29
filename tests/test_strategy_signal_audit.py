from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from rl.strategy_signal_audit import audit_strategy_signals
from scripts.audit_strategy_signals import format_strategy_signal_audit


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


@pytest.mark.unit
def test_strategy_signal_audit_flags_candidate_signal_regression(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_jsonl(
        baseline / "001.jsonl",
        [
            _row(
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=180.0,
                    ready_gateways=2.0,
                    pending_gateways=0.0,
                ),
                done=True,
                result="Result.Victory",
            ),
        ],
    )
    _write_jsonl(
        candidate / "001.jsonl",
        [
            _row(
                strategy_action=6,
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
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                    ready_gateways=2.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    audit = audit_strategy_signals(baseline, candidate)

    assert audit.signal_healthy is False
    assert audit.baseline.records_by_training_use == {"accept_positive": 1}
    assert audit.candidate.records_by_training_use == {"veto_negative": 1}
    assert audit.accept_positive_ratio_delta == -1.0
    assert audit.bad_signal_ratio_delta == 1.0
    assert audit.veto_negative_ratio_delta == 1.0
    assert audit.blocking_reasons == [
        "accept_positive_ratio_regressed",
        "bad_signal_ratio_regressed",
        "veto_negative_ratio_regressed",
    ]
    assert asdict(audit)["signal_healthy"] is False


@pytest.mark.unit
def test_strategy_signal_audit_allows_unchanged_clean_signal(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    rows = [
        _row(
            strategy_action=3,
            strategy_action_name="TECH_ROBO",
            strategy_observation=_observation(
                game_time=100.0,
                minerals=200.0,
                vespene=150.0,
                pending_robo=0.0,
                ready_robo=0.0,
            ),
        ),
        _row(
            strategy_observation=_observation(
                game_time=160.0,
                pending_robo=1.0,
                ready_robo=0.0,
            ),
            done=True,
            result="Result.Victory",
        ),
    ]
    _write_jsonl(baseline / "001.jsonl", rows)
    _write_jsonl(candidate / "001.jsonl", rows)

    audit = audit_strategy_signals(baseline, candidate)

    assert audit.signal_healthy is True
    assert audit.blocking_reasons == []
    assert audit.warnings == []
    assert audit.accept_positive_ratio_delta == 0.0


@pytest.mark.unit
def test_strategy_signal_audit_warns_on_counterfactual_rows(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_jsonl(
        baseline / "001.jsonl",
        [
            _row(strategy_action=0, strategy_action_name="STAY_COURSE"),
            _row(done=True, result="Result.Tie"),
        ],
    )
    _write_jsonl(
        candidate / "001.jsonl",
        [
            _row(
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    minerals=150.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                ),
            ),
            _row(done=True, result="Result.Tie"),
        ],
    )

    audit = audit_strategy_signals(
        baseline,
        candidate,
        include_before_filter_candidates=True,
    )

    assert audit.candidate.needs_fresh_ab_count == 1
    assert audit.warnings == ["candidate_has_counterfactual_rows_needing_fresh_ab"]


@pytest.mark.unit
def test_format_strategy_signal_audit_contains_gate_sections(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_jsonl(
        baseline / "001.jsonl",
        [
            _row(strategy_action=3, strategy_action_name="TECH_ROBO"),
            _row(
                strategy_observation=_observation(game_time=160.0, pending_robo=1.0),
                done=True,
                result="Result.Victory",
            ),
        ],
    )
    _write_jsonl(
        candidate / "001.jsonl",
        [
            _row(strategy_action=0, strategy_action_name="STAY_COURSE"),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    report = format_strategy_signal_audit(audit_strategy_signals(baseline, candidate))

    assert "Strategy signal audit" in report
    assert "signal_healthy: false" in report
    assert "blocking_reasons:" in report
    assert "accept_positive_ratio_regressed" in report
    assert "signal_ratios:" in report
    assert "training_use:" in report
