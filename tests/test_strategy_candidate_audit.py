from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl.strategy_candidate_audit import audit_strategy_candidate
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.audit_strategy_candidate import format_strategy_candidate_audit


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "minerals": 150.0,
            "vespene": 100.0,
            "workers": 20.0,
            "own_bases": 1.0,
            "ready_gateways": 1.0,
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
        "strategy_observation": _observation(game_time=100.0),
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
def test_strategy_candidate_audit_flags_regressions(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_jsonl(
        baseline / "001.jsonl",
        [
            _row(
                strategy_observation=_observation(game_time=90.0),
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
                strategy_execution_effect="build_structure",
            ),
            _row(
                strategy_observation=_observation(game_time=240.0),
                strategy_action=3,
                strategy_action_name="TECH_ROBO",
                strategy_execution_effect="build_structure",
            ),
            _row(
                strategy_observation=_observation(game_time=300.0, base_under_threat=1.0),
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_execution_effect="build_structure",
            ),
            _row(
                strategy_observation=_observation(game_time=360.0),
                done=True,
                result="Result.Victory",
            ),
        ],
    )
    _write_jsonl(
        candidate / "001.jsonl",
        [
            _row(
                strategy_observation=_observation(game_time=90.0),
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
                strategy_execution_effect="build_structure",
            ),
            _row(
                strategy_observation=_observation(game_time=240.0, base_under_threat=1.0),
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                tactic_id="RECOVERY",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
                strategy_execution_effect="delegate_train_army",
                strategy_execution_blocker="no_ready_robo",
            ),
            _row(
                strategy_observation=_observation(game_time=300.0, base_under_threat=1.0),
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                strategy_execution_effect="noop",
                strategy_execution_blocker="no_ready_robo",
            ),
            _row(
                strategy_observation=_observation(game_time=360.0),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    audit = audit_strategy_candidate(baseline, candidate)

    assert audit.promotable is False
    assert audit.baseline.result_counts == {"Result.Victory": 1}
    assert audit.candidate.result_counts == {"Result.Defeat": 1}
    assert audit.result_score_delta == -1.0
    assert audit.base_threat_rows_delta == 1
    assert audit.action_count_delta_by_name["ADD_GATEWAYS"] == 0
    assert audit.action_count_delta_by_name["TECH_ROBO"] == -1
    assert audit.action_count_delta_by_name["BUILD_STATIC_DEFENSE"] == -1
    assert audit.execution_blocker_delta == 2
    assert audit.blocking_reasons == [
        "candidate_result_worse_than_baseline",
        "base_threat_rows_regressed",
        "tech_robo_count_regressed",
        "build_static_defense_count_regressed",
        "execution_blockers_increased",
    ]
    assert audit.warnings == ["candidate_has_filter_changes"]


@pytest.mark.unit
def test_strategy_candidate_audit_allows_clean_candidate(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    rows = [
        _row(
            strategy_observation=_observation(game_time=90.0),
            strategy_action=2,
            strategy_action_name="ADD_GATEWAYS",
            strategy_execution_effect="build_structure",
        ),
        _row(
            strategy_observation=_observation(game_time=240.0),
            strategy_action=3,
            strategy_action_name="TECH_ROBO",
            strategy_execution_effect="build_structure",
        ),
        _row(
            strategy_observation=_observation(game_time=300.0),
            strategy_action=5,
            strategy_action_name="BUILD_STATIC_DEFENSE",
            strategy_execution_effect="build_structure",
        ),
        _row(
            strategy_observation=_observation(game_time=360.0),
            done=True,
            result="Result.Victory",
        ),
    ]
    _write_jsonl(baseline / "001.jsonl", rows)
    _write_jsonl(candidate / "001.jsonl", rows)

    audit = audit_strategy_candidate(baseline, candidate)

    assert audit.promotable is True
    assert audit.blocking_reasons == []
    assert audit.warnings == []
    assert audit.result_score_delta == 0.0


@pytest.mark.unit
def test_strategy_candidate_audit_scores_bare_result_names(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_jsonl(
        baseline / "001.jsonl",
        [
            _row(strategy_action=3, strategy_action_name="TECH_ROBO"),
            _row(done=True, result="Victory"),
        ],
    )
    _write_jsonl(
        candidate / "001.jsonl",
        [
            _row(strategy_action=3, strategy_action_name="TECH_ROBO"),
            _row(done=True, result="Defeat"),
        ],
    )

    audit = audit_strategy_candidate(baseline, candidate)

    assert audit.baseline.result_counts == {"Victory": 1}
    assert audit.candidate.result_counts == {"Defeat": 1}
    assert audit.result_score_delta == -1.0
    assert "candidate_result_worse_than_baseline" in audit.blocking_reasons


@pytest.mark.unit
def test_format_strategy_candidate_audit_contains_gate_sections(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_jsonl(
        baseline / "001.jsonl",
        [
            _row(strategy_action=3, strategy_action_name="TECH_ROBO"),
            _row(done=True, result="Result.Victory"),
        ],
    )
    _write_jsonl(
        candidate / "001.jsonl",
        [
            _row(strategy_action=0, strategy_action_name="STAY_COURSE"),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    audit = audit_strategy_candidate(baseline, candidate)
    report = format_strategy_candidate_audit(audit)

    assert "Strategy candidate audit" in report
    assert "promotable: false" in report
    assert "blocking_reasons:" in report
    assert "candidate_result_worse_than_baseline" in report
    assert "action_delta:" in report
    assert "TECH_ROBO=-1" in report
    assert asdict(audit)["promotable"] is False
