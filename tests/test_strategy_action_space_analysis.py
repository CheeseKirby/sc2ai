from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl.strategy_action_space_analysis import analyze_strategy_action_space
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.analyze_strategy_action_space import format_strategy_action_space_analysis


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
def test_strategy_action_space_analysis_clusters_only_stay_rows(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
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
                    workers=18.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                    workers=14.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    analysis = analyze_strategy_action_space(trajectory)

    assert analysis.rows == 1
    assert analysis.training_rows == 1
    assert analysis.only_stay_course_rows == 1
    assert analysis.only_stay_course_under_threat_rows == 1
    assert analysis.only_stay_course_veto_negative_rows == 0
    assert analysis.executable_action_count == {"1": 1}
    assert analysis.executable_action_sets == {"STAY_COURSE": 1}
    assert analysis.only_stay_course_by_training_use == {
        "action_space_exhausted": 1
    }
    assert analysis.only_stay_course_by_threat_state == {"ground_threat": 1}
    assert analysis.only_stay_course_blockers_by_action["ADD_GATEWAYS"] == {
        "target_gateways_reached": 1
    }
    assert analysis.only_stay_course_blockers_by_action["PRODUCE_ARMY"] == {
        "supply_blocked_army": 1
    }
    assert analysis.only_stay_course_blockers_by_action["TECH_ROBO"] == {
        "missing_cybernetics_core": 1
    }
    assert analysis.examples[0].executable_actions == ["STAY_COURSE"]
    assert "threat_persisted" in analysis.examples[0].reasons
    assert asdict(analysis)["examples"][0]["blocked_actions"]["EXPAND"] == (
        "cannot_afford_nexus"
    )


@pytest.mark.unit
def test_strategy_action_space_analysis_handles_open_action_space(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=500.0,
                    vespene=200.0,
                    supply_left=10.0,
                    own_bases=1.0,
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                    gateway_idle_count=1.0,
                    has_cybernetics_core=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                ),
            ),
            _row(done=True, result="Result.Victory"),
        ],
    )

    analysis = analyze_strategy_action_space(trajectory)

    assert analysis.rows == 1
    assert analysis.only_stay_course_rows == 0
    assert analysis.only_stay_course_blockers_by_action == {}
    assert analysis.examples == []
    assert "1" not in analysis.executable_action_count


@pytest.mark.unit
def test_strategy_action_space_analysis_rejects_negative_examples(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(trajectory, [_row(done=True, result="Result.Victory")])

    with pytest.raises(ValueError, match="max_examples"):
        analyze_strategy_action_space(trajectory, max_examples=-1)


@pytest.mark.unit
def test_format_strategy_action_space_analysis_includes_clusters(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=25.0,
                    supply_left=0.0,
                    own_bases=1.0,
                    ready_gateways=4.0,
                    has_cybernetics_core=0.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    report = format_strategy_action_space_analysis(
        analyze_strategy_action_space(trajectory)
    )

    assert "Strategy action-space analysis" in report
    assert "only_stay_course: 1/1" in report
    assert "only_stay_course_under_threat: 1/1" in report
    assert "PRODUCE_ARMY: supply_blocked_army=1" in report
    assert "examples:" in report
