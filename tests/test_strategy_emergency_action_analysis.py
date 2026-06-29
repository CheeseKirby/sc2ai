from __future__ import annotations

import json

import pytest

from rl.strategy_emergency_action_analysis import analyze_strategy_emergency_actions
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.analyze_strategy_emergency_actions import (
    format_strategy_emergency_action_analysis,
)


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "game_time": 100.0,
            "minerals": 25.0,
            "vespene": 0.0,
            "supply_left": 0.0,
            "workers": 18.0,
            "own_bases": 1.0,
            "ready_gateways": 4.0,
            "pending_gateways": 0.0,
            "ready_static_defense": 0.0,
            "pending_static_defense": 0.0,
            "has_cybernetics_core": 0.0,
            "worker_saturation_ratio": 0.8,
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


def _details(**overrides: float) -> dict[str, float]:
    details = {
        "ready_photon_cannons": 0.0,
        "pending_photon_cannons": 0.0,
        "ready_shield_batteries": 0.0,
        "pending_shield_batteries": 0.0,
    }
    details.update(overrides)
    return details


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_emergency_action_analysis_covers_ground_threat_with_assets(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=3.0,
                    zealots=3.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=1.0,
                    zealots=1.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    analysis = analyze_strategy_emergency_actions(trajectory)

    assert analysis.rows == 1
    assert analysis.observation_detail_rows == 0
    assert analysis.observation_detail_ratio == 0.0
    assert analysis.threatened_only_stay_course_rows == 1
    assert analysis.threatened_only_stay_course_detail_rows == 0
    assert analysis.action_space_exhausted_rows == 1
    assert analysis.addressable_threatened_only_stay_course_rows == 1
    assert analysis.addressable_action_space_exhausted_rows == 1
    assert analysis.emergency_action_count == {"1": 1}
    assert analysis.emergency_action_sets == {"EMERGENCY_DEFEND": 1}
    assert analysis.addressable_by_training_use == {"action_space_exhausted": 1}
    assert analysis.examples[0].emergency_actions == ["EMERGENCY_DEFEND"]
    assert analysis.examples[0].emergency_blockers == {}


@pytest.mark.unit
def test_emergency_action_analysis_keeps_air_threat_without_anti_air_uncovered(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    army_count=3.0,
                    zealots=3.0,
                    stalkers=0.0,
                    ready_static_defense=0.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    analysis = analyze_strategy_emergency_actions(trajectory)

    assert analysis.threatened_only_stay_course_rows == 1
    assert analysis.action_space_exhausted_rows == 1
    assert analysis.addressable_threatened_only_stay_course_rows == 0
    assert analysis.addressable_action_space_exhausted_rows == 0
    assert analysis.emergency_action_count == {"0": 1}
    assert analysis.unaddressed_by_threat_state == {"air_threat": 1}
    assert analysis.emergency_blockers_by_action == {
        "EMERGENCY_DEFEND": {"no_air_defense_assets": 1}
    }
    assert analysis.unaddressed_air_defense_gap_by_reason == {
        "no_observed_anti_air_assets": 1
    }
    assert analysis.examples[0].emergency_actions == []
    assert analysis.examples[0].emergency_blockers == {
        "EMERGENCY_DEFEND": "no_air_defense_assets"
    }


@pytest.mark.unit
def test_emergency_action_analysis_flags_ambiguous_static_defense_gap(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_static_defense=2.0,
                    pending_static_defense=1.0,
                    army_count=3.0,
                    zealots=3.0,
                    stalkers=0.0,
                    sentries=0.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    analysis = analyze_strategy_emergency_actions(trajectory)

    assert analysis.addressable_threatened_only_stay_course_rows == 0
    assert analysis.unaddressed_air_defense_gap_by_reason == {
        "static_defense_type_ambiguous": 1
    }
    assert analysis.examples[0].air_defense_gap_reason == (
        "static_defense_type_ambiguous"
    )


@pytest.mark.unit
def test_emergency_action_analysis_uses_photon_cannon_details_for_air_defense(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_static_defense=2.0,
                    army_count=3.0,
                    zealots=3.0,
                    stalkers=0.0,
                    sentries=0.0,
                ),
                strategy_observation_details=_details(
                    ready_photon_cannons=1.0,
                    ready_shield_batteries=1.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    analysis = analyze_strategy_emergency_actions(trajectory)

    assert analysis.addressable_threatened_only_stay_course_rows == 1
    assert analysis.observation_detail_rows == 1
    assert analysis.observation_detail_ratio == 1.0
    assert analysis.threatened_only_stay_course_detail_rows == 1
    assert analysis.air_threat_only_stay_course_rows == 1
    assert analysis.air_threat_only_stay_course_detail_rows == 1
    assert analysis.air_threat_only_stay_course_detail_ratio == 1.0
    assert analysis.unaddressed_air_defense_gap_by_reason == {}
    assert analysis.examples[0].emergency_actions == ["EMERGENCY_DEFEND"]
    assert analysis.examples[0].air_defense_gap_reason == "air_defense_assets_present"


@pytest.mark.unit
def test_emergency_action_analysis_uses_shield_only_details_as_no_anti_air(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_static_defense=2.0,
                    army_count=3.0,
                    zealots=3.0,
                    stalkers=0.0,
                    sentries=0.0,
                ),
                strategy_observation_details=_details(
                    ready_shield_batteries=2.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    analysis = analyze_strategy_emergency_actions(trajectory)

    assert analysis.addressable_threatened_only_stay_course_rows == 0
    assert analysis.observation_detail_rows == 1
    assert analysis.threatened_only_stay_course_detail_rows == 1
    assert analysis.air_threat_only_stay_course_detail_rows == 1
    assert analysis.unaddressed_air_defense_gap_by_reason == {
        "no_observed_anti_air_assets": 1
    }
    assert analysis.examples[0].air_defense_gap_reason == (
        "no_observed_anti_air_assets"
    )


@pytest.mark.unit
def test_emergency_action_analysis_rejects_negative_examples(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(trajectory, [_row(done=True, result="Result.Victory")])

    with pytest.raises(ValueError, match="max_examples"):
        analyze_strategy_emergency_actions(trajectory, max_examples=-1)


@pytest.mark.unit
def test_format_strategy_emergency_action_analysis_includes_coverage(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_observation=_observation(
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=2.0,
                    zealots=2.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    report = format_strategy_emergency_action_analysis(
        analyze_strategy_emergency_actions(trajectory)
    )

    assert "Strategy emergency action analysis" in report
    assert "observation_details: 0/1 ratio=0.000" in report
    assert "threatened_only_stay_course_details: 0/1 ratio=0.000" in report
    assert "threatened_only_stay_course: 1/1" in report
    assert "addressable_threatened_only_stay_course: 1/1" in report
    assert "unaddressed_air_defense_gap_by_reason: <none>" in report
    assert "EMERGENCY_DEFEND=1" in report
    assert "examples:" in report
