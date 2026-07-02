from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl.strategy_anti_air_recovery_analysis import (
    analyze_strategy_anti_air_recovery,
)
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.analyze_strategy_anti_air_recovery import (
    format_strategy_anti_air_recovery_analysis,
)


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "game_time": 100.0,
            "minerals": 200.0,
            "vespene": 120.0,
            "supply_left": 4.0,
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
            "stalkers": 2.0,
            "army_count": 4.0,
            "worker_saturation_ratio": 0.9,
        }
    )
    observation.update(overrides)
    return observation


def _details(**overrides: float) -> dict[str, float]:
    details = {
        "ready_photon_cannons": 0.0,
        "pending_photon_cannons": 0.0,
        "ready_shield_batteries": 0.0,
        "pending_shield_batteries": 0.0,
    }
    details.update(overrides)
    return details


def _row(**overrides) -> dict:
    row = {
        "step": 64,
        "map_name": "AcropolisLE",
        "difficulty": "Hard",
        "opponent_race": "Terran",
        "opponent_ai_build": "Air",
        "strategy_observation": _observation(),
        "strategy_observation_details": _details(),
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
def test_anti_air_recovery_analysis_detects_missed_pre_gap_window(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(game_time=100.0),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    stalkers=0.0,
                    sentries=0.0,
                    army_count=0.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    analysis = analyze_strategy_anti_air_recovery(trajectory)

    assert analysis.files == 1
    assert analysis.air_threat_rows == 1
    assert analysis.air_threat_rows_without_anti_air == 1
    assert analysis.anti_air_gap_files == 1
    assert analysis.files_with_pre_gap_recovery_window == 1
    assert analysis.files_with_pre_gap_recovery_selected == 0
    assert analysis.files_with_pre_gap_executable_recovery_selected == 0
    assert analysis.missed_recovery_windows == 1
    assert analysis.recovery_executable_counts_by_action == {
        "PRODUCE_ARMY": 1,
        "BUILD_STATIC_DEFENSE": 1,
        "TECH_ROBO": 1,
    }
    assert analysis.missed_executable_recovery_counts_by_action == {
        "PRODUCE_ARMY": 1,
        "BUILD_STATIC_DEFENSE": 1,
        "TECH_ROBO": 1,
    }

    summary = analysis.file_summaries[0]
    assert summary.first_air_threat_time == 120.0
    assert summary.first_air_threat_without_anti_air_time == 120.0
    assert summary.last_anti_air_before_gap_time == 100.0
    assert summary.first_anti_air_absent_after_asset_time == 120.0
    assert summary.recovery_window_rows == 1
    assert summary.missed_recovery_window is True

    assert analysis.examples[0].row_role == "pre_gap"
    assert analysis.examples[0].seconds_before_gap == 20.0
    assert analysis.examples[0].anti_air_assets_present is True
    assert asdict(analysis)["examples"][0]["start_metrics"]["stalkers"] == 2.0


@pytest.mark.unit
def test_anti_air_recovery_analysis_counts_selected_executable_recovery(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(game_time=100.0),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    stalkers=0.0,
                    army_count=0.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    analysis = analyze_strategy_anti_air_recovery(trajectory)

    assert analysis.files_with_pre_gap_recovery_selected == 1
    assert analysis.files_with_pre_gap_executable_recovery_selected == 1
    assert analysis.missed_recovery_windows == 0
    assert analysis.recovery_selected_counts_by_action == {
        "BUILD_STATIC_DEFENSE": 1
    }
    assert analysis.recovery_selected_executable_counts_by_action == {
        "BUILD_STATIC_DEFENSE": 1
    }
    assert analysis.file_summaries[0].missed_recovery_window is False


@pytest.mark.unit
def test_anti_air_recovery_analysis_separates_no_possible_recovery(
    tmp_path,
) -> None:
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
                    ready_gateways=0.0,
                    gateway_idle_count=0.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    ready_forge=1.0,
                    ready_static_defense=2.0,
                    pending_static_defense=0.0,
                    has_cybernetics_core=0.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    stalkers=0.0,
                    army_count=0.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    analysis = analyze_strategy_anti_air_recovery(trajectory)

    assert analysis.anti_air_gap_files == 1
    assert analysis.files_with_pre_gap_recovery_window == 0
    assert analysis.missed_recovery_windows == 0
    assert analysis.recovery_executable_counts_by_action == {}
    assert analysis.blockers_by_action == {
        "PRODUCE_ARMY": {"supply_blocked_army": 1},
        "BUILD_STATIC_DEFENSE": {"static_defense_cap_reached": 1},
        "TECH_ROBO": {"missing_cybernetics_core": 1},
    }


@pytest.mark.unit
def test_anti_air_recovery_analysis_uses_photon_cannon_details(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    game_time=100.0,
                    stalkers=0.0,
                    army_count=0.0,
                    ready_static_defense=1.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                ),
                strategy_observation_details=_details(ready_photon_cannons=1.0),
            ),
            _row(done=True, result="Result.Tie"),
        ],
    )

    analysis = analyze_strategy_anti_air_recovery(trajectory)

    assert analysis.air_threat_rows == 1
    assert analysis.air_threat_rows_with_anti_air == 1
    assert analysis.air_threat_rows_without_anti_air == 0
    assert analysis.anti_air_gap_files == 0
    assert analysis.file_summaries[0].first_air_threat_without_anti_air_time is None
    assert analysis.examples == []


@pytest.mark.unit
def test_anti_air_recovery_analysis_counts_successful_photon_execution(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=0.0,
                    ready_forge=1.0,
                    stalkers=2.0,
                    ready_static_defense=0.0,
                    pending_static_defense=1.0,
                ),
                strategy_execution_attempted=True,
                strategy_execution_effect="build_structure",
                strategy_execution_unit_type="PHOTONCANNON",
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    stalkers=0.0,
                    army_count=0.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    analysis = analyze_strategy_anti_air_recovery(trajectory)

    assert analysis.files_with_pre_gap_executable_recovery_selected == 1
    assert analysis.missed_recovery_windows == 0
    assert analysis.recovery_selected_executable_counts_by_action == {
        "BUILD_STATIC_DEFENSE": 1
    }


@pytest.mark.unit
def test_anti_air_recovery_analysis_rejects_negative_examples(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(trajectory, [_row(done=True, result="Result.Victory")])

    with pytest.raises(ValueError, match="max_examples"):
        analyze_strategy_anti_air_recovery(trajectory, max_examples=-1)


@pytest.mark.unit
def test_format_strategy_anti_air_recovery_analysis_includes_summary(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(strategy_observation=_observation(game_time=100.0)),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    stalkers=0.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    report = format_strategy_anti_air_recovery_analysis(
        analyze_strategy_anti_air_recovery(trajectory)
    )

    assert "Strategy anti-air recovery analysis" in report
    assert "air_threat_rows_without_anti_air: 1/1" in report
    assert "missed_recovery_windows: 1/1" in report
    assert "BUILD_STATIC_DEFENSE=1" in report
    assert "examples:" in report
