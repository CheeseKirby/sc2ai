from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl.strategy_actions import StrategyAction
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from rl.strategy_pre_collapse_recovery_analysis import (
    analyze_strategy_pre_collapse_recovery,
)
from scripts.analyze_strategy_pre_collapse_recovery import (
    format_strategy_pre_collapse_recovery_analysis,
    main as pre_collapse_main,
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


@pytest.mark.unit
def test_pre_collapse_analysis_marks_only_stay_target_unavoidable_with_missed_window(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(game_time=80.0),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    minerals=25.0,
                    vespene=0.0,
                    supply_left=0.0,
                    own_bases=1.0,
                    ready_gateways=4.0,
                    pending_gateways=0.0,
                    gateway_idle_count=0.0,
                    has_cybernetics_core=0.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    ready_forge=0.0,
                    ready_static_defense=2.0,
                    pending_static_defense=0.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(
                done=True,
                result="Result.Defeat",
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
            ),
        ],
    )

    analysis = analyze_strategy_pre_collapse_recovery(trajectory)

    assert analysis.target_rows == 1
    assert analysis.recommendation == "hold"
    assert analysis.blocking_reasons == [
        "missed_pre_collapse_recovery_rows",
        "missed_pre_collapse_recovery_rate",
    ]
    assert analysis.max_missed_pre_collapse_recovery_rows == 0
    assert analysis.max_missed_pre_collapse_recovery_rate == 0.0
    assert analysis.missed_pre_collapse_recovery_rate == 1.0
    assert analysis.avoidability_counts == {"unavoidable_only_stay_course": 1}
    assert analysis.target_training_use_counts == {"action_space_exhausted": 1}
    assert analysis.rows_with_pre_collapse_recovery_window == 1
    assert analysis.rows_with_pre_collapse_selected_executable_recovery == 0
    assert analysis.missed_pre_collapse_recovery_rows == 1
    assert analysis.no_pre_collapse_recovery_window_rows == 0
    assert analysis.pre_collapse_recovery_executable_counts_by_action == {
        "TECH_ROBO": 1,
        "PRODUCE_ARMY": 1,
        "BUILD_STATIC_DEFENSE": 1,
    }

    failure = analysis.failures[0]
    assert failure.recorded_training_use == "action_space_exhausted"
    assert failure.executable_actions == ["STAY_COURSE"]
    assert failure.last_executable_recovery_time == 80.0
    assert failure.last_executable_recovery_actions == [
        "TECH_ROBO",
        "PRODUCE_ARMY",
        "BUILD_STATIC_DEFENSE",
    ]
    assert failure.last_selected_executable_recovery_time is None
    assert failure.recovery_windows[0].seconds_before_target == 40.0
    assert asdict(analysis)["failures"][0]["avoidability"] == (
        "unavoidable_only_stay_course"
    )


@pytest.mark.unit
def test_pre_collapse_analysis_counts_selected_recovery_before_avoidable_target(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=int(StrategyAction.TECH_ROBO),
                strategy_action_name="TECH_ROBO",
                strategy_observation=_observation(game_time=80.0),
                strategy_execution_attempted=True,
                strategy_execution_effect="build_structure",
                strategy_execution_unit_type="ROBOTICSFACILITY",
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    minerals=200.0,
                    vespene=150.0,
                    supply_left=8.0,
                    has_cybernetics_core=1.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(
                done=True,
                result="Result.Defeat",
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
            ),
        ],
    )

    analysis = analyze_strategy_pre_collapse_recovery(trajectory)

    assert analysis.target_rows == 1
    assert analysis.recommendation == "ready"
    assert analysis.blocking_reasons == []
    assert analysis.missed_pre_collapse_recovery_rate == 0.0
    assert analysis.avoidability_counts == {"avoidable_recovery_available": 1}
    assert analysis.target_training_use_counts == {"veto_negative": 1}
    assert analysis.rows_with_pre_collapse_recovery_window == 1
    assert analysis.rows_with_pre_collapse_selected_recovery == 1
    assert analysis.rows_with_pre_collapse_selected_executable_recovery == 1
    assert analysis.missed_pre_collapse_recovery_rows == 0
    assert analysis.pre_collapse_recovery_selected_executable_counts_by_action == {
        "TECH_ROBO": 1
    }

    failure = analysis.failures[0]
    assert failure.executable_recovery_actions == [
        "TECH_ROBO",
        "PRODUCE_ARMY",
        "BUILD_STATIC_DEFENSE",
    ]
    assert failure.last_selected_executable_recovery_time == 80.0
    assert failure.last_selected_executable_recovery_action == "TECH_ROBO"


@pytest.mark.unit
def test_pre_collapse_analysis_validates_arguments(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(trajectory, [_row(done=True, result="Result.Victory")])

    with pytest.raises(ValueError, match="lookback_seconds"):
        analyze_strategy_pre_collapse_recovery(trajectory, lookback_seconds=0)
    with pytest.raises(ValueError, match="max_failures"):
        analyze_strategy_pre_collapse_recovery(trajectory, max_failures=-1)
    with pytest.raises(ValueError, match="max_windows_per_failure"):
        analyze_strategy_pre_collapse_recovery(
            trajectory,
            max_windows_per_failure=-1,
        )
    with pytest.raises(ValueError, match="max_missed_pre_collapse_recovery_rows"):
        analyze_strategy_pre_collapse_recovery(
            trajectory,
            max_missed_pre_collapse_recovery_rows=-1,
        )
    with pytest.raises(ValueError, match="max_missed_pre_collapse_recovery_rate"):
        analyze_strategy_pre_collapse_recovery(
            trajectory,
            max_missed_pre_collapse_recovery_rate=1.1,
        )


@pytest.mark.unit
def test_pre_collapse_action_counts_do_not_depend_on_display_window_cap(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(game_time=60.0),
            ),
            _row(
                step=96,
                strategy_observation=_observation(game_time=80.0),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    minerals=25.0,
                    vespene=0.0,
                    supply_left=0.0,
                    own_bases=1.0,
                    ready_gateways=4.0,
                    pending_gateways=0.0,
                    gateway_idle_count=0.0,
                    has_cybernetics_core=0.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    ready_forge=0.0,
                    ready_static_defense=2.0,
                    pending_static_defense=0.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    analysis = analyze_strategy_pre_collapse_recovery(
        trajectory,
        max_windows_per_failure=1,
    )

    assert len(analysis.failures[0].recovery_windows) == 1
    assert analysis.pre_collapse_recovery_executable_counts_by_action == {
        "TECH_ROBO": 2,
        "PRODUCE_ARMY": 2,
        "BUILD_STATIC_DEFENSE": 2,
    }
    assert analysis.failures[0].pre_collapse_recovery_executable_counts_by_action == {
        "TECH_ROBO": 2,
        "PRODUCE_ARMY": 2,
        "BUILD_STATIC_DEFENSE": 2,
    }


@pytest.mark.unit
def test_format_pre_collapse_analysis_includes_summary(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(strategy_observation=_observation(game_time=80.0)),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    minerals=25.0,
                    vespene=0.0,
                    supply_left=0.0,
                    own_bases=1.0,
                    ready_gateways=4.0,
                    has_cybernetics_core=0.0,
                    ready_static_defense=2.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(
                step=192,
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

    report = format_strategy_pre_collapse_recovery_analysis(
        analyze_strategy_pre_collapse_recovery(trajectory)
    )

    assert "Strategy pre-collapse recovery analysis" in report
    assert "recommendation: hold" in report
    assert "blocking_reasons: missed_pre_collapse_recovery_rows" in report
    assert "target_rows: 1" in report
    assert "unavoidable_only_stay_course=1" in report
    assert "missed_pre_collapse_recovery_rate: 1.000" in report
    assert "missed_pre_collapse_recovery_rows: 1/1" in report
    assert "failures:" in report


@pytest.mark.unit
def test_pre_collapse_cli_fail_on_hold_respects_gate_thresholds(
    tmp_path,
    monkeypatch,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(strategy_observation=_observation(game_time=80.0)),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=120.0,
                    minerals=25.0,
                    vespene=0.0,
                    supply_left=0.0,
                    own_bases=1.0,
                    ready_gateways=4.0,
                    has_cybernetics_core=0.0,
                    ready_static_defense=2.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_strategy_pre_collapse_recovery.py",
            str(trajectory),
            "--fail-on-hold",
        ],
    )
    assert pre_collapse_main() == 1

    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_strategy_pre_collapse_recovery.py",
            str(trajectory),
            "--fail-on-hold",
            "--max-missed-pre-collapse-recovery-rows",
            "1",
            "--max-missed-pre-collapse-recovery-rate",
            "1.0",
        ],
    )
    assert pre_collapse_main() == 0
