from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from rl.strategy_outcome_diagnostics import diagnose_strategy_outcomes
from scripts.diagnose_strategy_outcomes import format_strategy_outcome_diagnostics


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "minerals": 150.0,
            "vespene": 50.0,
            "workers": 20.0,
            "own_bases": 1.0,
            "ready_gateways": 1.0,
            "army_count": 4.0,
            "worker_saturation_ratio": 0.9,
        }
    )
    observation.update(overrides)
    return observation


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


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


@pytest.mark.unit
def test_strategy_outcome_diagnostics_reports_action_landing_windows(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                ),
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=128.0,
                    ready_gateways=1.0,
                    pending_gateways=1.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=170.0,
                    ready_gateways=2.0,
                    pending_gateways=0.0,
                    army_count=6.0,
                ),
            ),
            _row(
                step=256,
                strategy_observation=_observation(
                    game_time=220.0,
                    ready_gateways=2.0,
                    pending_gateways=0.0,
                    army_count=6.0,
                ),
                done=True,
                result="Result.Victory",
            ),
        ],
    )

    diagnostics = diagnose_strategy_outcomes(trajectory)

    assert diagnostics.files == 1
    assert diagnostics.rows == 4
    assert diagnostics.training_rows == 3
    assert diagnostics.result_counts == {"Result.Victory": 1}
    assert diagnostics.action_summaries_by_name["ADD_GATEWAYS"].count == 1
    window_30 = diagnostics.action_window_summaries["ADD_GATEWAYS"]["30s"]
    assert window_30.samples == 1
    assert window_30.event_counts["pending_gateway_seen"] == 1
    assert window_30.avg_event_times["first_pending_gateway_after_action"] == 28.0
    assert window_30.avg_metrics["ready_gateway_delta"] == 0.0
    window_120 = diagnostics.action_window_summaries["ADD_GATEWAYS"]["120s"]
    assert window_120.avg_metrics["ready_gateway_delta"] == 1.0
    assert window_120.event_counts["ready_gateway_increased"] == 1
    assert window_120.avg_event_times["first_ready_gateway_delta_time"] == 70.0


@pytest.mark.unit
def test_strategy_outcome_diagnostics_groups_tactic_filter_changes(tmp_path) -> None:
    trajectory = tmp_path / "tactic_strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    game_time=90.0,
                    ready_gateways=1.0,
                    workers=18.0,
                    army_count=4.0,
                ),
                strategy_action=7,
                strategy_action_name="BOOST_WORKERS",
                tactic_id="SAFE_MACRO",
                tactic_phase="OPENING",
                tactic_source="rule",
                strategy_action_before_tactic_filter=2,
                strategy_action_before_tactic_filter_name="ADD_GATEWAYS",
                strategy_action_after_tactic_filter=7,
                strategy_action_after_tactic_filter_name="BOOST_WORKERS",
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=150.0,
                    ready_gateways=1.0,
                    workers=21.0,
                    army_count=4.0,
                ),
                strategy_action=0,
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=210.0,
                    ready_gateways=1.0,
                    workers=22.0,
                    army_count=5.0,
                ),
                strategy_action=0,
            ),
            _row(
                step=256,
                strategy_observation=_observation(
                    game_time=260.0,
                    base_under_threat=1.0,
                    ready_static_defense=1.0,
                    army_count=5.0,
                ),
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
            ),
            _row(
                step=320,
                strategy_observation=_observation(
                    game_time=320.0,
                    base_under_threat=0.0,
                    ready_static_defense=1.0,
                    army_count=8.0,
                ),
                strategy_action=0,
            ),
            _row(
                step=384,
                strategy_observation=_observation(game_time=360.0),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    diagnostics = diagnose_strategy_outcomes(trajectory)

    safe_macro = next(
        summary
        for summary in diagnostics.filter_change_summaries
        if summary.tactic_id == "SAFE_MACRO"
    )
    assert safe_macro.before_action == "ADD_GATEWAYS"
    assert safe_macro.after_action == "BOOST_WORKERS"
    assert safe_macro.count == 1
    assert safe_macro.early_before_240_count == 1
    assert safe_macro.outcomes_by_window["120s"].avg_metrics["worker_delta"] == 4.0
    assert safe_macro.outcomes_by_window["120s"].avg_metrics["ready_gateway_delta"] == 0.0

    static_suppression = next(
        summary
        for summary in diagnostics.filter_change_summaries
        if summary.before_action == "BUILD_STATIC_DEFENSE"
    )
    assert static_suppression.after_action == "PRODUCE_ARMY"
    assert static_suppression.outcomes_by_window["60s"].avg_metrics["army_count_delta"] == 3.0
    assert static_suppression.outcomes_by_window["60s"].event_counts["threat_cleared"] == 1


@pytest.mark.unit
def test_strategy_outcome_diagnostics_classifies_robo_payoff_blockers(tmp_path) -> None:
    directory = tmp_path / "strategy_dir"
    directory.mkdir()
    _write_jsonl(
        directory / "001_no_action.jsonl",
        [
            _row(
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_robo=1.0,
                    robo_idle_count=1.0,
                    minerals=300.0,
                    vespene=150.0,
                    supply_left=8.0,
                ),
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
            ),
            _row(
                strategy_observation=_observation(game_time=140.0, ready_robo=1.0),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )
    _write_jsonl(
        directory / "002_not_idle.jsonl",
        [
            _row(
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_robo=1.0,
                    robo_idle_count=0.0,
                    minerals=300.0,
                    vespene=150.0,
                    supply_left=8.0,
                ),
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
            ),
            _row(
                strategy_observation=_observation(game_time=140.0, ready_robo=1.0),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )
    _write_jsonl(
        directory / "003_resource_blocked.jsonl",
        [
            _row(
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_robo=1.0,
                    robo_idle_count=1.0,
                    minerals=80.0,
                    vespene=150.0,
                    supply_left=8.0,
                ),
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
            ),
            _row(
                strategy_observation=_observation(
                    game_time=130.0,
                    ready_robo=1.0,
                    robo_idle_count=1.0,
                    observers=1.0,
                    minerals=120.0,
                    vespene=200.0,
                    supply_left=8.0,
                ),
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
            ),
            _row(
                strategy_observation=_observation(game_time=170.0, ready_robo=1.0),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    diagnostics = diagnose_strategy_outcomes(directory)
    payoff_by_file = {
        Path(summary.path).name: summary.robo_payoff
        for summary in diagnostics.file_summaries
    }

    assert payoff_by_file["001_no_action.jsonl"].immortal_blocker == "action_not_triggered"
    assert payoff_by_file["002_not_idle.jsonl"].immortal_blocker == "robo_not_idle"
    resource_blocked = payoff_by_file["003_resource_blocked.jsonl"]
    assert resource_blocked.observer_status == "produced_after_ready"
    assert resource_blocked.observer_after_ready_delay_seconds == 30.0
    assert resource_blocked.immortal_blocker == "resource_or_supply_blocked"
    assert resource_blocked.robo_action_rows_after_ready == 2
    assert resource_blocked.robo_idle_rows_after_ready == 2
    assert resource_blocked.immortal_mineral_blocked_candidate_rows == 2


@pytest.mark.unit
def test_strategy_outcome_source_first_actions_use_earliest_time_across_files(tmp_path) -> None:
    directory = tmp_path / "strategy_dir"
    directory.mkdir()
    _write_jsonl(
        directory / "001.jsonl",
        [
            _row(
                strategy_observation=_observation(game_time=300.0),
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
            ),
            _row(
                strategy_observation=_observation(game_time=340.0),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )
    _write_jsonl(
        directory / "002.jsonl",
        [
            _row(
                strategy_observation=_observation(game_time=120.0),
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
            ),
            _row(
                strategy_observation=_observation(game_time=160.0),
                done=True,
                result="Result.Victory",
            ),
        ],
    )

    diagnostics = diagnose_strategy_outcomes(directory)

    assert diagnostics.action_summaries_by_name["ADD_GATEWAYS"].first_game_time == 120.0
    assert diagnostics.source_summaries[0].action_first_game_time_by_name[
        "ADD_GATEWAYS"
    ] == 120.0


@pytest.mark.unit
def test_strategy_outcome_diagnostics_summarizes_execution_metadata(tmp_path) -> None:
    trajectory = tmp_path / "strategy_execution.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                strategy_observation=_observation(game_time=240.0),
                strategy_action=3,
                strategy_action_name="TECH_ROBO",
                strategy_execution_attempted=True,
                strategy_execution_effect="build_structure",
                strategy_execution_unit_type="ROBOTICSFACILITY",
                strategy_execution_target="power_field",
            ),
            _row(
                strategy_observation=_observation(game_time=300.0),
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_execution_attempted=True,
                strategy_execution_effect="delegate_train_army",
                strategy_execution_blocker="no_ready_robo",
            ),
            _row(
                strategy_observation=_observation(game_time=360.0),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    diagnostics = diagnose_strategy_outcomes(trajectory)

    assert diagnostics.execution_effect_counts == {
        "build_structure": 1,
        "delegate_train_army": 1,
    }
    assert diagnostics.execution_blocker_counts == {"no_ready_robo": 1}
    assert diagnostics.source_summaries[0].execution_effect_counts == {
        "build_structure": 1,
        "delegate_train_army": 1,
    }
    assert diagnostics.file_summaries[0].execution_blocker_counts == {
        "no_ready_robo": 1,
    }

    report = format_strategy_outcome_diagnostics(diagnostics, show_files=True)
    assert "execution:" in report
    assert "effects=build_structure=1, delegate_train_army=1" in report
    assert "blockers=no_ready_robo=1" in report


@pytest.mark.unit
def test_format_strategy_outcome_diagnostics_includes_key_sections(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(game_time=100.0),
                strategy_action=3,
                strategy_action_name="TECH_ROBO",
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=160.0,
                    pending_robo=1.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(game_time=220.0),
                done=True,
                result="Result.Tie",
            ),
        ],
    )

    report = format_strategy_outcome_diagnostics(
        diagnose_strategy_outcomes(trajectory),
        show_files=True,
    )

    assert "Strategy outcome diagnostics" in report
    assert "action_outcomes:" in report
    assert "TECH_ROBO" in report
    assert "pending_robo_seen" in report
    assert "filter_change_outcomes:" in report
    assert "results:" in report
    assert "Result.Tie: 1" in report
    assert str(trajectory) in report
