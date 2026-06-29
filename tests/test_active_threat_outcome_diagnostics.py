from __future__ import annotations

import json

import pytest

from rl.active_threat_outcome_diagnostics import diagnose_active_threat_outcomes
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.diagnose_active_threat_outcomes import (
    format_active_threat_outcome_diagnostics,
)


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "minerals": 150.0,
            "vespene": 50.0,
            "supply_left": 8.0,
            "workers": 32.0,
            "own_bases": 2.0,
            "ready_gateways": 3.0,
            "army_count": 8.0,
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
def test_active_threat_outcomes_group_static_filter_contexts(tmp_path) -> None:
    trajectory = tmp_path / "tactic_strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    minerals=45.0,
                    ready_static_defense=1.0,
                    army_count=8.0,
                ),
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                tactic_id="ANTI_AIR_RESPONSE",
                tactic_phase="ATTACK_WINDOW",
                tactic_source="rule",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=130.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    minerals=30.0,
                    ready_static_defense=1.0,
                    army_count=10.0,
                ),
                strategy_action=0,
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=160.0,
                    base_under_threat=0.0,
                    minerals=85.0,
                    ready_static_defense=1.0,
                    army_count=12.0,
                ),
                strategy_action=0,
            ),
            _row(
                step=256,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_air_threat=1.0,
                    minerals=180.0,
                    pending_static_defense=1.0,
                    ready_static_defense=1.0,
                    army_count=12.0,
                ),
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                tactic_id="RECOVERY",
                tactic_phase="RECOVERY",
                tactic_source="rule",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
            ),
            _row(
                step=320,
                strategy_observation=_observation(
                    game_time=250.0,
                    base_under_threat=0.0,
                    minerals=60.0,
                    pending_static_defense=0.0,
                    ready_static_defense=2.0,
                    army_count=12.0,
                ),
                strategy_action=0,
            ),
            _row(
                step=384,
                strategy_observation=_observation(game_time=300.0),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    diagnostics = diagnose_active_threat_outcomes(trajectory)

    assert diagnostics.files == 1
    assert diagnostics.rows == 6
    assert diagnostics.training_rows == 5
    assert diagnostics.active_threat_filter_rows == 2
    by_context = {
        (summary.tactic_id, summary.context): summary
        for summary in diagnostics.context_summaries
    }

    low_minerals = by_context[("ANTI_AIR_RESPONSE", "ready_static_low_minerals")]
    assert low_minerals.after_action == "PRODUCE_ARMY"
    assert low_minerals.count == 1
    assert low_minerals.avg_start_metrics["minerals"] == 45.0
    assert low_minerals.outcomes_by_window["30s"].event_counts["threat_persisted"] == 1
    assert low_minerals.outcomes_by_window["30s"].avg_metrics["army_count_delta"] == 2.0
    assert low_minerals.outcomes_by_window["60s"].event_counts["threat_cleared"] == 1

    pending_with_ready = by_context[("RECOVERY", "pending_static_with_ready")]
    assert pending_with_ready.outcomes_by_window["30s"].event_counts[
        "threat_cleared"
    ] == 1
    assert pending_with_ready.outcomes_by_window["30s"].avg_metrics[
        "static_defense_delta"
    ] == 1.0

    file_summary = diagnostics.file_summaries[0]
    assert file_summary.context_counts == {
        "pending_static_with_ready": 1,
        "ready_static_low_minerals": 1,
    }
    assert file_summary.filter_change_counts == {"BUILD_STATIC_DEFENSE->PRODUCE_ARMY": 2}


@pytest.mark.unit
def test_format_active_threat_outcomes_includes_key_sections(tmp_path) -> None:
    trajectory = tmp_path / "tactic_strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    minerals=40.0,
                ),
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=160.0,
                    base_under_threat=1.0,
                    minerals=120.0,
                ),
                strategy_action=0,
            ),
            _row(
                step=192,
                strategy_observation=_observation(game_time=200.0),
                done=True,
                result="Result.Tie",
            ),
        ],
    )

    report = format_active_threat_outcome_diagnostics(
        diagnose_active_threat_outcomes(trajectory),
        show_files=True,
    )

    assert "Active-threat outcome diagnostics" in report
    assert "active_threat_filter_rows: 1" in report
    assert "TECH_POWER, BUILD_STATIC_DEFENSE -> STAY_COURSE" in report
    assert "no_static_mineral_short" in report
    assert "threat_persisted" in report
    assert "files:" in report
    assert str(trajectory) in report
