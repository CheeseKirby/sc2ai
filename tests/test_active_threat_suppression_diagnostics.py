from __future__ import annotations

import json

import pytest

from rl.active_threat_suppression_diagnostics import (
    diagnose_active_threat_suppression,
)
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.diagnose_active_threat_suppression import (
    format_active_threat_suppression_diagnostics,
)


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "game_time": 100.0,
            "minerals": 150.0,
            "vespene": 200.0,
            "supply_left": 8.0,
            "workers": 32.0,
            "own_bases": 2.0,
            "ready_gateways": 4.0,
            "army_count": 10.0,
            "has_cybernetics_core": 1.0,
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
def test_suppression_diagnostics_group_outcomes_and_candidate_replay(tmp_path) -> None:
    no_filter_dir = tmp_path / "no_filter"
    tactic_dir = tmp_path / "tactic"
    _write_jsonl(
        no_filter_dir / "nofilter.jsonl",
        [
            _row(
                step=64,
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    minerals=150.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(game_time=130.0),
                done=True,
                result="Victory",
            ),
        ],
    )
    _write_jsonl(
        tactic_dir / "tactic.jsonl",
        [
            _row(
                step=64,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                tactic_id="RECOVERY",
                tactic_phase="RECOVERY",
                tactic_source="rule",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    minerals=40.0,
                    ready_static_defense=1.0,
                    army_count=10.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=130.0,
                    base_under_threat=1.0,
                    minerals=90.0,
                    ready_static_defense=1.0,
                    army_count=8.0,
                ),
            ),
            _row(
                step=192,
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=3,
                strategy_action_before_tactic_filter_name="TECH_ROBO",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=160.0,
                    base_under_threat=0.0,
                    minerals=80.0,
                    vespene=180.0,
                    pending_robo=0.0,
                    ready_robo=0.0,
                    army_count=8.0,
                ),
            ),
            _row(
                step=256,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=0.0,
                    minerals=160.0,
                    vespene=220.0,
                    pending_robo=1.0,
                    army_count=12.0,
                ),
            ),
            _row(
                step=320,
                strategy_observation=_observation(game_time=260.0),
                done=True,
                result="Defeat",
            ),
        ],
    )

    diagnostics = diagnose_active_threat_suppression([no_filter_dir, tactic_dir])

    assert diagnostics.files == 2
    assert diagnostics.rows == 7
    assert diagnostics.target_suppression_rows == 2
    assert diagnostics.source_summaries[0].target_suppression_rows == 0
    assert diagnostics.source_summaries[1].target_suppression_rows == 2

    by_key = {
        (summary.tactic_id, summary.before_action, summary.after_action, summary.context): summary
        for summary in diagnostics.context_summaries
    }
    static_summary = by_key[
        (
            "RECOVERY",
            "BUILD_STATIC_DEFENSE",
            "PRODUCE_ARMY",
            "ready_static_low_minerals",
        )
    ]
    assert static_summary.threat_state == "ground_threat"
    assert static_summary.count == 1
    assert static_summary.candidate_action == "BUILD_STATIC_DEFENSE"
    assert static_summary.immediate_candidate_executable_rows == 0
    assert static_summary.outcomes_by_window["30s"].event_counts["threat_persisted"] == 1
    assert static_summary.outcomes_by_window["30s"].avg_metrics["army_count_delta"] == -2.0

    robo_summary = by_key[
        (
            "TECH_POWER",
            "TECH_ROBO",
            "STAY_COURSE",
            "first_robo_mineral_short",
        )
    ]
    assert robo_summary.threat_state == "no_threat"
    assert robo_summary.outcomes_by_window["60s"].event_counts["pending_robo_seen"] == 1

    impact = diagnostics.replay_candidate_impact
    assert impact.affected_rows == 2
    assert impact.action_delta_by_name == {
        "BUILD_STATIC_DEFENSE": 1,
        "PRODUCE_ARMY": -1,
        "STAY_COURSE": -1,
        "TECH_ROBO": 1,
    }
    assert impact.immediate_candidate_executable_rows == 0

    tactic_file = diagnostics.file_summaries[1]
    assert tactic_file.target_suppression_rows == 2
    assert [event.candidate_action for event in tactic_file.timeline_events] == [
        "BUILD_STATIC_DEFENSE",
        "TECH_ROBO",
    ]
    assert tactic_file.timeline_events[0].outcomes_by_window["30s"].events[
        "threat_persisted"
    ]


@pytest.mark.unit
def test_format_suppression_diagnostics_includes_timeline_and_candidate(tmp_path) -> None:
    tactic_dir = tmp_path / "tactic"
    _write_jsonl(
        tactic_dir / "tactic.jsonl",
        [
            _row(
                step=64,
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    minerals=40.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=160.0,
                    base_under_threat=0.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(game_time=200.0),
                done=True,
                result="Tie",
            ),
        ],
    )

    report = format_active_threat_suppression_diagnostics(
        diagnose_active_threat_suppression(tactic_dir),
        show_files=True,
        show_timeline=True,
    )

    assert "Active-threat suppression diagnostics" in report
    assert "target_suppression_rows: 1" in report
    assert "TECH_POWER, BUILD_STATIC_DEFENSE -> STAY_COURSE" in report
    assert "replay_candidate_impact:" in report
    assert "action_delta: BUILD_STATIC_DEFENSE=1, STAY_COURSE=-1" in report
    assert "timeline:" in report
    assert "candidate=BUILD_STATIC_DEFENSE" in report
