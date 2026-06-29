from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from rl.strategy_replay_candidate import diagnose_strategy_replay_candidate
from scripts.diagnose_strategy_replay_candidate import (
    format_strategy_replay_candidate_diagnostics,
)


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "game_time": 100.0,
            "minerals": 150.0,
            "vespene": 100.0,
            "supply_left": 8.0,
            "workers": 30.0,
            "own_bases": 2.0,
            "ready_gateways": 3.0,
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
def test_replay_candidate_diagnostics_summarizes_changed_rows(tmp_path) -> None:
    trajectory_dir = tmp_path / "strategy"
    _write_jsonl(
        trajectory_dir / "001.jsonl",
        [
            _row(
                step=64,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                tactic_id="RECOVERY",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    minerals=120.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=130.0,
                    base_under_threat=0.0,
                    ready_static_defense=1.0,
                    army_count=11.0,
                ),
            ),
            _row(
                step=192,
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                tactic_id="TECH_POWER",
                strategy_action_before_tactic_filter=3,
                strategy_action_before_tactic_filter_name="TECH_ROBO",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=160.0,
                    base_under_threat=0.0,
                    minerals=80.0,
                    vespene=150.0,
                    pending_robo=0.0,
                    ready_robo=0.0,
                ),
            ),
            _row(
                step=256,
                strategy_observation=_observation(
                    game_time=220.0,
                    pending_robo=1.0,
                    ready_robo=0.0,
                ),
            ),
            _row(
                step=320,
                strategy_observation=_observation(game_time=260.0),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    diagnostics = diagnose_strategy_replay_candidate(trajectory_dir)

    assert diagnostics.candidate_source == "before_filter"
    assert diagnostics.gate_decision.recommendation == "hold_runtime_patch"
    assert diagnostics.gate_decision.runtime_patch_candidate is False
    assert "candidate_executability_low" in diagnostics.gate_decision.blocking_reasons
    assert diagnostics.files == 1
    assert diagnostics.rows == 5
    assert diagnostics.training_rows == 4
    assert diagnostics.candidate_rows == 2
    assert diagnostics.changed_rows == 2
    assert diagnostics.immediate_candidate_executable_rows == 1
    assert diagnostics.action_delta_by_name == {
        "BUILD_STATIC_DEFENSE": 1,
        "PRODUCE_ARMY": -1,
        "STAY_COURSE": -1,
        "TECH_ROBO": 1,
    }

    by_pair = {
        (summary.recorded_action, summary.candidate_action): summary
        for summary in diagnostics.group_summaries
    }
    static_summary = by_pair[("PRODUCE_ARMY", "BUILD_STATIC_DEFENSE")]
    assert static_summary.count == 1
    assert static_summary.immediate_candidate_executable_rows == 1
    assert static_summary.outcomes_by_window["30s"].event_counts["threat_cleared"] == 1
    assert (
        static_summary.outcomes_by_window["30s"].event_counts[
            "static_defense_increased"
        ]
        == 1
    )

    robo_summary = by_pair[("STAY_COURSE", "TECH_ROBO")]
    assert robo_summary.count == 1
    assert robo_summary.immediate_candidate_executable_rows == 0
    assert robo_summary.outcomes_by_window["60s"].event_counts["pending_robo_seen"] == 1

    file_summary = diagnostics.file_summaries[0]
    assert file_summary.changed_rows == 2
    assert [event.candidate_action for event in file_summary.timeline_events] == [
        "BUILD_STATIC_DEFENSE",
        "TECH_ROBO",
    ]
    assert file_summary.timeline_events[0].immediate_candidate_executable is True
    assert file_summary.timeline_events[1].candidate_blocker == "cannot_afford_robo"
    assert asdict(diagnostics)["changed_rows"] == 2


@pytest.mark.unit
def test_replay_candidate_counts_unchanged_candidates_separately(tmp_path) -> None:
    trajectory_dir = tmp_path / "strategy"
    _write_jsonl(
        trajectory_dir / "001.jsonl",
        [
            _row(
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                strategy_action_before_tactic_filter=0,
                strategy_action_before_tactic_filter_name="STAY_COURSE",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
            ),
            _row(done=True, result="Result.Victory"),
        ],
    )

    diagnostics = diagnose_strategy_replay_candidate(trajectory_dir)

    assert diagnostics.gate_decision.recommendation == "no_candidate_changes"
    assert diagnostics.gate_decision.runtime_patch_candidate is False
    assert diagnostics.candidate_rows == 1
    assert diagnostics.changed_rows == 0
    assert diagnostics.action_delta_by_name == {}
    assert diagnostics.group_summaries == []
    assert diagnostics.file_summaries[0].candidate_rows == 1
    assert diagnostics.file_summaries[0].changed_rows == 0


@pytest.mark.unit
def test_format_replay_candidate_diagnostics_contains_gate_sections(tmp_path) -> None:
    trajectory_dir = tmp_path / "strategy"
    _write_jsonl(
        trajectory_dir / "001.jsonl",
        [
            _row(
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                tactic_id="TECH_POWER",
                strategy_action_before_tactic_filter=3,
                strategy_action_before_tactic_filter_name="TECH_ROBO",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=80.0,
                    vespene=150.0,
                ),
            ),
            _row(
                strategy_observation=_observation(
                    game_time=160.0,
                    pending_robo=1.0,
                ),
            ),
            _row(
                strategy_observation=_observation(game_time=220.0),
                done=True,
                result="Result.Tie",
            ),
        ],
    )

    report = format_strategy_replay_candidate_diagnostics(
        diagnose_strategy_replay_candidate(trajectory_dir),
        show_files=True,
    )

    assert "Strategy replay candidate diagnostics" in report
    assert "candidate_source: before_filter" in report
    assert "gate_decision: hold_runtime_patch" in report
    assert "runtime_patch_candidate: false" in report
    assert "gate_blocking_reasons: candidate_executability_low" in report
    assert "changed_rows: 1" in report
    assert "candidate_executable=0/1" in report
    assert "action_delta: STAY_COURSE=-1, TECH_ROBO=1" in report
    assert "candidate=TECH_ROBO" in report
    assert "pending_robo_seen=1" in report


@pytest.mark.unit
def test_replay_candidate_gate_recommends_narrow_patch_review(tmp_path) -> None:
    trajectory_dir = tmp_path / "strategy"
    _write_jsonl(
        trajectory_dir / "001.jsonl",
        [
            _row(
                step=64,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                tactic_id="RECOVERY",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    minerals=150.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                ),
            ),
            _row(
                step=128,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                tactic_id="RECOVERY",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=130.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    minerals=180.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                ),
            ),
            _row(done=True, result="Result.Victory"),
        ],
    )

    diagnostics = diagnose_strategy_replay_candidate(trajectory_dir)

    assert diagnostics.changed_rows == 2
    assert diagnostics.immediate_candidate_executable_rows == 2
    assert diagnostics.gate_decision.recommendation == "review_narrow_runtime_patch"
    assert diagnostics.gate_decision.runtime_patch_candidate is True
    assert diagnostics.gate_decision.blocking_reasons == []
    assert diagnostics.gate_decision.executable_ratio == 1.0
    assert diagnostics.gate_decision.largest_group_count == 2
    assert diagnostics.gate_decision.largest_group_executable_ratio == 1.0
