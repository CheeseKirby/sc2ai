from __future__ import annotations

import json

import pytest

from rl.tactic_diagnostics import diagnose_tactics
from scripts.diagnose_tactics import format_tactic_diagnostics


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
        "strategy_observation": {"game_time": 300.0},
        "strategy_action": 6,
        "strategy_action_name": "PRODUCE_ARMY",
        "done": False,
        "tactic_id": "TECH_POWER",
        "tactic_phase": "POWER_SPIKE",
        "tactic_source": "rule",
        "strategy_action_before_tactic_filter": 3,
        "strategy_action_before_tactic_filter_name": "TECH_ROBO",
        "strategy_action_after_tactic_filter": 6,
        "strategy_action_after_tactic_filter_name": "PRODUCE_ARMY",
    }
    row.update(overrides)
    return row


@pytest.mark.unit
def test_tactic_diagnostics_counts_tactics_and_filter_changes(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(step=64, strategy_observation={"game_time": 300.0}),
            _row(
                step=128,
                strategy_observation={"game_time": 340.0},
                strategy_action_before_tactic_filter=6,
                strategy_action_before_tactic_filter_name="PRODUCE_ARMY",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
            ),
            _row(
                step=192,
                opponent_ai_build="Macro",
                strategy_observation={"game_time": 380.0},
                tactic_id="GATEWAY_PRESSURE",
                tactic_phase="ATTACK_WINDOW",
                strategy_action_before_tactic_filter=4,
                strategy_action_before_tactic_filter_name="FORGE_UPGRADES",
                strategy_action_after_tactic_filter=2,
                strategy_action_after_tactic_filter_name="ADD_GATEWAYS",
            ),
            _row(
                step=256,
                strategy_observation={"game_time": 420.0},
                done=True,
                result="Result.Victory",
            ),
        ],
    )

    diagnostics = diagnose_tactics(trajectory)

    assert diagnostics.files == 1
    assert diagnostics.rows == 4
    assert diagnostics.training_rows == 3
    assert diagnostics.rows_with_tactic_metadata == 4
    assert diagnostics.rows_with_filter_metadata == 4
    assert diagnostics.filter_change_rows == 3
    assert diagnostics.training_rows_with_tactic_metadata == 3
    assert diagnostics.training_rows_with_filter_metadata == 3
    assert diagnostics.training_filter_change_rows == 2
    assert diagnostics.opponent_ai_build_counts == {"Macro": 1, "Power": 2}
    assert diagnostics.tactic_counts == {"GATEWAY_PRESSURE": 1, "TECH_POWER": 2}
    assert diagnostics.tactic_phase_counts == {
        "ATTACK_WINDOW": 1,
        "POWER_SPIKE": 2,
    }
    assert diagnostics.tactic_source_counts == {"rule": 3}
    assert diagnostics.result_counts == {"Result.Victory": 1}
    assert [
        (
            change.opponent_ai_build,
            change.tactic_id,
            change.before_action,
            change.after_action,
            change.count,
        )
        for change in diagnostics.filter_changes
    ] == [
        ("Macro", "GATEWAY_PRESSURE", "FORGE_UPGRADES", "ADD_GATEWAYS", 1),
        ("Power", "TECH_POWER", "TECH_ROBO", "PRODUCE_ARMY", 1),
    ]
    assert diagnostics.file_summaries[0].timeline[0].tactic_id == "TECH_POWER"
    assert diagnostics.file_summaries[0].timeline[-1].tactic_id == "GATEWAY_PRESSURE"
    first_filter_event = diagnostics.file_summaries[0].filter_timeline[0]
    assert first_filter_event.changed is True
    assert first_filter_event.original_action == "TECH_ROBO"
    assert first_filter_event.selected_action == "PRODUCE_ARMY"
    assert first_filter_event.game_time == 300.0
    assert first_filter_event.minerals == 0.0


@pytest.mark.unit
def test_tactic_diagnostics_handles_missing_metadata(tmp_path) -> None:
    trajectory = tmp_path / "legacy_strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 64,
                "strategy_observation": {"game_time": 100.0},
                "strategy_action": 0,
                "done": False,
            },
            {
                "step": 128,
                "strategy_observation": {"game_time": 140.0},
                "strategy_action": 0,
                "done": True,
                "result": "Tie",
            },
        ],
    )

    diagnostics = diagnose_tactics(trajectory)

    assert diagnostics.rows == 2
    assert diagnostics.training_rows == 1
    assert diagnostics.rows_with_tactic_metadata == 0
    assert diagnostics.rows_with_filter_metadata == 0
    assert diagnostics.filter_change_rows == 0
    assert diagnostics.training_rows_with_tactic_metadata == 0
    assert diagnostics.training_rows_with_filter_metadata == 0
    assert diagnostics.training_filter_change_rows == 0
    assert diagnostics.opponent_ai_build_counts == {"RandomBuild": 1}
    assert diagnostics.tactic_counts == {}
    assert diagnostics.filter_changes == []
    assert diagnostics.file_summaries[0].timeline[0].tactic_id == "<none>"
    assert diagnostics.file_summaries[0].filter_timeline == []


@pytest.mark.unit
def test_tactic_diagnostics_records_filter_timeline_resources(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation={
                    "game_time": 102.9,
                    "minerals": 290.0,
                    "vespene": 68.0,
                    "pending_gateways": 2.0,
                    "ready_gateways": 0.0,
                    "base_under_threat": 0.0,
                    "gateway_idle_count": 0.0,
                    "robo_idle_count": 0.0,
                },
                tactic_id="SAFE_MACRO",
                tactic_phase="OPENING",
                strategy_action=7,
                strategy_action_name="BOOST_WORKERS",
                strategy_action_before_tactic_filter=2,
                strategy_action_before_tactic_filter_name="ADD_GATEWAYS",
                strategy_action_after_tactic_filter=7,
                strategy_action_after_tactic_filter_name="BOOST_WORKERS",
            ),
            _row(
                step=128,
                strategy_observation={
                    "game_time": 571.4,
                    "minerals": 175.0,
                    "vespene": 1853.0,
                    "pending_gateways": 0.0,
                    "ready_gateways": 0.0,
                    "base_under_threat": 0.0,
                    "gateway_idle_count": 0.0,
                    "robo_idle_count": 0.0,
                },
                tactic_id="RECOVERY",
                tactic_phase="RECOVERY",
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
                strategy_action_before_tactic_filter=2,
                strategy_action_before_tactic_filter_name="ADD_GATEWAYS",
                strategy_action_after_tactic_filter=2,
                strategy_action_after_tactic_filter_name="ADD_GATEWAYS",
            ),
            _row(
                step=192,
                strategy_observation={"game_time": 600.0},
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    diagnostics = diagnose_tactics(trajectory)

    timeline = diagnostics.file_summaries[0].filter_timeline
    assert [(event.original_action, event.selected_action, event.changed) for event in timeline] == [
        ("ADD_GATEWAYS", "BOOST_WORKERS", True),
        ("ADD_GATEWAYS", "ADD_GATEWAYS", False),
    ]
    first = timeline[0]
    assert first.tactic_id == "SAFE_MACRO"
    assert first.game_time == 102.9
    assert first.minerals == 290.0
    assert first.pending_gateways == 2.0
    assert first.ready_gateways == 0.0
    assert first.base_under_threat == 0.0
    assert first.gateway_idle_count == 0.0


@pytest.mark.unit
def test_format_tactic_diagnostics_includes_key_sections(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(step=64, strategy_observation={"game_time": 300.0}),
            _row(step=128, done=True, result="Result.Victory"),
        ],
    )

    report = format_tactic_diagnostics(
        diagnose_tactics(trajectory),
        show_files=True,
        show_filter_timeline=True,
    )

    assert "Tactic diagnostics" in report
    assert "rows_with_tactic_metadata: 2" in report
    assert "training_rows_with_tactic_metadata: 1" in report
    assert "training_filter_change_rows: 1" in report
    assert "opponent_ai_builds:" in report
    assert "Power: 1" in report
    assert "tactic_counts:" in report
    assert "TECH_POWER: 1" in report
    assert "filter_changes:" in report
    assert "Power, TECH_POWER, TECH_ROBO -> PRODUCE_ARMY: 1" in report
    assert "results:" in report
    assert "Result.Victory: 1" in report
    assert str(trajectory) in report
    assert "timeline:" in report
    assert "filter_timeline:" in report
    assert "TECH_ROBO -> PRODUCE_ARMY" in report
    assert "minerals=0.0" in report
