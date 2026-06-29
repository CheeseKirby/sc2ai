from __future__ import annotations

import json

import pytest

from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from rl.strategy_timing_diagnostics import diagnose_strategy_timing
from scripts.diagnose_strategy_timing import format_strategy_timing_diagnostics


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(overrides)
    return observation


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_strategy_timing_diagnostics_reports_action_times_and_timeline(tmp_path) -> None:
    trajectory = tmp_path / "hard_terran.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 64,
                "difficulty": "Hard",
                "opponent_race": "Terran",
                "strategy_observation": _observation(game_time=100.0),
                "strategy_action": 2,
                "done": False,
            },
            {
                "step": 128,
                "difficulty": "Hard",
                "opponent_race": "Terran",
                "strategy_observation": _observation(
                    game_time=160.0,
                    enemy_armored_units_known=2.0,
                    pending_robo=1.0,
                ),
                "strategy_action": 3,
                "done": False,
            },
            {
                "step": 192,
                "difficulty": "Hard",
                "opponent_race": "Terran",
                "strategy_observation": _observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    pending_static_defense=1.0,
                ),
                "strategy_action": 5,
                "done": False,
            },
            {
                "step": 256,
                "difficulty": "Hard",
                "opponent_race": "Terran",
                "strategy_observation": _observation(game_time=260.0),
                "strategy_action": 0,
                "done": True,
                "result": "Result.Defeat",
            },
        ],
    )

    diagnostics = diagnose_strategy_timing(trajectory)

    assert diagnostics.files == 1
    assert diagnostics.rows == 4
    assert diagnostics.training_rows == 3
    assert diagnostics.result_counts == {"Result.Defeat": 1}
    assert diagnostics.action_timing_by_name["ADD_GATEWAYS"].first_game_time == 100.0
    assert diagnostics.action_timing_by_name["TECH_ROBO"].avg_game_time == 160.0
    assert diagnostics.threat_action_counts_by_name == {"BUILD_STATIC_DEFENSE": 1}
    assert diagnostics.tech_robo_latency["armored_signal"].delays == [0.0]
    assert diagnostics.pending_repeat_counts_by_name == {
        "TECH_ROBO": 1,
        "BUILD_STATIC_DEFENSE": 1,
    }
    assert diagnostics.file_summaries[0].timeline[0].action_name == "ADD_GATEWAYS"
    assert diagnostics.file_summaries[0].timeline[-1].action_name == "STAY_COURSE"


@pytest.mark.unit
def test_strategy_timing_diagnostics_reports_missing_tech_after_signal(tmp_path) -> None:
    trajectory = tmp_path / "hard_zerg.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 64,
                "difficulty": "Hard",
                "opponent_race": "Zerg",
                "strategy_observation": _observation(
                    game_time=180.0,
                    enemy_cloaked_units_seen=1.0,
                ),
                "strategy_action": 7,
                "done": False,
            },
            {
                "step": 128,
                "difficulty": "Hard",
                "opponent_race": "Zerg",
                "strategy_observation": _observation(game_time=240.0),
                "strategy_action": 5,
                "done": False,
            },
            {
                "step": 192,
                "difficulty": "Hard",
                "opponent_race": "Zerg",
                "strategy_observation": _observation(game_time=260.0),
                "strategy_action": 0,
                "done": True,
                "result": "Result.Defeat",
            },
        ],
    )

    diagnostics = diagnose_strategy_timing(trajectory)

    latency = diagnostics.tech_robo_latency["cloaked_signal"]
    assert latency.files_with_signal == 1
    assert latency.files_with_tech_after_signal == 0
    assert latency.files_without_tech_after_signal == 1
    assert latency.missing_file_paths == [str(trajectory)]
    assert latency.avg_delay == 0.0


@pytest.mark.unit
def test_strategy_timing_diagnostics_distinguishes_tech_before_signal(tmp_path) -> None:
    trajectory = tmp_path / "early_robo.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 64,
                "strategy_observation": _observation(game_time=100.0),
                "strategy_action": 3,
                "done": False,
            },
            {
                "step": 128,
                "strategy_observation": _observation(
                    game_time=180.0,
                    enemy_armored_units_known=1.0,
                ),
                "strategy_action": 0,
                "done": False,
            },
            {
                "step": 192,
                "strategy_observation": _observation(game_time=220.0),
                "strategy_action": 0,
                "done": True,
                "result": "Result.Victory",
            },
        ],
    )

    diagnostics = diagnose_strategy_timing(trajectory)

    latency = diagnostics.tech_robo_latency["armored_signal"]
    assert latency.files_with_signal == 1
    assert latency.files_with_tech_after_signal == 0
    assert latency.files_with_tech_before_signal == 1
    assert latency.files_without_tech == 0
    assert latency.files_without_tech_after_signal == 1
    assert latency.early_leads == [80.0]
    assert latency.early_file_paths == [str(trajectory)]
    assert latency.no_tech_file_paths == []


@pytest.mark.unit
def test_strategy_timing_diagnostics_recognizes_bare_hard_defeat_result(tmp_path) -> None:
    trajectory = tmp_path / "hard_protoss.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 64,
                "difficulty": "Hard",
                "opponent_race": "Protoss",
                "strategy_observation": _observation(game_time=100.0),
                "strategy_action": 0,
                "done": False,
            },
            {
                "step": 128,
                "difficulty": "Hard",
                "opponent_race": "Protoss",
                "strategy_observation": _observation(game_time=140.0),
                "strategy_action": 0,
                "done": True,
                "result": "Defeat",
            },
        ],
    )

    diagnostics = diagnose_strategy_timing(trajectory)

    assert diagnostics.hard_defeat_file_paths == [str(trajectory)]


@pytest.mark.unit
def test_format_strategy_timing_diagnostics_includes_key_sections(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 64,
                "strategy_observation": _observation(game_time=100.0),
                "strategy_action": 1,
                "done": False,
            },
            {
                "step": 128,
                "strategy_observation": _observation(game_time=140.0),
                "strategy_action": 0,
                "done": True,
                "result": "Result.Victory",
            },
        ],
    )

    report = format_strategy_timing_diagnostics(
        diagnose_strategy_timing(trajectory),
        show_files=True,
    )

    assert "Strategy timing diagnostics" in report
    assert "action_timing:" in report
    assert "EXPAND: count=1 first=100.0" in report
    assert "threat_action_counts:" in report
    assert "tech_robo_latency:" in report
    assert "tech_before_signal=" in report
    assert "pending_repeat_counts:" in report
    assert str(trajectory) in report
