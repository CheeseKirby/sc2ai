from __future__ import annotations

import json

import pytest

from rl.power_tactic_diagnostics import diagnose_power_tactics
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from scripts.diagnose_power_tactics import format_power_tactic_diagnostics


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(
        {
            "supply_cap": 80.0,
            "supply_left": 10.0,
            "workers": 42.0,
            "own_bases": 2.0,
            "worker_saturation_ratio": 0.95,
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
        "episode_id": "episode",
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
def test_power_tactic_diagnostics_reports_power_signal_timings(tmp_path) -> None:
    trajectory = tmp_path / "coverage_power.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=120.0,
                    vespene=20.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    army_count=7.0,
                    worker_saturation_ratio=0.75,
                ),
            ),
            _row(
                step=128,
                strategy_action=3,
                strategy_action_name="TECH_ROBO",
                strategy_observation=_observation(
                    game_time=160.0,
                    minerals=550.0,
                    vespene=180.0,
                    pending_robo=1.0,
                    ready_gateways=3.0,
                    gateway_idle_count=2.0,
                    army_count=8.0,
                ),
            ),
            _row(
                step=192,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=220.0,
                    minerals=620.0,
                    vespene=340.0,
                    ready_robo=1.0,
                    observers=1.0,
                    immortals=1.0,
                    pending_forge=1.0,
                    ready_static_defense=1.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    ready_gateways=3.0,
                    gateway_idle_count=2.0,
                    army_count=12.0,
                ),
            ),
            _row(
                step=256,
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=260.0,
                    minerals=450.0,
                    vespene=360.0,
                    ready_robo=1.0,
                    ready_forge=1.0,
                    ground_weapon_upgrade_pending=1.0,
                    pending_static_defense=1.0,
                    army_count=16.0,
                ),
            ),
            _row(
                step=320,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=320.0,
                    minerals=300.0,
                    vespene=120.0,
                    ground_weapon_level=1.0,
                    army_count=20.0,
                ),
            ),
            _row(
                step=384,
                done=True,
                result="Result.Victory",
                strategy_observation=_observation(game_time=340.0),
            ),
        ],
    )

    diagnostics = diagnose_power_tactics(trajectory)
    summary = diagnostics.file_summaries[0]

    assert diagnostics.files == 1
    assert diagnostics.rows == 6
    assert diagnostics.training_rows == 5
    assert diagnostics.result_counts == {"Result.Victory": 1}
    assert summary.signals.first_tech_robo_action_time == 160.0
    assert summary.signals.first_pending_robo_time == 160.0
    assert summary.signals.first_ready_robo_time == 220.0
    assert summary.signals.first_observer_time == 220.0
    assert summary.signals.first_immortal_time == 220.0
    assert summary.signals.first_pending_forge_time == 220.0
    assert summary.signals.first_ready_forge_time == 260.0
    assert summary.signals.first_ground_upgrade_pending_time == 260.0
    assert summary.signals.first_ground_upgrade_complete_time == 320.0
    assert summary.signals.first_pending_static_defense_time == 260.0
    assert summary.signals.first_ready_static_defense_time == 220.0
    assert summary.signals.first_produce_army_action_time == 220.0
    assert summary.signals.army_count_first_reached == {
        ">=8": 160.0,
        ">=12": 220.0,
        ">=16": 260.0,
        ">=20": 320.0,
    }
    assert summary.threat.threat_action_counts_by_name == {"PRODUCE_ARMY": 1}
    assert summary.economy.mineral_bank_rows_ge_500 == 2
    assert summary.economy.vespene_bank_rows_ge_300 == 2
    assert summary.economy.dual_bank_rows_ge_500_300 == 1
    assert summary.economy.low_worker_saturation_rows_lt_0_8 == 1
    assert summary.gateways.gateway_idle_rows == 3
    assert summary.gateways.idle_gateway_bank_rows_ge_150_minerals == 2


@pytest.mark.unit
def test_power_tactic_diagnostics_counts_filter_counterfactuals(tmp_path) -> None:
    trajectory = tmp_path / "tactic_power.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(game_time=100.0),
                tactic_id="SAFE_MACRO",
                tactic_phase="OPENING",
                tactic_source="rule",
                strategy_action_before_tactic_filter=0,
                strategy_action_before_tactic_filter_name="STAY_COURSE",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
            ),
            _row(
                step=128,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(game_time=180.0, pending_robo=1.0),
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=3,
                strategy_action_before_tactic_filter_name="TECH_ROBO",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
            ),
            _row(
                step=192,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=240.0,
                    base_under_threat=1.0,
                    minerals=650.0,
                    vespene=360.0,
                    gateway_idle_count=2.0,
                ),
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
            ),
            _row(
                step=256,
                done=True,
                result="Result.Tie",
                strategy_observation=_observation(game_time=280.0),
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=6,
                strategy_action_before_tactic_filter_name="PRODUCE_ARMY",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
            ),
        ],
    )

    diagnostics = diagnose_power_tactics(trajectory)
    summary = diagnostics.file_summaries[0]

    assert diagnostics.rows_with_tactic_metadata == 4
    assert diagnostics.rows_with_filter_metadata == 4
    assert diagnostics.filter_change_rows == 2
    assert diagnostics.training_rows_with_tactic_metadata == 3
    assert diagnostics.training_rows_with_filter_metadata == 3
    assert diagnostics.training_filter_change_rows == 2
    assert diagnostics.before_filter_action_counts_by_name == {
        "BUILD_STATIC_DEFENSE": 1,
        "STAY_COURSE": 1,
        "TECH_ROBO": 1,
    }
    assert diagnostics.after_filter_action_counts_by_name == {
        "PRODUCE_ARMY": 2,
        "STAY_COURSE": 1,
    }
    assert diagnostics.filter_action_delta_by_name == {
        "BUILD_STATIC_DEFENSE": -1,
        "PRODUCE_ARMY": 2,
        "TECH_ROBO": -1,
    }
    assert [
        (change.tactic_id, change.before_action, change.after_action, change.count)
        for change in diagnostics.filter_changes
    ] == [
        ("TECH_POWER", "BUILD_STATIC_DEFENSE", "PRODUCE_ARMY", 1),
        ("TECH_POWER", "TECH_ROBO", "PRODUCE_ARMY", 1),
    ]
    assert summary.tactic_counts == {"SAFE_MACRO": 1, "TECH_POWER": 2}
    assert summary.tactic_timeline[0].tactic_id == "SAFE_MACRO"
    assert summary.tactic_timeline[-1].tactic_id == "TECH_POWER"
    assert summary.threat.produce_army_under_threat_rows == 1


@pytest.mark.unit
def test_power_tactic_diagnostics_classifies_robo_banking_filter_context(tmp_path) -> None:
    trajectory = tmp_path / "tactic_robo_banking.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    game_time=100.0,
                    has_cybernetics_core=1.0,
                    minerals=125.0,
                    vespene=120.0,
                ),
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=3,
                strategy_action_before_tactic_filter_name="TECH_ROBO",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=160.0,
                    has_cybernetics_core=1.0,
                    minerals=180.0,
                    vespene=120.0,
                ),
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=3,
                strategy_action_before_tactic_filter_name="TECH_ROBO",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=220.0,
                    has_cybernetics_core=1.0,
                    minerals=400.0,
                    vespene=200.0,
                    pending_robo=1.0,
                ),
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=3,
                strategy_action_before_tactic_filter_name="TECH_ROBO",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
            ),
            _row(
                step=256,
                strategy_observation=_observation(
                    game_time=280.0,
                    has_cybernetics_core=1.0,
                    minerals=400.0,
                    vespene=200.0,
                    ready_robo=1.0,
                ),
                tactic_id="ROBO_TIMING",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=3,
                strategy_action_before_tactic_filter_name="TECH_ROBO",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
            ),
            _row(
                step=320,
                done=True,
                result="Result.Defeat",
                strategy_observation=_observation(game_time=340.0),
            ),
        ],
    )

    diagnostics = diagnose_power_tactics(trajectory)
    summary = diagnostics.file_summaries[0]

    assert [
        (item.tactic_id, item.context, item.count)
        for item in diagnostics.robo_banking_filter_contexts
    ] == [
        ("TECH_POWER", "first_robo_affordable", 1),
        ("TECH_POWER", "first_robo_mineral_short", 1),
        ("TECH_POWER", "pending_robo_cap", 1),
        ("ROBO_TIMING", "ready_robo_already_exists", 1),
    ]
    assert [
        (item.tactic_id, item.context, item.count)
        for item in summary.robo_banking_filter_contexts
    ] == [
        ("TECH_POWER", "first_robo_affordable", 1),
        ("TECH_POWER", "first_robo_mineral_short", 1),
        ("TECH_POWER", "pending_robo_cap", 1),
        ("ROBO_TIMING", "ready_robo_already_exists", 1),
    ]

    report = format_power_tactic_diagnostics(diagnostics, show_files=True)

    assert "robo_banking_filter_context:" in report
    assert "TECH_POWER, first_robo_affordable: 1" in report
    assert "TECH_POWER, first_robo_mineral_short: 1" in report
    assert "TECH_POWER/first_robo_affordable=1" in report


@pytest.mark.unit
def test_power_tactic_diagnostics_classifies_static_defense_filter_context(
    tmp_path,
) -> None:
    trajectory = tmp_path / "tactic_static_defense.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_observation=_observation(
                    game_time=100.0,
                    base_under_threat=1.0,
                    minerals=140.0,
                ),
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
                    game_time=160.0,
                    base_under_threat=1.0,
                    minerals=60.0,
                ),
                tactic_id="ANTI_AIR_RESPONSE",
                tactic_phase="ATTACK_WINDOW",
                tactic_source="rule",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=0,
                strategy_action_after_tactic_filter_name="STAY_COURSE",
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    minerals=220.0,
                    pending_static_defense=1.0,
                ),
                tactic_id="ANTI_AIR_RESPONSE",
                tactic_phase="ATTACK_WINDOW",
                tactic_source="rule",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
            ),
            _row(
                step=256,
                strategy_observation=_observation(
                    game_time=280.0,
                    base_under_threat=1.0,
                    minerals=45.0,
                    ready_static_defense=1.0,
                ),
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
                    game_time=340.0,
                    base_under_threat=0.0,
                    minerals=40.0,
                ),
                tactic_id="RECOVERY",
                tactic_phase="RECOVERY",
                tactic_source="rule",
                strategy_action_before_tactic_filter=5,
                strategy_action_before_tactic_filter_name="BUILD_STATIC_DEFENSE",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
            ),
            _row(
                step=384,
                done=True,
                result="Result.Defeat",
                strategy_observation=_observation(game_time=400.0),
            ),
        ],
    )

    diagnostics = diagnose_power_tactics(trajectory)
    summary = diagnostics.file_summaries[0]

    assert [
        (item.tactic_id, item.after_action, item.context, item.count)
        for item in diagnostics.static_defense_filter_contexts
    ] == [
        ("ANTI_AIR_RESPONSE", "PRODUCE_ARMY", "no_static_affordable", 1),
        ("ANTI_AIR_RESPONSE", "STAY_COURSE", "no_static_mineral_short", 1),
        ("ANTI_AIR_RESPONSE", "PRODUCE_ARMY", "pending_static_waiting", 1),
        ("RECOVERY", "PRODUCE_ARMY", "ready_static_low_minerals", 1),
    ]
    assert [
        (item.tactic_id, item.after_action, item.context, item.count)
        for item in summary.static_defense_filter_contexts
    ] == [
        ("ANTI_AIR_RESPONSE", "PRODUCE_ARMY", "no_static_affordable", 1),
        ("ANTI_AIR_RESPONSE", "STAY_COURSE", "no_static_mineral_short", 1),
        ("ANTI_AIR_RESPONSE", "PRODUCE_ARMY", "pending_static_waiting", 1),
        ("RECOVERY", "PRODUCE_ARMY", "ready_static_low_minerals", 1),
    ]

    report = format_power_tactic_diagnostics(diagnostics, show_files=True)

    assert "static_defense_filter_context:" in report
    assert "ANTI_AIR_RESPONSE, PRODUCE_ARMY, no_static_affordable: 1" in report
    assert "RECOVERY, PRODUCE_ARMY, ready_static_low_minerals: 1" in report
    assert "ANTI_AIR_RESPONSE/no_static_affordable->PRODUCE_ARMY=1" in report


@pytest.mark.unit
def test_format_power_tactic_diagnostics_includes_key_sections(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(game_time=100.0, army_count=8.0),
                tactic_id="TECH_POWER",
                tactic_phase="POWER_SPIKE",
                tactic_source="rule",
                strategy_action_before_tactic_filter=3,
                strategy_action_before_tactic_filter_name="TECH_ROBO",
                strategy_action_after_tactic_filter=6,
                strategy_action_after_tactic_filter_name="PRODUCE_ARMY",
            ),
            _row(
                step=128,
                done=True,
                result="Result.Defeat",
                strategy_observation=_observation(game_time=140.0),
            ),
        ],
    )

    report = format_power_tactic_diagnostics(
        diagnose_power_tactics(trajectory),
        show_files=True,
    )

    assert "Power tactic diagnostics" in report
    assert "results:" in report
    assert "Result.Defeat: 1" in report
    assert "opponent_ai_builds:" in report
    assert "Power: 1" in report
    assert "action_timing:" in report
    assert "PRODUCE_ARMY: count=1 first=100.0" in report
    assert "filter_metadata:" in report
    assert "training_filter_change_rows: 1" in report
    assert "filter_changes:" in report
    assert "Power, TECH_POWER, TECH_ROBO -> PRODUCE_ARMY: 1" in report
    assert "counterfactual_filter_delta:" in report
    assert "TECH_ROBO: -1" in report
    assert str(trajectory) in report
    assert "tech_robo:" in report
    assert "economy:" in report
    assert "gateways:" in report
    assert "tactic_timeline:" in report
