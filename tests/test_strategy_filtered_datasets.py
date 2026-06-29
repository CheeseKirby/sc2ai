from __future__ import annotations

import json

import pytest

from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_filtered_datasets import (
    load_signal_filtered_strategy_trajectory_dataset,
)
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS


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
            "ready_gateways": 2.0,
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
def test_trainable_signal_filter_drops_non_executable_and_veto_rows(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
                strategy_observation=_observation(
                    game_time=100.0,
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=150.0,
                    ready_gateways=2.0,
                    pending_gateways=0.0,
                ),
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
            ),
            _row(
                step=192,
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=180.0,
                    minerals=25.0,
                    ready_static_defense=1.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                ),
            ),
            _row(
                step=256,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=10.0,
                    ready_gateways=2.0,
                    gateway_idle_count=1.0,
                    minerals=200.0,
                ),
            ),
            _row(
                step=320,
                strategy_observation=_observation(
                    game_time=340.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    filtered = load_signal_filtered_strategy_trajectory_dataset(
        trajectory,
        filter_name="trainable",
    )

    assert filtered.dataset.size == 2
    assert filtered.dataset.actions.tolist() == [2, 0]
    assert filtered.summary.original_examples == 4
    assert filtered.summary.kept_examples == 2
    assert filtered.summary.removed_examples == 2
    assert filtered.summary.kept_by_training_use == {
        "accept_positive": 1,
        "drop_ambiguous": 1,
    }
    assert filtered.summary.removed_by_training_use == {
        "drop_non_executable": 1,
        "veto_negative": 1,
    }
    assert filtered.summary.kept_action_counts_by_name == {
        "STAY_COURSE": 1,
        "ADD_GATEWAYS": 1,
    }
    assert filtered.summary.removed_action_counts_by_name == {
        "BUILD_STATIC_DEFENSE": 1,
        "PRODUCE_ARMY": 1,
    }


@pytest.mark.unit
def test_strict_positive_signal_filter_keeps_only_payoff_rows(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=3,
                strategy_action_name="TECH_ROBO",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=200.0,
                    vespene=150.0,
                    pending_robo=0.0,
                    ready_robo=0.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=160.0,
                    pending_robo=1.0,
                ),
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
            ),
            _row(done=True, result="Result.Tie"),
        ],
    )

    filtered = load_signal_filtered_strategy_trajectory_dataset(
        trajectory,
        filter_name="strict-positive",
    )

    assert filtered.dataset.size == 1
    assert filtered.dataset.actions.tolist() == [3]
    assert filtered.summary.allowed_training_uses == ("accept_positive",)
    assert filtered.summary.removed_by_training_use == {"drop_ambiguous": 1}


@pytest.mark.unit
def test_signal_filter_rejects_unknown_preset(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown signal filter"):
        load_signal_filtered_strategy_trajectory_dataset(
            tmp_path,
            filter_name="made-up",
        )
