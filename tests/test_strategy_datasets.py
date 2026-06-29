from __future__ import annotations

import json

import pytest

from rl.strategy_datasets import load_strategy_trajectory_dataset
from rl.strategy_observations import (
    STRATEGY_OBSERVATION_FIELDS,
    STRATEGY_OBSERVATION_FIELDS_V1,
)


def _strategy_observation(value: float) -> dict[str, float]:
    return {field: value for field in STRATEGY_OBSERVATION_FIELDS}


@pytest.mark.unit
def test_load_strategy_dataset_skips_terminal_by_default(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    rows = [
        {
            "step": 64,
            "strategy_observation": _strategy_observation(1.0),
            "strategy_action": 3,
            "strategy_action_name": "TECH_ROBO",
            "done": False,
        },
        {
            "step": 128,
            "strategy_observation": _strategy_observation(2.0),
            "strategy_action": 0,
            "strategy_action_name": "STAY_COURSE",
            "done": True,
            "result": "Result.Defeat",
        },
    ]
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = load_strategy_trajectory_dataset(trajectory)

    assert dataset.size == 1
    assert dataset.observation_dim == len(STRATEGY_OBSERVATION_FIELDS)
    assert dataset.actions.tolist() == [3]
    assert dataset.action_counts == {3: 1}
    assert dataset.observation_schema_counts == {"strategy_v2": 1}
    assert dataset.rows_defaulted_observation_fields == 0


@pytest.mark.unit
def test_load_strategy_dataset_defaults_v1_rows(tmp_path) -> None:
    trajectory = tmp_path / "strategy_v1.jsonl"
    row = {
        "step": 64,
        "strategy_observation": {
            field: float(index)
            for index, field in enumerate(STRATEGY_OBSERVATION_FIELDS_V1)
        },
        "strategy_action": 1,
        "strategy_action_name": "EXPAND",
        "done": False,
    }
    trajectory.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = load_strategy_trajectory_dataset(trajectory)

    assert dataset.size == 1
    assert dataset.observation_dim == len(STRATEGY_OBSERVATION_FIELDS)
    assert dataset.observation_schema_counts == {"strategy_v1": 1}
    assert dataset.rows_defaulted_observation_fields == 1


@pytest.mark.unit
def test_load_strategy_dataset_can_reject_v1_defaults(tmp_path) -> None:
    trajectory = tmp_path / "strategy_v1.jsonl"
    row = {
        "step": 64,
        "strategy_observation": {field: 1.0 for field in STRATEGY_OBSERVATION_FIELDS_V1},
        "strategy_action": 1,
        "done": False,
    }
    trajectory.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pending_bases"):
        load_strategy_trajectory_dataset(
            trajectory,
            allow_observation_defaults=False,
        )
