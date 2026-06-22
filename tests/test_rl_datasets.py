from __future__ import annotations

import json

import pytest

from rl.datasets import discover_trajectory_files, load_trajectory_dataset
from rl.observations import (
    OBSERVATION_FIELDS,
    OBSERVATION_FIELDS_V1,
    OBSERVATION_FIELDS_V2,
)


def _observation(value: float) -> dict[str, float]:
    return {field: value for field in OBSERVATION_FIELDS}


def _v1_observation(value: float) -> dict[str, float]:
    return {field: value for field in OBSERVATION_FIELDS_V1}


def _v2_observation(value: float) -> dict[str, float]:
    return {field: value for field in OBSERVATION_FIELDS_V2}


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_discover_trajectory_files_expands_directories(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "a.jsonl"
    second = nested / "b.jsonl"
    ignored = nested / "notes.txt"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    ignored.write_text("", encoding="utf-8")

    assert discover_trajectory_files([tmp_path]) == [first, second]


@pytest.mark.unit
def test_discover_trajectory_files_accepts_single_directory_path(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text("", encoding="utf-8")

    assert discover_trajectory_files(tmp_path) == [trajectory]


@pytest.mark.unit
def test_load_trajectory_dataset_skips_terminal_by_default(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 8,
                "observation": _observation(1.0),
                "action": 0,
                "done": False,
                "result": None,
            },
            {
                "step": 16,
                "observation": _observation(2.0),
                "action": 1,
                "done": True,
                "result": "Victory",
            },
        ],
    )

    dataset = load_trajectory_dataset([trajectory])

    assert dataset.size == 1
    assert dataset.observation_dim == len(OBSERVATION_FIELDS)
    assert dataset.observations.shape == (1, len(OBSERVATION_FIELDS))
    assert dataset.actions.tolist() == [0]
    assert dataset.action_counts == {0: 1}
    assert dataset.observation_schema_counts == {"3": 1}
    assert dataset.rows_defaulted_observation_fields == 0


@pytest.mark.unit
def test_load_trajectory_dataset_accepts_single_file_path(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 8,
                "observation": _observation(1.0),
                "action": 0,
                "done": False,
            },
        ],
    )

    dataset = load_trajectory_dataset(trajectory)

    assert dataset.size == 1
    assert dataset.actions.tolist() == [0]


@pytest.mark.unit
def test_load_trajectory_dataset_defaults_v1_observations_to_current_schema(
    tmp_path,
) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 8,
                "observation": _v1_observation(1.0),
                "action": 0,
                "done": False,
            },
        ],
    )

    dataset = load_trajectory_dataset(trajectory)

    assert dataset.size == 1
    assert dataset.observations.shape == (1, len(OBSERVATION_FIELDS))
    assert dataset.examples[0].observation_schema_version == 1
    assert dataset.examples[0].defaulted_observation_fields == (
        "base_under_threat",
        "enemy_to_home_distance",
        "army_idle_count",
        "army_busy_count",
        "attack_army_peak",
        "army_lost_from_peak",
        "army_lost_from_peak_ratio",
        "army_count_delta",
    )
    assert dataset.observation_schema_counts == {"1": 1}
    assert dataset.rows_defaulted_observation_fields == 1


@pytest.mark.unit
def test_load_trajectory_dataset_defaults_v2_observations_to_current_schema(
    tmp_path,
) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 8,
                "observation": _v2_observation(1.0),
                "action": 0,
                "done": False,
            },
        ],
    )

    dataset = load_trajectory_dataset(trajectory)

    assert dataset.size == 1
    assert dataset.examples[0].observation_schema_version == 2
    assert dataset.examples[0].defaulted_observation_fields == (
        "attack_army_peak",
        "army_lost_from_peak",
        "army_lost_from_peak_ratio",
        "army_count_delta",
    )
    assert dataset.observation_schema_counts == {"2": 1}
    assert dataset.rows_defaulted_observation_fields == 1


@pytest.mark.unit
def test_load_trajectory_dataset_can_reject_v1_observations(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 8,
                "observation": _v1_observation(1.0),
                "action": 0,
                "done": False,
            },
        ],
    )

    with pytest.raises(ValueError, match="base_under_threat"):
        load_trajectory_dataset(trajectory, allow_observation_defaults=False)


@pytest.mark.unit
def test_load_trajectory_dataset_can_include_terminal_rows(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _write_jsonl(
        trajectory,
        [
            {
                "step": 8,
                "observation": _observation(1.0),
                "action": 0,
                "done": False,
            },
            {
                "step": 16,
                "observation": _observation(2.0),
                "action": 1,
                "done": True,
                "result": "Victory",
            },
        ],
    )

    dataset = load_trajectory_dataset([trajectory], include_terminal=True)

    assert dataset.size == 2
    assert dataset.actions.tolist() == [0, 1]
    assert dataset.action_counts == {0: 1, 1: 1}
    assert dataset.examples[1].done is True
    assert dataset.examples[1].result == "Victory"
