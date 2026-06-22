from __future__ import annotations

import json

import numpy as np
import pytest

from rl.experiments import create_experiment_run, read_json
from rl.imitation import (
    ImitationTrainConfig,
    build_class_weights,
    build_confusion_matrix,
    per_action_accuracy_by_name,
    split_dataset,
    train_imitation_policy,
)
from rl.datasets import load_trajectory_dataset
from rl.observations import OBSERVATION_FIELDS


def _observation(value: float) -> dict[str, float]:
    return {field: value for field in OBSERVATION_FIELDS}


@pytest.mark.unit
def test_train_imitation_policy_writes_checkpoint_and_metrics(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    rows = [
        {
            "step": index,
            "observation": _observation(float(index)),
            "action": index % 2,
            "done": False,
        }
        for index in range(8)
    ]
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    run = create_experiment_run(
        root=tmp_path / "runs",
        name="imitation-unit",
        kind="imitation",
        timestamp="20260618_120000",
    )

    metrics = train_imitation_policy(
        config=ImitationTrainConfig(
            inputs=(str(trajectory),),
            epochs=1,
            batch_size=4,
            hidden_sizes=(8,),
            validation_fraction=0.25,
            seed=3,
        ),
        run=run,
    )

    assert metrics.examples == 8
    assert metrics.train_examples == 6
    assert metrics.validation_examples == 2
    assert metrics.observation_dim == len(OBSERVATION_FIELDS)
    assert metrics.observation_schema_counts == {"3": 8}
    assert metrics.rows_defaulted_observation_fields == 0
    assert metrics.action_counts == {0: 4, 1: 4}
    assert metrics.action_names[0] == "RALLY"
    assert metrics.action_counts_by_name == {"RALLY": 4, "ATTACK_MAIN": 4}
    assert metrics.missing_action_names == [
        "RETREAT_HOME",
        "DEFEND_BASE",
        "HOLD",
    ]
    assert metrics.class_weighting == "none"
    assert metrics.class_weights_by_name["RALLY"] == 1.0
    assert metrics.validation_accuracy is not None
    assert len(metrics.confusion_matrix) == 5
    assert set(metrics.per_action_accuracy_by_name) == {
        "RALLY",
        "ATTACK_MAIN",
        "RETREAT_HOME",
        "DEFEND_BASE",
        "HOLD",
    }
    assert (run.checkpoints_dir / "policy.pt").is_file()
    assert (run.artifacts_dir / "normalizer.json").is_file()
    metrics_json = read_json(run.artifacts_dir / "metrics.json")
    assert metrics_json["examples"] == 8
    assert metrics_json["normalizer_path"] is not None
    assert metrics_json["observation_schema_counts"] == {"3": 8}
    assert metrics_json["class_weighting"] == "none"
    assert metrics_json["missing_action_names"] == [
        "RETREAT_HOME",
        "DEFEND_BASE",
        "HOLD",
    ]


@pytest.mark.unit
def test_split_dataset_is_deterministic(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    rows = [
        {
            "step": index,
            "observation": _observation(float(index)),
            "action": index % 2,
            "done": False,
        }
        for index in range(10)
    ]
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    dataset = load_trajectory_dataset([trajectory])

    first = split_dataset(dataset, validation_fraction=0.2, seed=11)
    second = split_dataset(dataset, validation_fraction=0.2, seed=11)

    assert first.train_actions.tolist() == second.train_actions.tolist()
    assert first.validation_actions.tolist() == second.validation_actions.tolist()
    assert first.validation_actions.shape[0] == 2


@pytest.mark.unit
def test_build_confusion_matrix_uses_rows_as_labels() -> None:
    matrix = build_confusion_matrix(
        predictions=np.asarray([0, 1, 1, 0]),
        labels=np.asarray([0, 0, 1, 1]),
        action_dim=2,
    )

    assert matrix == [[1, 1], [1, 1]]


@pytest.mark.unit
def test_build_class_weights_balances_present_labels() -> None:
    weights = build_class_weights(
        np.asarray([0, 0, 0, 1], dtype=np.int64),
        action_dim=3,
        strategy="balanced",
    )

    assert weights[0] == pytest.approx(4 / (2 * 3))
    assert weights[1] == pytest.approx(4 / (2 * 1))
    assert weights[2] == 0.0


@pytest.mark.unit
def test_per_action_accuracy_by_name_handles_missing_rows() -> None:
    accuracy = per_action_accuracy_by_name(
        [
            [2, 1, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1],
        ]
    )

    assert accuracy["RALLY"] == pytest.approx(2 / 3)
    assert accuracy["ATTACK_MAIN"] == 1.0
    assert accuracy["RETREAT_HOME"] is None
    assert accuracy["HOLD"] == pytest.approx(1 / 2)
