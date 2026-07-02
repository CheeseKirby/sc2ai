from __future__ import annotations

import json

import pytest

from rl.experiments import create_experiment_run, read_json
from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_imitation import (
    StrategyImitationTrainConfig,
    train_strategy_imitation_policy,
)
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS


def _observation(value: float) -> dict[str, float]:
    return {field: value for field in STRATEGY_OBSERVATION_FIELDS}


@pytest.mark.unit
def test_train_strategy_imitation_writes_strategy_checkpoint_and_metrics(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    rows = [
        {
            "step": index * 64,
            "strategy_observation": _observation(float(index)),
            "strategy_action": index % 4,
            "strategy_action_name": STRATEGY_ACTION_NAMES[index % 4],
            "done": False,
        }
        for index in range(12)
    ]
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    run = create_experiment_run(
        root=tmp_path / "runs",
        name="strategy-imitation-unit",
        kind="strategy_imitation",
        timestamp="20260623_120000",
    )

    metrics = train_strategy_imitation_policy(
        config=StrategyImitationTrainConfig(
            inputs=(str(trajectory),),
            epochs=1,
            batch_size=4,
            hidden_sizes=(8,),
            validation_fraction=0.25,
            seed=5,
        ),
        run=run,
    )

    assert metrics.examples == 12
    assert metrics.train_examples == 9
    assert metrics.validation_examples == 3
    assert metrics.observation_dim == len(STRATEGY_OBSERVATION_FIELDS)
    assert metrics.observation_schema_counts == {"strategy_v2": 12}
    assert metrics.rows_defaulted_observation_fields == 0
    assert metrics.action_counts == {0: 3, 1: 3, 2: 3, 3: 3}
    assert metrics.action_names[0] == "STAY_COURSE"
    assert metrics.action_counts_by_name == {
        "STAY_COURSE": 3,
        "EXPAND": 3,
        "ADD_GATEWAYS": 3,
        "TECH_ROBO": 3,
    }
    assert metrics.missing_action_names == [
        "FORGE_UPGRADES",
        "BUILD_STATIC_DEFENSE",
        "PRODUCE_ARMY",
        "BOOST_WORKERS",
    ]
    assert metrics.validation_accuracy is not None
    assert len(metrics.confusion_matrix) == 8
    assert set(metrics.per_action_accuracy_by_name) == set(STRATEGY_ACTION_NAMES.values())
    assert (run.checkpoints_dir / "policy.pt").is_file()
    assert (run.artifacts_dir / "normalizer.json").is_file()

    metrics_json = read_json(run.artifacts_dir / "metrics.json")
    assert metrics_json["examples"] == 12
    assert metrics_json["observation_schema_counts"] == {"strategy_v2": 12}
    assert metrics_json["checkpoint_path"] == str(run.checkpoints_dir / "policy.pt")


@pytest.mark.unit
def test_train_strategy_imitation_can_use_signal_filter(tmp_path) -> None:
    trajectory = tmp_path / "strategy_signal.jsonl"
    rows = [
        {
            "step": 64,
            "strategy_observation": {
                **_observation(1.0),
                "game_time": 100.0,
                "minerals": 200.0,
                "vespene": 150.0,
                "pending_robo": 0.0,
                "ready_robo": 0.0,
                "has_cybernetics_core": 1.0,
            },
            "strategy_action": 3,
            "strategy_action_name": "TECH_ROBO",
            "done": False,
        },
        {
            "step": 128,
            "strategy_observation": {
                **_observation(2.0),
                "game_time": 150.0,
                "pending_robo": 1.0,
                "base_under_threat": 0.0,
                "base_under_air_threat": 0.0,
                "base_under_ground_threat": 0.0,
            },
            "strategy_action": 0,
            "strategy_action_name": "STAY_COURSE",
            "done": False,
        },
        {
            "step": 192,
            "strategy_observation": {
                **_observation(3.0),
                "game_time": 180.0,
                "minerals": 25.0,
                "ready_static_defense": 1.0,
                "base_under_threat": 1.0,
                "base_under_ground_threat": 1.0,
            },
            "strategy_action": 5,
            "strategy_action_name": "BUILD_STATIC_DEFENSE",
            "done": False,
        },
        {
            "step": 256,
            "strategy_observation": {
                **_observation(4.0),
                "game_time": 240.0,
            },
            "strategy_action": 0,
            "strategy_action_name": "STAY_COURSE",
            "done": True,
            "result": "Result.Tie",
        },
    ]
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    run = create_experiment_run(
        root=tmp_path / "runs",
        name="strategy-imitation-filtered-unit",
        kind="strategy_imitation",
        timestamp="20260623_130000",
    )

    metrics = train_strategy_imitation_policy(
        config=StrategyImitationTrainConfig(
            inputs=(str(trajectory),),
            epochs=1,
            batch_size=2,
            hidden_sizes=(8,),
            validation_fraction=0.0,
            seed=5,
            signal_filter="trainable",
        ),
        run=run,
    )

    assert metrics.examples == 2
    assert metrics.signal_filter == "trainable"
    assert metrics.signal_filter_summary is not None
    assert metrics.signal_filter_summary["original_examples"] == 3
    assert metrics.signal_filter_summary["kept_examples"] == 2
    assert metrics.signal_filter_summary["removed_by_training_use"] == {
        "drop_non_executable": 1,
    }

    metrics_json = read_json(run.artifacts_dir / "metrics.json")
    assert metrics_json["signal_filter"] == "trainable"
    assert metrics_json["signal_filter_summary"]["kept_examples"] == 2


@pytest.mark.unit
def test_train_strategy_imitation_can_cap_ambiguous_signal_rows(tmp_path) -> None:
    trajectory = tmp_path / "strategy_signal_capped.jsonl"
    rows = [
        {
            "step": 64,
            "strategy_observation": {
                **_observation(1.0),
                "game_time": 100.0,
                "minerals": 200.0,
                "vespene": 150.0,
                "pending_robo": 0.0,
                "ready_robo": 0.0,
                "has_cybernetics_core": 1.0,
            },
            "strategy_action": 3,
            "strategy_action_name": "TECH_ROBO",
            "done": False,
        },
        {
            "step": 128,
            "strategy_observation": {
                **_observation(2.0),
                "game_time": 150.0,
                "pending_robo": 1.0,
                "base_under_threat": 0.0,
                "base_under_air_threat": 0.0,
                "base_under_ground_threat": 0.0,
            },
            "strategy_action": 0,
            "strategy_action_name": "STAY_COURSE",
            "done": False,
        },
        {
            "step": 192,
            "strategy_observation": {
                **_observation(3.0),
                "game_time": 180.0,
                "pending_robo": 1.0,
                "base_under_threat": 0.0,
                "base_under_air_threat": 0.0,
                "base_under_ground_threat": 0.0,
            },
            "strategy_action": 0,
            "strategy_action_name": "STAY_COURSE",
            "done": False,
        },
    ]
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    run = create_experiment_run(
        root=tmp_path / "runs",
        name="strategy-imitation-capped-filter-unit",
        kind="strategy_imitation",
        timestamp="20260623_133000",
    )

    metrics = train_strategy_imitation_policy(
        config=StrategyImitationTrainConfig(
            inputs=(str(trajectory),),
            epochs=1,
            batch_size=1,
            hidden_sizes=(8,),
            validation_fraction=0.0,
            seed=5,
            signal_filter="trainable",
            max_drop_ambiguous_per_positive=0.0,
        ),
        run=run,
    )

    assert metrics.examples == 1
    assert metrics.action_counts_by_name == {"TECH_ROBO": 1}
    assert metrics.signal_filter_summary is not None
    assert metrics.signal_filter_summary["kept_by_training_use"] == {
        "accept_positive": 1,
    }
    assert metrics.signal_filter_summary["removed_by_training_use"] == {
        "drop_ambiguous": 2,
    }
    assert (
        metrics.signal_filter_summary["max_drop_ambiguous_per_positive"] == 0.0
    )
    assert metrics.signal_filter_summary["drop_ambiguous_examples_kept"] == 0

    metrics_json = read_json(run.artifacts_dir / "metrics.json")
    assert (
        metrics_json["signal_filter_summary"]["max_drop_ambiguous_per_positive"]
        == 0.0
    )


@pytest.mark.unit
def test_train_strategy_imitation_can_weight_recovery_accept_positive_rows(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy_weighted_signal.jsonl"
    rows = [
        {
            "step": 64,
            "strategy_observation": {
                **_observation(1.0),
                "game_time": 100.0,
                "minerals": 200.0,
                "vespene": 150.0,
                "ready_robo": 0.0,
                "pending_robo": 0.0,
                "has_cybernetics_core": 1.0,
            },
            "strategy_action": 3,
            "strategy_action_name": "TECH_ROBO",
            "done": False,
        },
        {
            "step": 128,
            "strategy_observation": {
                **_observation(2.0),
                "game_time": 160.0,
                "ready_gateways": 1.0,
                "pending_gateways": 0.0,
                "pending_robo": 1.0,
                "minerals": 200.0,
            },
            "strategy_action": 2,
            "strategy_action_name": "ADD_GATEWAYS",
            "done": False,
        },
        {
            "step": 192,
            "strategy_observation": {
                **_observation(3.0),
                "game_time": 220.0,
                "ready_gateways": 2.0,
                "pending_robo": 1.0,
            },
            "strategy_action": 0,
            "strategy_action_name": "STAY_COURSE",
            "done": False,
        },
    ]
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    run = create_experiment_run(
        root=tmp_path / "runs",
        name="strategy-imitation-weighted-filter-unit",
        kind="strategy_imitation",
        timestamp="20260623_134000",
    )

    metrics = train_strategy_imitation_policy(
        config=StrategyImitationTrainConfig(
            inputs=(str(trajectory),),
            epochs=1,
            batch_size=1,
            hidden_sizes=(8,),
            validation_fraction=0.0,
            seed=5,
            signal_filter="trainable",
            recovery_accept_positive_loss_weight=2.0,
            recovery_accept_positive_action_loss_weights={"TECH_ROBO": 4.0},
        ),
        run=run,
    )

    assert metrics.examples == 2
    assert metrics.recovery_accept_positive_loss_weight == 2.0
    assert metrics.recovery_accept_positive_action_loss_weights == {"TECH_ROBO": 4.0}
    assert metrics.recovery_accept_positive_context_filter == "off"
    assert metrics.recovery_accept_positive_context_oversample_factor == 1
    assert metrics.sample_weighted_examples == 1
    assert metrics.sample_weight_sum == pytest.approx(5.0)
    assert metrics.signal_filter_summary is not None
    assert (
        metrics.signal_filter_summary["recovery_accept_positive_action_loss_weights"]
        == {"TECH_ROBO": 4.0}
    )
    assert (
        metrics.signal_filter_summary["recovery_accept_positive_weighted_examples"]
        == 1
    )

    metrics_json = read_json(run.artifacts_dir / "metrics.json")
    assert metrics_json["recovery_accept_positive_loss_weight"] == 2.0
    assert metrics_json["recovery_accept_positive_action_loss_weights"] == {
        "TECH_ROBO": 4.0,
    }
    assert metrics_json["recovery_accept_positive_context_filter"] == "off"
    assert metrics_json["recovery_accept_positive_context_oversample_factor"] == 1
    assert metrics_json["sample_weighted_examples"] == 1
    assert metrics_json["sample_weight_sum"] == pytest.approx(5.0)


@pytest.mark.unit
def test_train_strategy_imitation_blocks_failed_observation_detail_gate(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    rows = [
        {
            "step": index * 64,
            "strategy_observation": _observation(float(index)),
            "strategy_action": index % 4,
            "strategy_action_name": STRATEGY_ACTION_NAMES[index % 4],
            "done": False,
        }
        for index in range(8)
    ]
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    gate_path = tmp_path / "observation_detail_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "ready": False,
                "blocking_reasons": [
                    "observation_detail_coverage_low",
                    "static_defense_type_ambiguous_rows_high",
                ],
            }
        ),
        encoding="utf-8",
    )
    run = create_experiment_run(
        root=tmp_path / "runs",
        name="strategy-imitation-gated-unit",
        kind="strategy_imitation",
        timestamp="20260623_140000",
    )

    with pytest.raises(ValueError, match="Observation detail gate failed"):
        train_strategy_imitation_policy(
            config=StrategyImitationTrainConfig(
                inputs=(str(trajectory),),
                epochs=1,
                batch_size=4,
                hidden_sizes=(8,),
                validation_fraction=0.0,
                seed=5,
                observation_detail_gate_path=str(gate_path),
            ),
            run=run,
        )

    assert not (run.checkpoints_dir / "policy.pt").exists()


@pytest.mark.unit
def test_train_strategy_imitation_records_passed_observation_detail_gate(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    rows = [
        {
            "step": index * 64,
            "strategy_observation": _observation(float(index)),
            "strategy_action": index % 4,
            "strategy_action_name": STRATEGY_ACTION_NAMES[index % 4],
            "done": False,
        }
        for index in range(8)
    ]
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    gate_path = tmp_path / "observation_detail_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "ready": True,
                "blocking_reasons": [],
                "analysis_inputs": [str(trajectory)],
            }
        ),
        encoding="utf-8",
    )
    run = create_experiment_run(
        root=tmp_path / "runs",
        name="strategy-imitation-gated-pass-unit",
        kind="strategy_imitation",
        timestamp="20260623_150000",
    )

    metrics = train_strategy_imitation_policy(
        config=StrategyImitationTrainConfig(
            inputs=(str(trajectory),),
            epochs=1,
            batch_size=4,
            hidden_sizes=(8,),
            validation_fraction=0.0,
            seed=5,
            observation_detail_gate_path=str(gate_path),
        ),
        run=run,
    )

    assert metrics.observation_detail_gate_path == str(gate_path)
    assert metrics.observation_detail_gate_ready is True
    assert metrics.observation_detail_gate_inputs == [str(trajectory)]
    metrics_json = read_json(run.artifacts_dir / "metrics.json")
    assert metrics_json["observation_detail_gate_ready"] is True
    assert metrics_json["observation_detail_gate_inputs"] == [str(trajectory)]


@pytest.mark.unit
def test_train_strategy_imitation_blocks_mismatched_observation_detail_gate_inputs(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    other_trajectory = tmp_path / "other_strategy.jsonl"
    rows = [
        {
            "step": index * 64,
            "strategy_observation": _observation(float(index)),
            "strategy_action": index % 4,
            "strategy_action_name": STRATEGY_ACTION_NAMES[index % 4],
            "done": False,
        }
        for index in range(8)
    ]
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    gate_path = tmp_path / "observation_detail_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "ready": True,
                "blocking_reasons": [],
                "analysis_inputs": [str(other_trajectory)],
            }
        ),
        encoding="utf-8",
    )
    run = create_experiment_run(
        root=tmp_path / "runs",
        name="strategy-imitation-mismatched-gate-unit",
        kind="strategy_imitation",
        timestamp="20260623_160000",
    )

    with pytest.raises(ValueError, match="Observation detail gate inputs mismatch"):
        train_strategy_imitation_policy(
            config=StrategyImitationTrainConfig(
                inputs=(str(trajectory),),
                epochs=1,
                batch_size=4,
                hidden_sizes=(8,),
                validation_fraction=0.0,
                seed=5,
                observation_detail_gate_path=str(gate_path),
            ),
            run=run,
        )

    assert not (run.checkpoints_dir / "policy.pt").exists()
