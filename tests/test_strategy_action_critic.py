from __future__ import annotations

import json

import pytest

from rl.experiments import create_experiment_run, read_json
from rl.strategy_action_critic import (
    ACTION_CRITIC_FEATURE_SCHEMA_V2,
    StrategyActionCriticTrainConfig,
    action_critic_feature_fields,
    action_critic_feature_vector_from_observation,
    load_strategy_action_critic_dataset,
    load_strategy_action_critic_checkpoint,
    non_executable_blocker_group,
    train_strategy_action_critic,
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
            "ready_gateways": 1.0,
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


def _critic_trajectory(path) -> None:
    _write_jsonl(
        path,
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
                    game_time=140.0,
                    pending_robo=1.0,
                ),
            ),
            _row(
                step=192,
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
                    supply_left=8.0,
                ),
            ),
            _row(
                step=256,
                strategy_observation=_observation(
                    game_time=340.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=6.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )


@pytest.mark.unit
def test_strategy_action_critic_dataset_labels_signal_quality(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _critic_trajectory(trajectory)

    dataset = load_strategy_action_critic_dataset(trajectory)

    assert dataset.size == 4
    assert dataset.feature_dim == len(dataset.feature_fields)
    assert dataset.label_counts == {0: 3, 1: 1}
    assert dataset.label_counts_by_name == {"safe": 3, "unsafe": 1}
    assert dataset.training_use_counts == {
        "accept_positive": 1,
        "drop_ambiguous": 2,
        "veto_negative": 1,
    }
    assert dataset.records[0].candidate_action == "TECH_ROBO"
    assert dataset.records[0].label == 0
    assert dataset.records[2].candidate_action == "PRODUCE_ARMY"
    assert dataset.records[2].label == 1


@pytest.mark.unit
def test_strategy_action_critic_dataset_drops_action_space_exhausted_rows(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=25.0,
                    vespene=0.0,
                    supply_left=0.0,
                    own_bases=1.0,
                    ready_gateways=4.0,
                    pending_gateways=0.0,
                    has_cybernetics_core=0.0,
                    ready_forge=0.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=8.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=220.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                    army_count=5.0,
                ),
                done=True,
                result="Result.Defeat",
            ),
        ],
    )

    dataset = load_strategy_action_critic_dataset(trajectory)

    assert dataset.size == 0
    assert dataset.dropped_records_by_training_use == {"action_space_exhausted": 1}


@pytest.mark.unit
def test_strategy_action_critic_conservative_label_policy_drops_ambiguous(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _critic_trajectory(trajectory)

    dataset = load_strategy_action_critic_dataset(
        trajectory,
        label_policy="conservative",
    )

    assert dataset.label_policy == "conservative"
    assert dataset.size == 2
    assert dataset.label_counts == {0: 1, 1: 1}
    assert dataset.label_counts_by_name == {"safe": 1, "unsafe": 1}
    assert dataset.training_use_counts == {
        "accept_positive": 1,
        "veto_negative": 1,
    }
    assert dataset.dropped_records_by_training_use == {"drop_ambiguous": 2}


@pytest.mark.unit
def test_strategy_action_critic_outcome_conservative_policy_drops_non_executable(
    tmp_path,
) -> None:
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
                    ready_gateways=4.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
            ),
            _row(
                step=128,
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
                    supply_left=8.0,
                ),
            ),
            _row(
                step=192,
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

    dataset = load_strategy_action_critic_dataset(
        trajectory,
        label_policy="outcome-conservative",
    )

    assert dataset.label_policy == "outcome-conservative"
    assert dataset.size == 1
    assert dataset.label_counts == {1: 1}
    assert dataset.label_counts_by_name == {"unsafe": 1}
    assert dataset.training_use_counts == {"veto_negative": 1}
    assert dataset.dropped_records_by_training_use == {"drop_non_executable": 1}


@pytest.mark.unit
def test_strategy_action_critic_can_downweight_non_executable_labels(tmp_path) -> None:
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
                    ready_gateways=4.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
            ),
            _row(
                step=128,
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
                    supply_left=8.0,
                ),
            ),
            _row(
                step=192,
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

    dataset = load_strategy_action_critic_dataset(
        trajectory,
        label_policy="conservative",
        drop_non_executable_weight=0.25,
    )

    assert dataset.drop_non_executable_weight == 0.25
    assert dataset.label_counts_by_name == {"unsafe": 2}
    assert dataset.training_use_counts == {
        "drop_non_executable": 1,
        "veto_negative": 1,
    }
    assert dataset.training_use_weight_sums == {
        "drop_non_executable": 0.25,
        "veto_negative": 1.0,
    }
    assert dataset.non_executable_blocker_counts == {"target_gateways_reached": 1}
    assert dataset.non_executable_blocker_group_counts == {"cap_or_duplicate": 1}
    assert dataset.non_executable_blocker_weight_sums == {
        "target_gateways_reached": 0.25
    }
    assert dataset.non_executable_blocker_group_weight_sums == {
        "cap_or_duplicate": 0.25
    }
    assert dataset.example_weights.tolist() == [0.25, 1.0]

    with pytest.raises(ValueError, match="drop_non_executable_weight"):
        load_strategy_action_critic_dataset(
            trajectory,
            drop_non_executable_weight=-0.1,
        )


@pytest.mark.unit
def test_strategy_action_critic_can_weight_non_executable_by_blocker_group(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=100.0,
                    minerals=25.0,
                    has_cybernetics_core=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                ),
            ),
            _row(
                step=128,
                strategy_action=6,
                strategy_action_name="PRODUCE_ARMY",
                strategy_observation=_observation(
                    game_time=120.0,
                    minerals=25.0,
                    ready_gateways=1.0,
                    gateway_idle_count=1.0,
                    supply_left=8.0,
                ),
            ),
            _row(done=True, result="Result.Defeat"),
        ],
    )

    dataset = load_strategy_action_critic_dataset(
        trajectory,
        label_policy="conservative",
        drop_non_executable_weight=1.0,
        non_executable_blocker_weights={
            "resource_short": 0.5,
            "cannot_afford_army": 0.25,
        },
    )

    assert non_executable_blocker_group("cannot_afford_army") == "resource_short"
    assert non_executable_blocker_group("no_ready_nexus") == "production_missing"
    assert dataset.non_executable_blocker_counts == {
        "cannot_afford_army": 1,
        "cannot_afford_static_defense": 1,
    }
    assert dataset.non_executable_blocker_group_counts == {"resource_short": 2}
    assert dataset.non_executable_blocker_weight_sums == {
        "cannot_afford_army": 0.25,
        "cannot_afford_static_defense": 0.5,
    }
    assert dataset.non_executable_blocker_group_weight_sums == {
        "resource_short": 0.75
    }
    assert dataset.example_weights.tolist() == [0.5, 0.25]


@pytest.mark.unit
def test_strategy_action_critic_v2_features_include_action_threat_interactions() -> None:
    fields = action_critic_feature_fields(ACTION_CRITIC_FEATURE_SCHEMA_V2)
    observation = _observation(
        base_under_threat=1.0,
        base_under_air_threat=0.0,
        base_under_ground_threat=1.0,
    )

    vector = action_critic_feature_vector_from_observation(
        observation,
        "STAY_COURSE",
        feature_fields=fields,
    )
    values = dict(zip(fields, vector.tolist()))

    assert "threat:any" in fields
    assert "action_threat:STAY_COURSE:ground" in fields
    assert values["threat:any"] == 1.0
    assert values["threat:air"] == 0.0
    assert values["threat:ground"] == 1.0
    assert values["action_threat:STAY_COURSE:ground"] == 1.0
    assert values["action_threat:PRODUCE_ARMY:ground"] == 0.0


@pytest.mark.unit
def test_train_strategy_action_critic_writes_checkpoint_and_metrics(tmp_path) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _critic_trajectory(trajectory)
    run = create_experiment_run(
        root=tmp_path / "runs",
        name="strategy-action-critic-unit",
        kind="strategy_action_critic",
        timestamp="20260629_120000",
    )

    metrics = train_strategy_action_critic(
        config=StrategyActionCriticTrainConfig(
            inputs=(str(trajectory),),
            epochs=2,
            batch_size=2,
            hidden_sizes=(8,),
            validation_fraction=0.0,
            seed=5,
            class_weighting="balanced",
            label_policy="conservative",
            feature_schema_version=ACTION_CRITIC_FEATURE_SCHEMA_V2,
            drop_non_executable_weight=0.5,
        ),
        run=run,
    )

    assert metrics.examples == 2
    assert metrics.unsafe_examples == 1
    assert metrics.safe_examples == 1
    assert metrics.train_examples == 2
    assert metrics.validation_examples == 0
    assert metrics.label_policy == "conservative"
    assert metrics.feature_schema_version == ACTION_CRITIC_FEATURE_SCHEMA_V2
    assert metrics.feature_dim == len(metrics.feature_fields)
    assert "action:PRODUCE_ARMY" in metrics.feature_fields
    assert "action_threat:PRODUCE_ARMY:ground" in metrics.feature_fields
    assert metrics.class_weighting == "balanced"
    assert metrics.drop_non_executable_weight == 0.5
    assert metrics.train_weight_sum == 2.0
    assert metrics.validation_weight_sum == 0.0
    assert (run.checkpoints_dir / "critic.pt").is_file()
    assert (run.artifacts_dir / "normalizer.json").is_file()

    loaded = load_strategy_action_critic_checkpoint(run.checkpoints_dir / "critic.pt")
    assert loaded.metadata.feature_fields == tuple(metrics.feature_fields)
    assert loaded.metadata.feature_schema_version == ACTION_CRITIC_FEATURE_SCHEMA_V2

    metrics_json = read_json(run.artifacts_dir / "metrics.json")
    assert metrics_json["examples"] == 2
    assert metrics_json["label_policy"] == "conservative"
    assert metrics_json["feature_schema_version"] == ACTION_CRITIC_FEATURE_SCHEMA_V2
    assert metrics_json["drop_non_executable_weight"] == 0.5
    assert metrics_json["checkpoint_path"] == str(run.checkpoints_dir / "critic.pt")


@pytest.mark.unit
def test_train_strategy_action_critic_blocks_failed_observation_detail_gate(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _critic_trajectory(trajectory)
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
        name="strategy-action-critic-gated-unit",
        kind="strategy_action_critic",
        timestamp="20260629_120010",
    )

    with pytest.raises(ValueError, match="Observation detail gate failed"):
        train_strategy_action_critic(
            config=StrategyActionCriticTrainConfig(
                inputs=(str(trajectory),),
                epochs=1,
                batch_size=2,
                hidden_sizes=(8,),
                validation_fraction=0.0,
                seed=5,
                observation_detail_gate_path=str(gate_path),
            ),
            run=run,
        )

    assert not (run.checkpoints_dir / "critic.pt").exists()


@pytest.mark.unit
def test_train_strategy_action_critic_records_passed_observation_detail_gate(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _critic_trajectory(trajectory)
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
        name="strategy-action-critic-gated-pass-unit",
        kind="strategy_action_critic",
        timestamp="20260629_120020",
    )

    metrics = train_strategy_action_critic(
        config=StrategyActionCriticTrainConfig(
            inputs=(str(trajectory),),
            epochs=1,
            batch_size=2,
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
def test_train_strategy_action_critic_blocks_mismatched_observation_detail_gate_inputs(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    other_trajectory = tmp_path / "other_strategy.jsonl"
    _critic_trajectory(trajectory)
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
        name="strategy-action-critic-mismatched-gate-unit",
        kind="strategy_action_critic",
        timestamp="20260629_120030",
    )

    with pytest.raises(ValueError, match="Observation detail gate inputs mismatch"):
        train_strategy_action_critic(
            config=StrategyActionCriticTrainConfig(
                inputs=(str(trajectory),),
                epochs=1,
                batch_size=2,
                hidden_sizes=(8,),
                validation_fraction=0.0,
                seed=5,
                observation_detail_gate_path=str(gate_path),
            ),
            run=run,
        )

    assert not (run.checkpoints_dir / "critic.pt").exists()


@pytest.mark.unit
def test_train_strategy_action_critic_accepts_outcome_conservative_policy(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _critic_trajectory(trajectory)
    run = create_experiment_run(
        root=tmp_path / "runs",
        name="strategy-action-critic-outcome-unit",
        kind="strategy_action_critic",
        timestamp="20260629_120100",
    )

    metrics = train_strategy_action_critic(
        config=StrategyActionCriticTrainConfig(
            inputs=(str(trajectory),),
            epochs=1,
            batch_size=2,
            hidden_sizes=(8,),
            validation_fraction=0.0,
            seed=5,
            class_weighting="none",
            label_policy="outcome-conservative",
        ),
        run=run,
    )

    assert metrics.label_policy == "outcome-conservative"
    assert metrics.examples == 2
    assert metrics.safe_examples == 1
    assert metrics.unsafe_examples == 1
    assert metrics.dropped_records_by_training_use == {"drop_ambiguous": 2}

    metrics_json = read_json(run.artifacts_dir / "metrics.json")
    assert metrics_json["label_policy"] == "outcome-conservative"
    assert metrics_json["checkpoint_path"] == str(run.checkpoints_dir / "critic.pt")
