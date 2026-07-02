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
def test_trainable_signal_filter_can_cap_drop_ambiguous_rows(tmp_path) -> None:
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
            _row(step=128, strategy_action=0, strategy_action_name="STAY_COURSE"),
            _row(step=192, strategy_action=0, strategy_action_name="STAY_COURSE"),
            _row(step=256, strategy_action=0, strategy_action_name="STAY_COURSE"),
            _row(done=True, result="Result.Tie"),
        ],
    )

    filtered = load_signal_filtered_strategy_trajectory_dataset(
        trajectory,
        filter_name="trainable",
        max_drop_ambiguous_per_positive=1.0,
        balance_seed=3,
    )

    assert filtered.dataset.size == 2
    assert filtered.dataset.actions.tolist() == [2, 0]
    assert filtered.summary.kept_by_training_use == {
        "accept_positive": 1,
        "drop_ambiguous": 1,
    }
    assert filtered.summary.removed_by_training_use == {"drop_ambiguous": 2}
    assert filtered.summary.max_drop_ambiguous_per_positive == 1.0
    assert filtered.summary.balance_seed == 3
    assert filtered.summary.positive_examples_for_balance == 1
    assert filtered.summary.drop_ambiguous_examples_before_balance == 3
    assert filtered.summary.drop_ambiguous_examples_kept == 1
    assert filtered.summary.drop_ambiguous_examples_removed_by_balance == 2


@pytest.mark.unit
def test_recovery_safe_filter_drops_ambiguous_wait_when_recovery_is_executable(
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
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                    minerals=200.0,
                ),
            ),
            _row(
                step=128,
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=150.0,
                    minerals=200.0,
                    vespene=150.0,
                    supply_left=8.0,
                    ready_gateways=2.0,
                    pending_gateways=0.0,
                    has_cybernetics_core=1.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                ),
            ),
            _row(
                step=192,
                strategy_action=0,
                strategy_action_name="STAY_COURSE",
                strategy_observation=_observation(
                    game_time=180.0,
                    minerals=0.0,
                    vespene=0.0,
                    supply_left=0.0,
                    ready_gateways=2.0,
                    gateway_idle_count=0.0,
                    has_cybernetics_core=0.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    ready_static_defense=2.0,
                    pending_static_defense=0.0,
                ),
            ),
            _row(done=True, result="Result.Tie"),
        ],
    )

    baseline = load_signal_filtered_strategy_trajectory_dataset(
        trajectory,
        filter_name="trainable",
    )
    filtered = load_signal_filtered_strategy_trajectory_dataset(
        trajectory,
        filter_name="trainable-recovery-safe",
    )

    assert baseline.dataset.size == 3
    assert baseline.summary.recovery_safe_filter_enabled is False
    assert filtered.dataset.size == 2
    assert filtered.dataset.actions.tolist() == [2, 0]
    assert filtered.summary.recovery_safe_filter_enabled is True
    assert filtered.summary.recovery_opportunity_ambiguous_examples_removed == 1
    assert filtered.summary.recovery_opportunity_removed_actions_by_name == {
        "STAY_COURSE": 1
    }
    assert filtered.summary.kept_by_training_use == {
        "accept_positive": 1,
        "drop_ambiguous": 1,
    }
    assert filtered.summary.removed_by_training_use == {
        "drop_ambiguous_recovery_opportunity": 1
    }


@pytest.mark.unit
def test_signal_filter_can_oversample_observed_positive_recovery_rows(tmp_path) -> None:
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
                    ready_robo=0.0,
                    pending_robo=0.0,
                    has_cybernetics_core=1.0,
                ),
            ),
            _row(
                step=128,
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
                strategy_observation=_observation(
                    game_time=160.0,
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                    pending_robo=1.0,
                    minerals=200.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=220.0,
                    ready_gateways=2.0,
                    pending_robo=1.0,
                ),
            ),
            _row(done=True, result="Result.Tie"),
        ],
    )

    filtered = load_signal_filtered_strategy_trajectory_dataset(
        trajectory,
        filter_name="trainable",
        recovery_positive_oversample_factor=3,
    )

    assert filtered.summary.original_examples == 3
    assert filtered.summary.removed_examples == 0
    assert filtered.dataset.size == 5
    assert filtered.dataset.actions.tolist() == [3, 2, 0, 3, 3]
    assert filtered.summary.kept_by_training_use == {
        "accept_positive": 4,
        "drop_ambiguous": 1,
    }
    assert filtered.summary.recovery_positive_oversample_factor == 3
    assert filtered.summary.recovery_positive_examples_before_oversample == 1
    assert filtered.summary.recovery_positive_examples_added_by_oversample == 2
    assert filtered.summary.recovery_positive_oversampled_actions_by_name == {
        "TECH_ROBO": 2
    }
    assert filtered.summary.kept_action_counts_by_name == {
        "ADD_GATEWAYS": 1,
        "STAY_COURSE": 1,
        "TECH_ROBO": 3,
    }


@pytest.mark.unit
def test_signal_filter_can_weight_observed_positive_recovery_rows(tmp_path) -> None:
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
                    ready_robo=0.0,
                    pending_robo=0.0,
                    has_cybernetics_core=1.0,
                ),
            ),
            _row(
                step=128,
                strategy_action=2,
                strategy_action_name="ADD_GATEWAYS",
                strategy_observation=_observation(
                    game_time=160.0,
                    ready_gateways=1.0,
                    pending_gateways=0.0,
                    pending_robo=1.0,
                    minerals=200.0,
                ),
            ),
            _row(
                step=192,
                strategy_observation=_observation(
                    game_time=220.0,
                    ready_gateways=2.0,
                    pending_robo=1.0,
                ),
            ),
            _row(done=True, result="Result.Tie"),
        ],
    )

    filtered = load_signal_filtered_strategy_trajectory_dataset(
        trajectory,
        filter_name="trainable",
        recovery_accept_positive_loss_weight=4.0,
    )

    assert filtered.dataset.actions.tolist() == [3, 2, 0]
    assert filtered.sample_weights.tolist() == pytest.approx([4.0, 1.0, 1.0])
    assert filtered.summary.recovery_accept_positive_loss_weight == 4.0
    assert filtered.summary.recovery_accept_positive_weighted_examples == 1
    assert filtered.summary.sample_weight_sum == pytest.approx(6.0)


@pytest.mark.unit
def test_signal_filter_can_weight_recovery_accept_positive_rows_by_action(
    tmp_path,
) -> None:
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
                    ready_robo=0.0,
                    pending_robo=0.0,
                    has_cybernetics_core=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=160.0,
                    pending_robo=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                ),
            ),
            _row(
                step=192,
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=220.0,
                    minerals=200.0,
                    ready_forge=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                ),
            ),
            _row(
                step=256,
                strategy_observation=_observation(
                    game_time=280.0,
                    pending_robo=1.0,
                    ready_static_defense=1.0,
                    pending_static_defense=0.0,
                ),
                done=True,
                result="Result.Tie",
            ),
        ],
    )

    filtered = load_signal_filtered_strategy_trajectory_dataset(
        trajectory,
        filter_name="trainable",
        recovery_accept_positive_loss_weight=2.0,
        recovery_accept_positive_action_loss_weights={
            "BUILD_STATIC_DEFENSE": 5.0,
        },
    )

    assert filtered.dataset.actions.tolist() == [3, 0, 5]
    assert filtered.sample_weights.tolist() == pytest.approx([2.0, 1.0, 5.0])
    assert filtered.summary.recovery_accept_positive_loss_weight == 2.0
    assert filtered.summary.recovery_accept_positive_action_loss_weights == {
        "BUILD_STATIC_DEFENSE": 5.0,
    }
    assert filtered.summary.recovery_accept_positive_weighted_examples == 2
    assert filtered.summary.recovery_accept_positive_weighted_actions_by_name == {
        "BUILD_STATIC_DEFENSE": 1,
        "TECH_ROBO": 1,
    }
    assert filtered.summary.sample_weight_sum == pytest.approx(8.0)


@pytest.mark.unit
def test_signal_filter_can_context_gate_recovery_accept_positive_weights(
    tmp_path,
) -> None:
    trajectory = tmp_path / "strategy.jsonl"
    _write_jsonl(
        trajectory,
        [
            _row(
                step=64,
                strategy_action=3,
                strategy_action_name="TECH_ROBO",
                strategy_observation=_observation(
                    game_time=560.0,
                    minerals=200.0,
                    vespene=600.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    has_cybernetics_core=1.0,
                    base_under_threat=0.0,
                ),
            ),
            _row(
                step=128,
                strategy_observation=_observation(
                    game_time=576.0,
                    pending_robo=1.0,
                ),
            ),
            _row(
                step=192,
                strategy_action=3,
                strategy_action_name="TECH_ROBO",
                strategy_observation=_observation(
                    game_time=300.0,
                    minerals=200.0,
                    vespene=150.0,
                    ready_robo=0.0,
                    pending_robo=0.0,
                    has_cybernetics_core=1.0,
                    base_under_threat=0.0,
                ),
            ),
            _row(
                step=256,
                strategy_observation=_observation(
                    game_time=316.0,
                    pending_robo=1.0,
                ),
            ),
            _row(
                step=320,
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=430.0,
                    minerals=150.0,
                    ready_forge=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    own_bases=1.0,
                    base_under_threat=0.0,
                ),
            ),
            _row(
                step=384,
                strategy_observation=_observation(
                    game_time=446.0,
                    ready_static_defense=1.0,
                    pending_static_defense=0.0,
                ),
            ),
            _row(
                step=448,
                strategy_action=5,
                strategy_action_name="BUILD_STATIC_DEFENSE",
                strategy_observation=_observation(
                    game_time=430.0,
                    minerals=150.0,
                    ready_forge=1.0,
                    ready_static_defense=0.0,
                    pending_static_defense=0.0,
                    own_bases=1.0,
                    base_under_threat=1.0,
                    base_under_ground_threat=1.0,
                ),
            ),
            _row(
                step=512,
                strategy_observation=_observation(
                    game_time=446.0,
                    ready_static_defense=1.0,
                    pending_static_defense=0.0,
                    base_under_threat=0.0,
                    base_under_ground_threat=0.0,
                ),
            ),
            _row(done=True, result="Result.Tie"),
        ],
    )

    filtered = load_signal_filtered_strategy_trajectory_dataset(
        trajectory,
        filter_name="trainable",
        recovery_accept_positive_action_loss_weights={
            "BUILD_STATIC_DEFENSE": 5.0,
            "TECH_ROBO": 4.0,
        },
        recovery_accept_positive_context_filter="pre-collapse-recovery",
    )

    assert filtered.summary.recovery_accept_positive_context_filter == (
        "pre-collapse-recovery"
    )
    assert filtered.summary.recovery_accept_positive_context_matched_examples == 2
    assert filtered.summary.recovery_accept_positive_context_skipped_examples == 2
    assert filtered.summary.recovery_accept_positive_context_matched_actions_by_name == {
        "BUILD_STATIC_DEFENSE": 1,
        "TECH_ROBO": 1,
    }
    assert filtered.summary.recovery_accept_positive_context_skipped_actions_by_name == {
        "BUILD_STATIC_DEFENSE": 1,
        "TECH_ROBO": 1,
    }
    assert filtered.summary.recovery_accept_positive_weighted_examples == 2
    assert filtered.summary.recovery_accept_positive_weighted_actions_by_name == {
        "BUILD_STATIC_DEFENSE": 1,
        "TECH_ROBO": 1,
    }
    assert sorted(filtered.sample_weights.tolist()) == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 5.0]
    )

    oversampled = load_signal_filtered_strategy_trajectory_dataset(
        trajectory,
        filter_name="trainable",
        recovery_accept_positive_action_loss_weights={
            "BUILD_STATIC_DEFENSE": 5.0,
            "TECH_ROBO": 4.0,
        },
        recovery_accept_positive_context_filter="pre-collapse-recovery",
        recovery_accept_positive_context_oversample_factor=3,
    )

    assert oversampled.dataset.size == 12
    assert (
        oversampled.summary.recovery_accept_positive_context_examples_before_oversample
        == 2
    )
    assert (
        oversampled.summary.recovery_accept_positive_context_examples_added_by_oversample
        == 4
    )
    assert (
        oversampled.summary.recovery_accept_positive_context_oversampled_actions_by_name
        == {"BUILD_STATIC_DEFENSE": 2, "TECH_ROBO": 2}
    )
    assert oversampled.summary.recovery_accept_positive_weighted_examples == 6
    assert sorted(oversampled.sample_weights.tolist()) == pytest.approx(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            4.0,
            4.0,
            4.0,
            5.0,
            5.0,
            5.0,
        ]
    )


@pytest.mark.unit
def test_signal_filter_rejects_unknown_preset(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown signal filter"):
        load_signal_filtered_strategy_trajectory_dataset(
            tmp_path,
            filter_name="made-up",
        )


@pytest.mark.unit
def test_signal_filter_rejects_negative_drop_ambiguous_cap(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        load_signal_filtered_strategy_trajectory_dataset(
            tmp_path,
            filter_name="trainable",
            max_drop_ambiguous_per_positive=-1.0,
        )


@pytest.mark.unit
def test_signal_filter_rejects_non_positive_recovery_oversample_factor(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="recovery_positive_oversample_factor"):
        load_signal_filtered_strategy_trajectory_dataset(
            tmp_path,
            filter_name="trainable",
            recovery_positive_oversample_factor=0,
        )


@pytest.mark.unit
def test_signal_filter_rejects_low_recovery_loss_weight(tmp_path) -> None:
    with pytest.raises(ValueError, match="recovery_accept_positive_loss_weight"):
        load_signal_filtered_strategy_trajectory_dataset(
            tmp_path,
            filter_name="trainable",
            recovery_accept_positive_loss_weight=0.5,
        )


@pytest.mark.unit
def test_signal_filter_rejects_unknown_recovery_action_loss_weight(tmp_path) -> None:
    with pytest.raises(ValueError, match="keys must be one of"):
        load_signal_filtered_strategy_trajectory_dataset(
            tmp_path,
            filter_name="trainable",
            recovery_accept_positive_action_loss_weights={"EXPAND": 3.0},
        )


@pytest.mark.unit
def test_signal_filter_rejects_unknown_recovery_context_filter(tmp_path) -> None:
    with pytest.raises(ValueError, match="recovery_accept_positive_context_filter"):
        load_signal_filtered_strategy_trajectory_dataset(
            tmp_path,
            filter_name="trainable",
            recovery_accept_positive_context_filter="made-up",
        )


@pytest.mark.unit
def test_signal_filter_rejects_invalid_recovery_context_oversampling(
    tmp_path,
) -> None:
    with pytest.raises(
        ValueError,
        match="recovery_accept_positive_context_oversample_factor",
    ):
        load_signal_filtered_strategy_trajectory_dataset(
            tmp_path,
            filter_name="trainable",
            recovery_accept_positive_context_oversample_factor=0,
        )

    with pytest.raises(
        ValueError,
        match="requires a recovery_accept_positive_context_filter",
    ):
        load_signal_filtered_strategy_trajectory_dataset(
            tmp_path,
            filter_name="trainable",
            recovery_accept_positive_context_oversample_factor=2,
        )
