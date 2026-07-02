"""Signal-filtered strategy trajectory datasets for safer imitation training."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_datasets import (
    StrategyTrajectoryDataset,
    StrategyTrajectoryExample,
    StrategyTrajectoryPathInput,
    iter_strategy_trajectory_examples,
)
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS
from rl.strategy_replay_candidate import candidate_executability_from_observation
from rl.strategy_signal_dataset import build_strategy_signal_dataset


SIGNAL_FILTER_PRESETS: dict[str, tuple[str, ...]] = {
    "strict-positive": ("accept_positive",),
    "trainable": ("accept_positive", "drop_ambiguous", "weak_context"),
    "trainable-recovery-safe": (
        "accept_positive",
        "drop_ambiguous",
        "weak_context",
    ),
}
RECOVERY_ACTION_NAMES: tuple[str, ...] = (
    "TECH_ROBO",
    "PRODUCE_ARMY",
    "BUILD_STATIC_DEFENSE",
)
RECOVERY_ACCEPT_POSITIVE_CONTEXT_FILTERS: tuple[str, ...] = (
    "off",
    "pre-collapse-recovery",
)
ROBO_MINERALS = 150.0
STATIC_DEFENSE_MINERALS = 100.0
PHOTON_CANNON_MINERALS = 150.0
PRODUCE_ARMY_MINERALS = 100.0
PRE_COLLAPSE_ROBO_TIME = 540.0
PRE_COLLAPSE_ROBO_VESPENE_BANK = 500.0
PRE_COLLAPSE_STATIC_DEFENSE_TIME = 420.0
DESIRED_MIN_ARMY = 10.0


@dataclass(frozen=True)
class StrategySignalFilterSummary:
    """Summary of signal filtering applied before strategy imitation training."""

    filter_name: str
    allowed_training_uses: tuple[str, ...]
    total_signal_records: int
    original_examples: int
    kept_examples: int
    removed_examples: int
    kept_by_training_use: dict[str, int]
    removed_by_training_use: dict[str, int]
    kept_action_counts: dict[int, int]
    removed_action_counts: dict[int, int]
    kept_action_counts_by_name: dict[str, int]
    removed_action_counts_by_name: dict[str, int]
    max_drop_ambiguous_per_positive: float | None = None
    balance_seed: int | None = None
    positive_examples_for_balance: int = 0
    drop_ambiguous_examples_before_balance: int = 0
    drop_ambiguous_examples_kept: int = 0
    drop_ambiguous_examples_removed_by_balance: int = 0
    recovery_safe_filter_enabled: bool = False
    recovery_opportunity_ambiguous_examples_removed: int = 0
    recovery_opportunity_removed_actions: dict[int, int] | None = None
    recovery_opportunity_removed_actions_by_name: dict[str, int] | None = None
    recovery_positive_oversample_factor: int = 1
    recovery_positive_examples_before_oversample: int = 0
    recovery_positive_examples_added_by_oversample: int = 0
    recovery_positive_oversampled_actions: dict[int, int] | None = None
    recovery_positive_oversampled_actions_by_name: dict[str, int] | None = None
    recovery_accept_positive_loss_weight: float = 1.0
    recovery_accept_positive_action_loss_weights: dict[str, float] | None = None
    recovery_accept_positive_context_filter: str = "off"
    recovery_accept_positive_context_matched_examples: int = 0
    recovery_accept_positive_context_skipped_examples: int = 0
    recovery_accept_positive_context_matched_actions_by_name: dict[str, int] | None = None
    recovery_accept_positive_context_skipped_actions_by_name: dict[str, int] | None = None
    recovery_accept_positive_context_oversample_factor: int = 1
    recovery_accept_positive_context_examples_before_oversample: int = 0
    recovery_accept_positive_context_examples_added_by_oversample: int = 0
    recovery_accept_positive_context_oversampled_actions_by_name: dict[str, int] | None = None
    recovery_accept_positive_weighted_examples: int = 0
    recovery_accept_positive_weighted_actions_by_name: dict[str, int] | None = None
    sample_weight_sum: float = 0.0


@dataclass(frozen=True)
class SignalFilteredStrategyDataset:
    """Filtered dataset plus its signal-filter summary."""

    dataset: StrategyTrajectoryDataset
    summary: StrategySignalFilterSummary
    sample_weights: np.ndarray


def load_signal_filtered_strategy_trajectory_dataset(
    paths: StrategyTrajectoryPathInput,
    *,
    filter_name: str = "trainable",
    include_terminal: bool = False,
    allow_observation_defaults: bool = True,
    max_drop_ambiguous_per_positive: float | None = None,
    balance_seed: int = 7,
    recovery_positive_oversample_factor: int = 1,
    recovery_accept_positive_loss_weight: float = 1.0,
    recovery_accept_positive_action_loss_weights: dict[str, float] | None = None,
    recovery_accept_positive_context_filter: str = "off",
    recovery_accept_positive_context_oversample_factor: int = 1,
) -> SignalFilteredStrategyDataset:
    """Load strategy examples after dropping low-quality signal rows."""
    if filter_name not in SIGNAL_FILTER_PRESETS:
        names = ", ".join(sorted(SIGNAL_FILTER_PRESETS))
        raise ValueError(f"Unknown signal filter {filter_name!r}; expected one of {names}")
    if (
        max_drop_ambiguous_per_positive is not None
        and max_drop_ambiguous_per_positive < 0
    ):
        raise ValueError("max_drop_ambiguous_per_positive must be non-negative")
    if recovery_positive_oversample_factor < 1:
        raise ValueError("recovery_positive_oversample_factor must be >= 1")
    if recovery_accept_positive_context_oversample_factor < 1:
        raise ValueError(
            "recovery_accept_positive_context_oversample_factor must be >= 1"
        )
    if recovery_accept_positive_loss_weight < 1.0:
        raise ValueError("recovery_accept_positive_loss_weight must be >= 1.0")
    if recovery_accept_positive_context_filter not in (
        RECOVERY_ACCEPT_POSITIVE_CONTEXT_FILTERS
    ):
        names = ", ".join(RECOVERY_ACCEPT_POSITIVE_CONTEXT_FILTERS)
        raise ValueError(
            "recovery_accept_positive_context_filter must be one of: "
            f"{names}"
        )
    if (
        recovery_accept_positive_context_oversample_factor > 1
        and recovery_accept_positive_context_filter == "off"
    ):
        raise ValueError(
            "recovery_accept_positive_context_oversample_factor requires "
            "a recovery_accept_positive_context_filter"
        )
    action_loss_weights = _normalize_recovery_action_loss_weights(
        recovery_accept_positive_action_loss_weights
    )
    allowed_uses = SIGNAL_FILTER_PRESETS[filter_name]
    signal_dataset = build_strategy_signal_dataset(
        paths,
        include_before_filter_candidates=False,
    )
    signal_by_key = {
        _record_key(record.path, record.step, record.candidate_action): record
        for record in signal_dataset.records
        if record.candidate_source == "recorded"
    }
    examples = tuple(
        iter_strategy_trajectory_examples(
            paths,
            include_terminal=include_terminal,
            allow_observation_defaults=allow_observation_defaults,
        )
    )

    eligible: list[tuple[StrategyTrajectoryExample, str]] = []
    kept_uses: Counter[str] = Counter()
    removed_uses: Counter[str] = Counter()
    kept_actions: Counter[int] = Counter()
    removed_actions: Counter[int] = Counter()
    recovery_removed_actions: Counter[int] = Counter()
    recovery_removed = 0
    recovery_positive_examples: list[StrategyTrajectoryExample] = []
    recovery_positive_oversampled_actions: Counter[int] = Counter()
    recovery_positive_added = 0
    context_positive_oversampled_actions: Counter[int] = Counter()
    context_positive_added = 0
    recovery_safe_filter = filter_name == "trainable-recovery-safe"

    for example in examples:
        action_name = STRATEGY_ACTION_NAMES.get(int(example.action), str(example.action))
        key = _record_key(example.source_path, example.step, action_name)
        signal = signal_by_key.get(key)
        training_use = (
            signal.recommended_training_use if signal is not None else "missing_signal"
        )
        if (
            recovery_safe_filter
            and training_use == "drop_ambiguous"
            and action_name == "STAY_COURSE"
            and _executable_recovery_actions(example)
        ):
            recovery_removed += 1
            removed_uses["drop_ambiguous_recovery_opportunity"] += 1
            removed_actions[int(example.action)] += 1
            recovery_removed_actions[int(example.action)] += 1
        elif training_use in allowed_uses:
            eligible.append((example, training_use))
        else:
            removed_uses[training_use] += 1
            removed_actions[int(example.action)] += 1

    positive_examples = sum(1 for _example, use in eligible if use == "accept_positive")
    ambiguous_indexes = [
        index for index, (_example, use) in enumerate(eligible) if use == "drop_ambiguous"
    ]
    selected_ambiguous_indexes = set(ambiguous_indexes)
    ambiguous_removed_by_balance = 0
    if max_drop_ambiguous_per_positive is not None:
        max_ambiguous = int(max_drop_ambiguous_per_positive * positive_examples)
        if max_ambiguous < len(ambiguous_indexes):
            if max_ambiguous <= 0:
                selected_ambiguous_indexes = set()
            else:
                rng = np.random.default_rng(balance_seed)
                selected = rng.choice(
                    np.asarray(ambiguous_indexes, dtype=np.int64),
                    size=max_ambiguous,
                    replace=False,
                )
                selected_ambiguous_indexes = {int(index) for index in selected.tolist()}
            ambiguous_removed_by_balance = len(ambiguous_indexes) - len(
                selected_ambiguous_indexes
            )

    kept: list[StrategyTrajectoryExample] = []
    kept_training_uses: list[str] = []
    for index, (example, training_use) in enumerate(eligible):
        if training_use == "drop_ambiguous" and index not in selected_ambiguous_indexes:
            removed_uses[training_use] += 1
            removed_actions[int(example.action)] += 1
            continue
        kept.append(example)
        kept_training_uses.append(training_use)
        kept_uses[training_use] += 1
        kept_actions[int(example.action)] += 1
        if (
            training_use == "accept_positive"
            and STRATEGY_ACTION_NAMES.get(int(example.action)) in RECOVERY_ACTION_NAMES
        ):
            recovery_positive_examples.append(example)

    selected_original_examples = len(kept)
    context_positive_examples = [
        example
        for example in recovery_positive_examples
        if _recovery_accept_positive_context_matches(
            example,
            STRATEGY_ACTION_NAMES.get(int(example.action)),
            recovery_accept_positive_context_filter,
        )
    ]
    if recovery_positive_oversample_factor > 1:
        for example in recovery_positive_examples:
            action_id = int(example.action)
            for _ in range(recovery_positive_oversample_factor - 1):
                kept.append(example)
                kept_training_uses.append("accept_positive")
                kept_uses["accept_positive"] += 1
                kept_actions[action_id] += 1
                recovery_positive_oversampled_actions[action_id] += 1
                recovery_positive_added += 1

    if recovery_accept_positive_context_oversample_factor > 1:
        for example in context_positive_examples:
            action_id = int(example.action)
            for _ in range(
                recovery_accept_positive_context_oversample_factor - 1
            ):
                kept.append(example)
                kept_training_uses.append("accept_positive")
                kept_uses["accept_positive"] += 1
                kept_actions[action_id] += 1
                context_positive_oversampled_actions[action_id] += 1
                context_positive_added += 1

    weighted_actions: Counter[int] = Counter()
    context_matched_actions: Counter[int] = Counter()
    context_skipped_actions: Counter[int] = Counter()
    sample_weight_values: list[float] = []
    for example, training_use in zip(kept, kept_training_uses):
        action_name = STRATEGY_ACTION_NAMES.get(int(example.action))
        is_recovery_accept_positive = (
            training_use == "accept_positive"
            and action_name in RECOVERY_ACTION_NAMES
        )
        context_matches = (
            _recovery_accept_positive_context_matches(
                example,
                action_name,
                recovery_accept_positive_context_filter,
            )
            if is_recovery_accept_positive
            else False
        )
        if is_recovery_accept_positive:
            if context_matches:
                context_matched_actions[int(example.action)] += 1
            else:
                context_skipped_actions[int(example.action)] += 1
        weight = _sample_weight(
            example,
            training_use,
            recovery_accept_positive_loss_weight=(
                recovery_accept_positive_loss_weight
            ),
            recovery_accept_positive_action_loss_weights=action_loss_weights,
            context_matches=context_matches,
        )
        sample_weight_values.append(weight)
        if weight > 1.0:
            weighted_actions[int(example.action)] += 1
    sample_weights = np.asarray(sample_weight_values, dtype=np.float32)
    dataset = _dataset_from_examples(tuple(kept))
    summary = StrategySignalFilterSummary(
        filter_name=filter_name,
        allowed_training_uses=allowed_uses,
        total_signal_records=len(signal_dataset.records),
        original_examples=len(examples),
        kept_examples=dataset.size,
        removed_examples=len(examples) - selected_original_examples,
        kept_by_training_use=_sorted_counts(kept_uses),
        removed_by_training_use=_sorted_counts(removed_uses),
        kept_action_counts=_sorted_int_counts(kept_actions),
        removed_action_counts=_sorted_int_counts(removed_actions),
        kept_action_counts_by_name=_counts_by_name(kept_actions),
        removed_action_counts_by_name=_counts_by_name(removed_actions),
        max_drop_ambiguous_per_positive=max_drop_ambiguous_per_positive,
        balance_seed=(
            balance_seed if max_drop_ambiguous_per_positive is not None else None
        ),
        positive_examples_for_balance=positive_examples,
        drop_ambiguous_examples_before_balance=len(ambiguous_indexes),
        drop_ambiguous_examples_kept=kept_uses.get("drop_ambiguous", 0),
        drop_ambiguous_examples_removed_by_balance=ambiguous_removed_by_balance,
        recovery_safe_filter_enabled=recovery_safe_filter,
        recovery_opportunity_ambiguous_examples_removed=recovery_removed,
        recovery_opportunity_removed_actions=_sorted_int_counts(
            recovery_removed_actions
        ),
        recovery_opportunity_removed_actions_by_name=_counts_by_name(
            recovery_removed_actions
        ),
        recovery_positive_oversample_factor=recovery_positive_oversample_factor,
        recovery_positive_examples_before_oversample=len(recovery_positive_examples),
        recovery_positive_examples_added_by_oversample=recovery_positive_added,
        recovery_positive_oversampled_actions=_sorted_int_counts(
            recovery_positive_oversampled_actions
        ),
        recovery_positive_oversampled_actions_by_name=_counts_by_name(
            recovery_positive_oversampled_actions
        ),
        recovery_accept_positive_loss_weight=float(
            recovery_accept_positive_loss_weight
        ),
        recovery_accept_positive_action_loss_weights=(
            action_loss_weights if action_loss_weights else None
        ),
        recovery_accept_positive_context_filter=(
            recovery_accept_positive_context_filter
        ),
        recovery_accept_positive_context_matched_examples=sum(
            context_matched_actions.values()
        ),
        recovery_accept_positive_context_skipped_examples=sum(
            context_skipped_actions.values()
        ),
        recovery_accept_positive_context_matched_actions_by_name=_counts_by_name(
            context_matched_actions
        ),
        recovery_accept_positive_context_skipped_actions_by_name=_counts_by_name(
            context_skipped_actions
        ),
        recovery_accept_positive_context_oversample_factor=(
            recovery_accept_positive_context_oversample_factor
        ),
        recovery_accept_positive_context_examples_before_oversample=len(
            context_positive_examples
        ),
        recovery_accept_positive_context_examples_added_by_oversample=(
            context_positive_added
        ),
        recovery_accept_positive_context_oversampled_actions_by_name=_counts_by_name(
            context_positive_oversampled_actions
        ),
        recovery_accept_positive_weighted_examples=int(
            np.count_nonzero(sample_weights > 1.0)
        ),
        recovery_accept_positive_weighted_actions_by_name=_counts_by_name(
            weighted_actions
        ),
        sample_weight_sum=float(sample_weights.sum()),
    )
    return SignalFilteredStrategyDataset(
        dataset=dataset,
        summary=summary,
        sample_weights=sample_weights,
    )


def _dataset_from_examples(
    examples: tuple[StrategyTrajectoryExample, ...],
) -> StrategyTrajectoryDataset:
    if not examples:
        observations = np.empty(
            (0, len(STRATEGY_OBSERVATION_FIELDS)),
            dtype=np.float32,
        )
        actions = np.empty((0,), dtype=np.int64)
    else:
        observations = np.stack([example.observation for example in examples]).astype(
            np.float32
        )
        actions = np.asarray([example.action for example in examples], dtype=np.int64)

    return StrategyTrajectoryDataset(
        observations=observations,
        actions=actions,
        examples=examples,
        action_counts=dict(Counter(int(action) for action in actions)),
        observation_schema_counts=dict(
            Counter(str(example.observation_schema_version) for example in examples)
        ),
        rows_defaulted_observation_fields=sum(
            1 for example in examples if example.defaulted_observation_fields
        ),
    )


def _record_key(path: str | Path, step: int, action_name: str) -> tuple[str, int, str]:
    return (str(Path(path).resolve()), int(step), action_name)


def _executable_recovery_actions(example: StrategyTrajectoryExample) -> list[str]:
    observation = _observation_from_example(example)
    return [
        action
        for action in RECOVERY_ACTION_NAMES
        if candidate_executability_from_observation(observation, action)[0]
    ]


def _sample_weight(
    example: StrategyTrajectoryExample,
    training_use: str,
    *,
    recovery_accept_positive_loss_weight: float,
    recovery_accept_positive_action_loss_weights: dict[str, float],
    context_matches: bool,
) -> float:
    action_name = STRATEGY_ACTION_NAMES.get(int(example.action))
    if (
        training_use == "accept_positive"
        and action_name in RECOVERY_ACTION_NAMES
        and context_matches
    ):
        if action_name in recovery_accept_positive_action_loss_weights:
            return float(recovery_accept_positive_action_loss_weights[action_name])
        return float(recovery_accept_positive_loss_weight)
    return 1.0


def _recovery_accept_positive_context_matches(
    example: StrategyTrajectoryExample,
    action_name: str | None,
    context_filter: str,
) -> bool:
    observation = _observation_from_example(example)
    return recovery_accept_positive_context_matches_observation(
        observation,
        action_name,
        context_filter,
    )


def recovery_accept_positive_context_matches_observation(
    observation: dict[str, float],
    action_name: str | None,
    context_filter: str,
) -> bool:
    """Return whether a recovery action belongs to an opt-in context slice."""
    if context_filter == "off":
        return True
    if action_name not in RECOVERY_ACTION_NAMES:
        return False
    executable, _blocker = candidate_executability_from_observation(
        observation,
        action_name,
    )
    if not executable:
        return False
    if context_filter == "pre-collapse-recovery":
        return _pre_collapse_recovery_context_matches(observation, action_name)
    raise ValueError(
        "recovery_accept_positive_context_filter must be one of: "
        f"{', '.join(RECOVERY_ACCEPT_POSITIVE_CONTEXT_FILTERS)}"
    )


def _pre_collapse_recovery_context_matches(
    observation: dict[str, float],
    action_name: str,
) -> bool:
    if action_name == "TECH_ROBO":
        robo_count = _value(observation, "ready_robo") + _value(
            observation,
            "pending_robo",
        )
        return (
            _value(observation, "game_time") >= PRE_COLLAPSE_ROBO_TIME
            and _value(observation, "base_under_threat") <= 0.0
            and _value(observation, "has_cybernetics_core") > 0.0
            and robo_count <= 0.0
            and _value(observation, "minerals") >= ROBO_MINERALS
            and _value(observation, "vespene")
            >= PRE_COLLAPSE_ROBO_VESPENE_BANK
        )
    if action_name == "BUILD_STATIC_DEFENSE":
        bases = max(_value(observation, "own_bases"), 1.0)
        static_defense_count = _value(
            observation,
            "ready_static_defense",
        ) + _value(observation, "pending_static_defense")
        can_build_cannon = (
            _value(observation, "ready_forge") > 0.0
            and _value(observation, "minerals") >= PHOTON_CANNON_MINERALS
        )
        can_build_battery = (
            _value(observation, "has_cybernetics_core") > 0.0
            and _value(observation, "minerals") >= STATIC_DEFENSE_MINERALS
        )
        return (
            _value(observation, "game_time") >= PRE_COLLAPSE_STATIC_DEFENSE_TIME
            and _value(observation, "base_under_threat") <= 0.0
            and static_defense_count < bases
            and (can_build_cannon or can_build_battery)
        )
    if action_name == "PRODUCE_ARMY":
        idle_production = _value(observation, "gateway_idle_count") + _value(
            observation,
            "robo_idle_count",
        )
        return (
            _value(observation, "supply_left") > 0.0
            and _value(observation, "minerals") >= PRODUCE_ARMY_MINERALS
            and (
                idle_production > 0.0
                or _value(observation, "army_count") < DESIRED_MIN_ARMY
            )
        )
    return False


def _observation_from_example(
    example: StrategyTrajectoryExample,
) -> dict[str, float]:
    return {
        field: float(example.observation[index])
        for index, field in enumerate(STRATEGY_OBSERVATION_FIELDS)
    }


def _value(observation: dict[str, float], field: str) -> float:
    return float(observation.get(field, 0.0))


def _normalize_recovery_action_loss_weights(
    values: dict[str, float] | None,
) -> dict[str, float]:
    if not values:
        return {}
    normalized: dict[str, float] = {}
    for action_name, weight in values.items():
        action = str(action_name)
        if action not in RECOVERY_ACTION_NAMES:
            names = ", ".join(RECOVERY_ACTION_NAMES)
            raise ValueError(
                "recovery_accept_positive_action_loss_weights keys must be "
                f"one of: {names}"
            )
        weight_value = float(weight)
        if weight_value < 1.0:
            raise ValueError(
                "recovery_accept_positive_action_loss_weights values must be >= 1.0"
            )
        normalized[action] = weight_value
    return dict(sorted(normalized.items()))


def _counts_by_name(counts: Counter[int]) -> dict[str, int]:
    return {
        STRATEGY_ACTION_NAMES[action_id]: int(count)
        for action_id, count in sorted(counts.items())
        if action_id in STRATEGY_ACTION_NAMES
    }


def _sorted_counts(counts: Counter[str]) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in counts.items()))


def _sorted_int_counts(counts: Counter[int]) -> dict[int, int]:
    return dict(sorted((int(name), int(count)) for name, count in counts.items()))
