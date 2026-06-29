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
from rl.strategy_signal_dataset import build_strategy_signal_dataset


SIGNAL_FILTER_PRESETS: dict[str, tuple[str, ...]] = {
    "strict-positive": ("accept_positive",),
    "trainable": ("accept_positive", "drop_ambiguous", "weak_context"),
}


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


@dataclass(frozen=True)
class SignalFilteredStrategyDataset:
    """Filtered dataset plus its signal-filter summary."""

    dataset: StrategyTrajectoryDataset
    summary: StrategySignalFilterSummary


def load_signal_filtered_strategy_trajectory_dataset(
    paths: StrategyTrajectoryPathInput,
    *,
    filter_name: str = "trainable",
    include_terminal: bool = False,
    allow_observation_defaults: bool = True,
) -> SignalFilteredStrategyDataset:
    """Load strategy examples after dropping low-quality signal rows."""
    if filter_name not in SIGNAL_FILTER_PRESETS:
        names = ", ".join(sorted(SIGNAL_FILTER_PRESETS))
        raise ValueError(f"Unknown signal filter {filter_name!r}; expected one of {names}")
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

    kept: list[StrategyTrajectoryExample] = []
    kept_uses: Counter[str] = Counter()
    removed_uses: Counter[str] = Counter()
    kept_actions: Counter[int] = Counter()
    removed_actions: Counter[int] = Counter()

    for example in examples:
        action_name = STRATEGY_ACTION_NAMES.get(int(example.action), str(example.action))
        key = _record_key(example.source_path, example.step, action_name)
        signal = signal_by_key.get(key)
        training_use = (
            signal.recommended_training_use if signal is not None else "missing_signal"
        )
        if training_use in allowed_uses:
            kept.append(example)
            kept_uses[training_use] += 1
            kept_actions[int(example.action)] += 1
        else:
            removed_uses[training_use] += 1
            removed_actions[int(example.action)] += 1

    dataset = _dataset_from_examples(tuple(kept))
    summary = StrategySignalFilterSummary(
        filter_name=filter_name,
        allowed_training_uses=allowed_uses,
        total_signal_records=len(signal_dataset.records),
        original_examples=len(examples),
        kept_examples=dataset.size,
        removed_examples=len(examples) - dataset.size,
        kept_by_training_use=_sorted_counts(kept_uses),
        removed_by_training_use=_sorted_counts(removed_uses),
        kept_action_counts=_sorted_int_counts(kept_actions),
        removed_action_counts=_sorted_int_counts(removed_actions),
        kept_action_counts_by_name=_counts_by_name(kept_actions),
        removed_action_counts_by_name=_counts_by_name(removed_actions),
    )
    return SignalFilteredStrategyDataset(dataset=dataset, summary=summary)


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
