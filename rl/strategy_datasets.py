"""Strategy trajectory loading utilities for imitation learning."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from rl.strategy_observations import (
    STRATEGY_OBSERVATION_DEFAULTS,
    STRATEGY_OBSERVATION_FIELDS,
    infer_strategy_observation_schema_version,
    strategy_observation_dict_to_vector,
)

StrategyTrajectoryPathInput = str | Path | Iterable[str | Path]


@dataclass(frozen=True)
class StrategyTrajectoryExample:
    """One supervised strategy example from a strategy trajectory row."""

    observation: np.ndarray
    action: int
    source_path: Path
    step: int
    done: bool
    result: str | None
    observation_schema_version: str | None
    defaulted_observation_fields: tuple[str, ...]


@dataclass(frozen=True)
class StrategyTrajectoryDataset:
    """In-memory strategy arrays plus metadata for training and diagnostics."""

    observations: np.ndarray
    actions: np.ndarray
    examples: tuple[StrategyTrajectoryExample, ...]
    action_counts: dict[int, int]
    observation_schema_counts: dict[str, int]
    rows_defaulted_observation_fields: int

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def observation_dim(self) -> int:
        if self.size:
            return int(self.observations.shape[1])
        return len(STRATEGY_OBSERVATION_FIELDS)


def _normalize_paths(
    paths: StrategyTrajectoryPathInput,
) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def discover_strategy_trajectory_files(
    paths: StrategyTrajectoryPathInput,
) -> list[Path]:
    """Expand files/directories into sorted JSONL strategy trajectory files."""
    files: list[Path] = []
    for raw_path in _normalize_paths(paths):
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            files.append(path)
    return sorted(dict.fromkeys(files))


def iter_strategy_trajectory_examples(
    paths: StrategyTrajectoryPathInput,
    *,
    include_terminal: bool = False,
    allow_observation_defaults: bool = True,
) -> Iterator[StrategyTrajectoryExample]:
    """Yield strategy examples from JSONL files."""
    for path in discover_strategy_trajectory_files(paths):
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                done = bool(row.get("done", False))
                if done and not include_terminal:
                    continue
                observation = row["strategy_observation"]
                yield StrategyTrajectoryExample(
                    observation=strategy_observation_dict_to_vector(
                        observation,
                        allow_missing_defaults=allow_observation_defaults,
                    ),
                    action=int(row["strategy_action"]),
                    source_path=path,
                    step=int(row.get("step", line_number)),
                    done=done,
                    result=row.get("result"),
                    observation_schema_version=(
                        infer_strategy_observation_schema_version(observation)
                    ),
                    defaulted_observation_fields=tuple(
                        field
                        for field in STRATEGY_OBSERVATION_DEFAULTS
                        if field not in observation
                    ),
                )


def load_strategy_trajectory_dataset(
    paths: StrategyTrajectoryPathInput,
    *,
    include_terminal: bool = False,
    allow_observation_defaults: bool = True,
) -> StrategyTrajectoryDataset:
    """Load strategy trajectory examples into numpy arrays."""
    examples = tuple(
        iter_strategy_trajectory_examples(
            paths,
            include_terminal=include_terminal,
            allow_observation_defaults=allow_observation_defaults,
        )
    )
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
            Counter(
                str(example.observation_schema_version)
                for example in examples
            )
        ),
        rows_defaulted_observation_fields=sum(
            1 for example in examples if example.defaulted_observation_fields
        ),
    )
