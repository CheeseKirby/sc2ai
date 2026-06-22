"""Trajectory loading utilities for imitation learning and analysis."""
from __future__ import annotations

import json
from collections.abc import Iterable as IterableABC
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from rl.observations import (
    OBSERVATION_DEFAULTS,
    OBSERVATION_FIELDS,
    infer_observation_schema_version,
    observation_dict_to_vector,
)

TrajectoryPathInput = str | Path | Iterable[str | Path]


@dataclass(frozen=True)
class TrajectoryExample:
    """One supervised learning example from a trajectory JSONL file."""

    observation: np.ndarray
    action: int
    source_path: Path
    step: int
    done: bool
    result: str | None
    observation_schema_version: int | None
    defaulted_observation_fields: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryDataset:
    """In-memory arrays plus useful metadata for framework scripts."""

    observations: np.ndarray
    actions: np.ndarray
    examples: tuple[TrajectoryExample, ...]
    action_counts: dict[int, int]
    observation_schema_counts: dict[str, int]
    rows_defaulted_observation_fields: int

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[1]) if self.size else len(OBSERVATION_FIELDS)


def _normalize_paths(paths: TrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def discover_trajectory_files(paths: TrajectoryPathInput) -> list[Path]:
    """Expand files/directories into a sorted list of JSONL trajectory files."""
    files: list[Path] = []
    for raw_path in _normalize_paths(paths):
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            files.append(path)
    return sorted(dict.fromkeys(files))


def iter_trajectory_examples(
    paths: TrajectoryPathInput,
    *,
    include_terminal: bool = False,
    allow_observation_defaults: bool = True,
) -> Iterator[TrajectoryExample]:
    """Yield trajectory examples from JSONL files."""
    for path in discover_trajectory_files(paths):
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                done = bool(row.get("done", False))
                if done and not include_terminal:
                    continue
                observation = row["observation"]
                yield TrajectoryExample(
                    observation=observation_dict_to_vector(
                        observation,
                        allow_missing_defaults=allow_observation_defaults,
                    ),
                    action=int(row["action"]),
                    source_path=path,
                    step=int(row.get("step", line_number)),
                    done=done,
                    result=row.get("result"),
                    observation_schema_version=infer_observation_schema_version(
                        observation
                    ),
                    defaulted_observation_fields=tuple(
                        field
                        for field in OBSERVATION_DEFAULTS
                        if field not in observation
                    ),
                )


def load_trajectory_dataset(
    paths: TrajectoryPathInput,
    *,
    include_terminal: bool = False,
    allow_observation_defaults: bool = True,
) -> TrajectoryDataset:
    """Load trajectory examples into numpy arrays."""
    examples = tuple(
        iter_trajectory_examples(
            paths,
            include_terminal=include_terminal,
            allow_observation_defaults=allow_observation_defaults,
        )
    )
    if not examples:
        observations = np.empty((0, len(OBSERVATION_FIELDS)), dtype=np.float32)
        actions = np.empty((0,), dtype=np.int64)
    else:
        observations = np.stack([example.observation for example in examples]).astype(
            np.float32
        )
        actions = np.asarray([example.action for example in examples], dtype=np.int64)

    return TrajectoryDataset(
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
