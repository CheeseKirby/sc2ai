"""Observation normalization shared by training and inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.observations import OBSERVATION_FIELDS, OBSERVATION_SCHEMA_VERSION


@dataclass(frozen=True)
class ObservationNormalizer:
    """Mean/std normalizer for stable observation vectors."""

    mean: np.ndarray
    std: np.ndarray
    fields: tuple[str, ...] = OBSERVATION_FIELDS
    schema_version: int | str = OBSERVATION_SCHEMA_VERSION
    epsilon: float = 1e-6

    def transform(self, observations: np.ndarray) -> np.ndarray:
        """Normalize one vector or a batch of vectors."""
        array = np.asarray(observations, dtype=np.float32)
        if array.shape[-1] != len(self.fields):
            raise ValueError(
                f"Observation dim mismatch: {array.shape[-1]} != {len(self.fields)}"
            )
        return ((array - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        """Return a JSON/checkpoint friendly payload."""
        return {
            "schema_version": self.schema_version,
            "fields": list(self.fields),
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict,
        *,
        expected_fields: tuple[str, ...] = OBSERVATION_FIELDS,
        expected_schema_version: int | str = OBSERVATION_SCHEMA_VERSION,
    ) -> ObservationNormalizer:
        """Rebuild a normalizer from metadata."""
        fields = tuple(payload["fields"])
        if fields != expected_fields:
            raise ValueError("Normalizer observation fields do not match current schema")
        schema_version = payload["schema_version"]
        if isinstance(expected_schema_version, int):
            schema_version = int(schema_version)
        else:
            schema_version = str(schema_version)
        if schema_version != expected_schema_version:
            raise ValueError(
                "Normalizer schema version mismatch: "
                f"{schema_version} != {expected_schema_version}"
            )
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            fields=fields,
            schema_version=schema_version,
            epsilon=float(payload.get("epsilon", 1e-6)),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ObservationNormalizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_observation_normalizer(
    observations: np.ndarray,
    *,
    epsilon: float = 1e-6,
    fields: tuple[str, ...] = OBSERVATION_FIELDS,
    schema_version: int | str = OBSERVATION_SCHEMA_VERSION,
) -> ObservationNormalizer:
    """Fit mean/std stats from a 2D observation matrix."""
    array = np.asarray(observations, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Expected a 2D observation matrix")
    if array.shape[0] == 0:
        raise ValueError("Cannot fit normalizer on an empty observation matrix")
    if array.shape[1] != len(fields):
        raise ValueError(
            f"Observation dim mismatch: {array.shape[1]} != {len(fields)}"
        )
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std = np.where(std < epsilon, 1.0, std).astype(np.float32)
    return ObservationNormalizer(
        mean=mean.astype(np.float32),
        std=std,
        fields=fields,
        schema_version=schema_version,
        epsilon=epsilon,
    )
