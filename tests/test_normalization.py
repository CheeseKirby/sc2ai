from __future__ import annotations

import numpy as np
import pytest

from rl.normalization import ObservationNormalizer, fit_observation_normalizer
from rl.observations import OBSERVATION_FIELDS


@pytest.mark.unit
def test_fit_observation_normalizer_transforms_to_zero_mean() -> None:
    observations = np.asarray(
        [
            [1.0] * len(OBSERVATION_FIELDS),
            [3.0] * len(OBSERVATION_FIELDS),
        ],
        dtype=np.float32,
    )

    normalizer = fit_observation_normalizer(observations)
    transformed = normalizer.transform(observations)

    assert np.allclose(transformed.mean(axis=0), 0.0)
    assert np.allclose(transformed[:, 0], [-1.0, 1.0])


@pytest.mark.unit
def test_normalizer_handles_constant_columns() -> None:
    observations = np.ones((3, len(OBSERVATION_FIELDS)), dtype=np.float32)

    normalizer = fit_observation_normalizer(observations)

    assert np.allclose(normalizer.std, 1.0)
    assert np.allclose(normalizer.transform(observations), 0.0)


@pytest.mark.unit
def test_normalizer_round_trip_json(tmp_path) -> None:
    observations = np.arange(
        3 * len(OBSERVATION_FIELDS),
        dtype=np.float32,
    ).reshape(3, len(OBSERVATION_FIELDS))
    normalizer = fit_observation_normalizer(observations)
    path = tmp_path / "normalizer.json"

    normalizer.save(path)
    loaded = ObservationNormalizer.load(path)

    assert loaded.fields == OBSERVATION_FIELDS
    assert np.allclose(loaded.mean, normalizer.mean)
    assert np.allclose(loaded.std, normalizer.std)

