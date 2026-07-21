from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl.ppo_surrogate_backend import ScenarioStrategyBackend
from scripts.train_ppo import resolve_backend_factory


@pytest.mark.unit
def test_train_ppo_cli_resolves_surrogate_without_starting_training() -> None:
    factory = resolve_backend_factory(
        SimpleNamespace(
            backend="surrogate",
            backend_factory=None,
            surrogate_max_steps=3,
        )
    )

    backend = factory()

    assert isinstance(backend, ScenarioStrategyBackend)
    assert backend.max_steps == 3


@pytest.mark.unit
def test_train_ppo_cli_rejects_ambiguous_backend_configuration() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_backend_factory(
            SimpleNamespace(
                backend="surrogate",
                backend_factory="package.module:factory",
                surrogate_max_steps=3,
            )
        )
