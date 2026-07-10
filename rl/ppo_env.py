"""Gymnasium adapter for a future StarCraft II strategy backend."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.ppo_rewards import StrategyRewardCalculator
from rl.ppo_types import StrategyEnvBackend
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import (
    STRATEGY_OBSERVATION_FIELDS,
    STRATEGY_OBSERVATION_SCHEMA_VERSION,
    normalize_strategy_observation_dict,
    strategy_observation_dict_to_vector,
)


class TransitionContractError(RuntimeError):
    """Raised when a backend returns a transition with inconsistent state."""


class SC2StrategyPPOEnv(gym.Env):
    """Thin Gymnasium environment around an injected SC2/replay backend.

    The environment deliberately has no built-in live SC2 launcher. A backend
    must provide a complete state-before/action/execution/state-after contract.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: StrategyEnvBackend,
        *,
        reward_calculator: StrategyRewardCalculator | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.reward_calculator = reward_calculator or StrategyRewardCalculator()
        self.action_space = spaces.Discrete(len(StrategyAction))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(STRATEGY_OBSERVATION_FIELDS),),
            dtype=np.float32,
        )
        self._current_observation: dict[str, float] | None = None
        self._current_vector: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observation = _normalize_observation(
            self.backend.reset(seed=seed, options=options)
        )
        vector = strategy_observation_dict_to_vector(observation)
        self._current_observation = observation
        self._current_vector = vector
        return vector.copy(), {
            "schema": STRATEGY_OBSERVATION_SCHEMA_VERSION,
            "observation_fields": list(STRATEGY_OBSERVATION_FIELDS),
        }

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._current_observation is None or self._current_vector is None:
            raise RuntimeError("reset() must be called before step()")
        try:
            selected_action = StrategyAction(int(action))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid strategy action: {action!r}") from exc

        transition = self.backend.step(selected_action)
        if transition.action is not selected_action:
            raise TransitionContractError(
                "backend transition action does not match the requested action"
            )

        state_before = _normalize_observation(transition.state_before)
        before_vector = strategy_observation_dict_to_vector(state_before)
        if not np.allclose(before_vector, self._current_vector, rtol=1e-5, atol=1e-5):
            raise TransitionContractError(
                "backend state_before does not match the environment's current state"
            )

        state_after = _normalize_observation(transition.state_after)
        after_vector = strategy_observation_dict_to_vector(state_after)
        reward = self.reward_calculator.calculate(transition)
        info = dict(transition.info)
        info.update(
            {
                "action_name": selected_action.name,
                "execution_result": asdict(transition.execution_result),
                "outcome": transition.outcome,
            }
        )
        self._current_observation = state_after
        self._current_vector = after_vector
        return (
            after_vector.copy(),
            reward,
            bool(transition.terminated),
            bool(transition.truncated),
            info,
        )

    def close(self) -> None:
        self.backend.close()
        self._current_observation = None
        self._current_vector = None


def _normalize_observation(observation: dict[str, float]) -> dict[str, float]:
    return normalize_strategy_observation_dict(
        dict(observation),
        allow_missing_defaults=False,
    )
