"""Stable-Baselines3 PPO adapter for low-frequency strategy inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from bot.managers.strategy_policy import write_strategy_decision_metadata
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import build_strategy_observation


class PredictivePolicy(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[Any, Any]:
        """Return a Stable-Baselines3-style action and optional state."""


ObservationBuilder = Callable[[Any], np.ndarray]


class PPOStrategyPolicy:
    """Use a Stable-Baselines3 PPO checkpoint for strategy decisions."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str = "cpu",
        model: PredictivePolicy | None = None,
        observation_builder: ObservationBuilder | None = None,
    ) -> None:
        if model is None:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when model is omitted")
            from stable_baselines3 import PPO

            model = PPO.load(str(checkpoint_path), device=device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = model
        self.observation_builder = observation_builder or _build_observation_vector
        self.last_decision_source = "ppo"
        self.last_decision_reason = "uninitialized"

    def decide_strategy(self, bot: Any) -> StrategyAction:
        observation = np.asarray(self.observation_builder(bot), dtype=np.float32)
        raw_action, _state = self.model.predict(observation, deterministic=True)
        action_values = np.asarray(raw_action).reshape(-1)
        if action_values.size != 1:
            raise ValueError(
                "PPO strategy policy expected one action, "
                f"received shape {np.asarray(raw_action).shape}"
            )
        try:
            action = StrategyAction(int(action_values[0]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid PPO strategy action: {raw_action!r}") from exc

        self.last_decision_source = "ppo"
        self.last_decision_reason = f"ppo_deterministic_action:{action.name}"
        write_strategy_decision_metadata(
            bot,
            source=self.last_decision_source,
            reason=self.last_decision_reason,
        )
        return action


def _build_observation_vector(bot: Any) -> np.ndarray:
    return build_strategy_observation(bot).to_vector()
