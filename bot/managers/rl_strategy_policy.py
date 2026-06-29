"""Checkpoint-backed low-frequency strategy policy inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bot.managers.strategy_executor import StrategyExecutor
from bot.managers.strategy_policy import write_strategy_decision_metadata
from rl.normalization import ObservationNormalizer
from rl.strategy_actions import StrategyAction
from rl.strategy_checkpoints import (
    LoadedStrategyPolicyCheckpoint,
    load_strategy_policy_checkpoint,
)
from rl.strategy_observations import build_strategy_observation
from rl.strategy_observations import (
    STRATEGY_OBSERVATION_FIELDS,
    STRATEGY_OBSERVATION_SCHEMA_VERSION,
)


class RLStrategyPolicy:
    """Load a strategy checkpoint and use it for macro strategy decisions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.loaded: LoadedStrategyPolicyCheckpoint = load_strategy_policy_checkpoint(
            self.checkpoint_path,
            map_location=self.device,
        )
        self.model = self.loaded.model.to(self.device)
        self.model.eval()
        self.normalizer = (
            ObservationNormalizer.from_dict(
                self.loaded.metadata.normalizer,
                expected_fields=STRATEGY_OBSERVATION_FIELDS,
                expected_schema_version=STRATEGY_OBSERVATION_SCHEMA_VERSION,
            )
            if self.loaded.metadata.normalizer is not None
            else None
        )
        self.executor = StrategyExecutor()
        self.last_decision_source = "checkpoint"
        self.last_decision_reason = "uninitialized"

    @torch.no_grad()
    def decide_strategy(self, bot: Any) -> StrategyAction:
        """Return the greedy strategy action for the current bot state."""
        observation = build_strategy_observation(bot).to_vector()
        if self.normalizer is not None:
            observation = self.normalizer.transform(observation)
        tensor = torch.from_numpy(observation).to(self.device)
        action = StrategyAction(self.model.predict_action(tensor))
        self.last_decision_source = "checkpoint"
        self.last_decision_reason = "checkpoint_greedy_action"
        write_strategy_decision_metadata(
            bot,
            source=self.last_decision_source,
            reason=self.last_decision_reason,
        )
        return action

    async def decide_and_execute(self, bot: Any) -> StrategyAction:
        """Choose and execute a strategy action.

        ProtossRuleBot normally handles execution after calling decide_strategy().
        This helper keeps the policy directly testable.
        """
        action = self.decide_strategy(bot)
        await self.executor.execute(bot, action)
        return action
