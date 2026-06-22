"""Checkpoint-backed army policy inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bot.managers.army_policy import ArmyAction
from bot.managers.rule_army_policy import COMBAT_UNIT_TYPES, RuleArmyPolicy
from rl.checkpoints import LoadedPolicyCheckpoint, load_policy_checkpoint
from rl.normalization import ObservationNormalizer
from rl.observations import build_observation


class RLArmyPolicy:
    """Load a policy checkpoint and use it for high-level army decisions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.loaded: LoadedPolicyCheckpoint = load_policy_checkpoint(
            self.checkpoint_path,
            map_location=self.device,
        )
        self.model = self.loaded.model.to(self.device)
        self.model.eval()
        self.normalizer = (
            ObservationNormalizer.from_dict(self.loaded.metadata.normalizer)
            if self.loaded.metadata.normalizer is not None
            else None
        )
        self.executor = RuleArmyPolicy()

    def manage_army(self, bot: Any) -> ArmyAction:
        """Choose an action with the loaded model and execute it."""
        action = self.decide(bot)
        self._apply_action_state(bot, action)
        army = bot.units.of_type(COMBAT_UNIT_TYPES)
        self.executor.execute(bot, action, army)
        return action

    @torch.no_grad()
    def decide(self, bot: Any) -> ArmyAction:
        """Return the greedy model action for the current bot state."""
        observation = build_observation(bot).to_vector()
        if self.normalizer is not None:
            observation = self.normalizer.transform(observation)
        tensor = torch.from_numpy(observation).to(self.device)
        return ArmyAction(self.model.predict_action(tensor))

    def _apply_action_state(self, bot: Any, action: ArmyAction) -> None:
        if action is ArmyAction.ATTACK_MAIN:
            bot.is_attacking = True
        elif action in {
            ArmyAction.RALLY,
            ArmyAction.RETREAT_HOME,
            ArmyAction.DEFEND_BASE,
        }:
            bot.is_attacking = False
