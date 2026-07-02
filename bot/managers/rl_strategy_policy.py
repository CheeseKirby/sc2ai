"""Checkpoint-backed low-frequency strategy policy inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bot.managers.strategy_executor import StrategyExecutor
from bot.managers.strategy_policy import write_strategy_decision_metadata
from rl.normalization import ObservationNormalizer
from rl.observations import build_observation
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
from rl.strategy_replay_candidate import candidate_executability_from_observation


ACTION_CRITIC_FALLBACK_POLICIES: tuple[str, ...] = (
    "lowest-risk",
    "first-executable",
)


class RLStrategyPolicy:
    """Load a strategy checkpoint and use it for macro strategy decisions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        action_critic_checkpoint_path: str | Path | None = None,
        action_critic_threshold: float = 0.5,
        action_critic_fallback_policy: str = "lowest-risk",
    ) -> None:
        if not 0.0 <= action_critic_threshold <= 1.0:
            raise ValueError("action_critic_threshold must be in [0.0, 1.0]")
        if action_critic_fallback_policy not in ACTION_CRITIC_FALLBACK_POLICIES:
            names = ", ".join(ACTION_CRITIC_FALLBACK_POLICIES)
            raise ValueError(
                "Unknown action_critic_fallback_policy "
                f"{action_critic_fallback_policy!r}; expected {names}"
            )
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
        self.action_critic_checkpoint_path = (
            Path(action_critic_checkpoint_path)
            if action_critic_checkpoint_path is not None
            else None
        )
        self.action_critic_threshold = float(action_critic_threshold)
        self.action_critic_fallback_policy = action_critic_fallback_policy
        self.action_critic_loaded: Any | None = None
        self.action_critic_model: Any | None = None
        self.action_critic_normalizer: ObservationNormalizer | None = None
        self.action_critic_feature_fields: tuple[str, ...] = ()
        if self.action_critic_checkpoint_path is not None:
            from rl.strategy_action_critic import load_strategy_action_critic_checkpoint

            self.action_critic_loaded = load_strategy_action_critic_checkpoint(
                self.action_critic_checkpoint_path,
                map_location=self.device,
            )
            self.action_critic_model = self.action_critic_loaded.model.to(self.device)
            self.action_critic_model.eval()
            self.action_critic_feature_fields = (
                self.action_critic_loaded.metadata.feature_fields
            )
            self.action_critic_normalizer = (
                ObservationNormalizer.from_dict(
                    self.action_critic_loaded.metadata.normalizer,
                    expected_fields=self.action_critic_feature_fields,
                    expected_schema_version=(
                        self.action_critic_loaded.metadata.feature_schema_version
                    ),
                )
                if self.action_critic_loaded.metadata.normalizer is not None
                else None
            )
        self.executor = StrategyExecutor()
        self.last_decision_source = "checkpoint"
        self.last_decision_reason = "uninitialized"

    @torch.no_grad()
    def decide_strategy(self, bot: Any) -> StrategyAction:
        """Return the greedy strategy action for the current bot state."""
        strategy_observation = build_strategy_observation(bot)
        observation_dict = strategy_observation.to_dict()
        observation = strategy_observation.to_vector()
        if self.normalizer is not None:
            observation = self.normalizer.transform(observation)
        tensor = torch.from_numpy(observation).to(self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        logits = self.model(tensor).squeeze(0)
        ranked_action_ids = [
            int(action_id)
            for action_id in torch.argsort(logits, descending=True)
            .detach()
            .cpu()
            .tolist()
        ]
        raw_action = StrategyAction(ranked_action_ids[0])
        if self.action_critic_model is None:
            action = raw_action
            reason = "checkpoint_greedy_action"
        else:
            army_observation = build_observation(bot).to_dict()
            action, reason = self._action_critic_masked_action(
                observation_dict=observation_dict,
                army_observation=army_observation,
                raw_action=raw_action,
                ranked_action_ids=ranked_action_ids,
            )
        self.last_decision_source = "checkpoint"
        self.last_decision_reason = reason
        write_strategy_decision_metadata(
            bot,
            source=self.last_decision_source,
            reason=self.last_decision_reason,
        )
        return action

    def _action_critic_masked_action(
        self,
        *,
        observation_dict: dict[str, float],
        army_observation: dict[str, float],
        raw_action: StrategyAction,
        ranked_action_ids: list[int],
    ) -> tuple[StrategyAction, str]:
        action_critic_candidates: list[tuple[float, StrategyAction]] = []
        first_executable_action: StrategyAction | None = None
        for action_id in ranked_action_ids:
            candidate_action = StrategyAction(action_id)
            executable, _ = candidate_executability_from_observation(
                observation_dict,
                candidate_action.name,
                army_observation=army_observation,
            )
            if not executable:
                continue
            if first_executable_action is None:
                first_executable_action = candidate_action
            unsafe_probability = self._unsafe_probability(
                observation_dict,
                candidate_action,
            )
            action_critic_candidates.append((unsafe_probability, candidate_action))
            if unsafe_probability < self.action_critic_threshold:
                return (
                    candidate_action,
                    "checkpoint_action_critic_mask "
                    f"raw={raw_action.name} selected={candidate_action.name} "
                    f"unsafe={unsafe_probability:.3f} "
                    f"threshold={self.action_critic_threshold:.3f}",
                )

        if first_executable_action is None or not action_critic_candidates:
            return (
                StrategyAction.STAY_COURSE,
                "checkpoint_action_critic_no_executable_candidates "
                f"raw={raw_action.name}",
            )

        fallback_probability, fallback_action = self._fallback_action(
            first_executable_action=first_executable_action,
            action_critic_candidates=action_critic_candidates,
        )
        return (
            fallback_action,
            "checkpoint_action_critic_fallback "
            f"policy={self.action_critic_fallback_policy} raw={raw_action.name} "
            f"selected={fallback_action.name} unsafe={fallback_probability:.3f} "
            f"threshold={self.action_critic_threshold:.3f} "
            f"vetoed={len(action_critic_candidates)}",
        )

    def _unsafe_probability(
        self,
        observation_dict: dict[str, float],
        action: StrategyAction,
    ) -> float:
        if self.action_critic_model is None:
            raise ValueError("action critic model is not loaded")
        from rl.strategy_action_critic import action_critic_feature_vector_from_observation

        features = action_critic_feature_vector_from_observation(
            observation_dict,
            action.name,
            feature_fields=self.action_critic_feature_fields,
        )
        if self.action_critic_normalizer is not None:
            features = self.action_critic_normalizer.transform(features)
        tensor = torch.from_numpy(features).to(self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        probability = self.action_critic_model.predict_unsafe_probability(tensor)
        return float(probability.item())

    def _fallback_action(
        self,
        *,
        first_executable_action: StrategyAction,
        action_critic_candidates: list[tuple[float, StrategyAction]],
    ) -> tuple[float, StrategyAction]:
        if self.action_critic_fallback_policy == "lowest-risk":
            return min(action_critic_candidates, key=lambda item: item[0])
        if self.action_critic_fallback_policy == "first-executable":
            for probability, action in action_critic_candidates:
                if action == first_executable_action:
                    return probability, action
        raise ValueError(
            "Unknown action_critic_fallback_policy "
            f"{self.action_critic_fallback_policy!r}"
        )

    async def decide_and_execute(self, bot: Any) -> StrategyAction:
        """Choose and execute a strategy action.

        ProtossRuleBot normally handles execution after calling decide_strategy().
        This helper keeps the policy directly testable.
        """
        action = self.decide_strategy(bot)
        await self.executor.execute(bot, action)
        return action
