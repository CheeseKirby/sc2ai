"""Placeholder reward shaping for the strategy PPO scaffold."""
from __future__ import annotations

from dataclasses import dataclass

from rl.ppo_types import StrategyPPOTransition


@dataclass(frozen=True)
class StrategyRewardConfig:
    """Small, explicit reward surface intended for later tuning."""

    terminal_victory: float = 1.0
    terminal_defeat: float = -1.0
    terminal_tie: float = 0.0
    army_delta_weight: float = 0.01
    worker_delta_weight: float = 0.002
    base_delta_weight: float = 0.10
    threat_relief_weight: float = 0.05
    successful_execution_bonus: float = 0.01
    blocked_action_penalty: float = -0.01
    clip_abs: float | None = 2.0

    def __post_init__(self) -> None:
        if self.clip_abs is not None and self.clip_abs <= 0:
            raise ValueError("clip_abs must be positive or None")


class StrategyRewardCalculator:
    """Calculate a transparent placeholder reward from one transition."""

    def __init__(self, config: StrategyRewardConfig | None = None) -> None:
        self.config = config or StrategyRewardConfig()

    def calculate(self, transition: StrategyPPOTransition) -> float:
        before = transition.state_before
        after = transition.state_after
        config = self.config

        reward = 0.0
        reward += _delta(before, after, "army_count") * config.army_delta_weight
        reward += _delta(before, after, "workers") * config.worker_delta_weight
        reward += _delta(before, after, "own_bases") * config.base_delta_weight
        reward += (
            _value(before, "base_under_threat")
            - _value(after, "base_under_threat")
        ) * config.threat_relief_weight

        execution = transition.execution_result
        if execution.attempted and execution.effect != "noop":
            reward += config.successful_execution_bonus
        elif execution.blocker:
            reward += config.blocked_action_penalty

        outcome = (transition.outcome or "").lower()
        if "victory" in outcome:
            reward += config.terminal_victory
        elif "defeat" in outcome:
            reward += config.terminal_defeat
        elif "tie" in outcome:
            reward += config.terminal_tie

        if config.clip_abs is not None:
            reward = max(-config.clip_abs, min(config.clip_abs, reward))
        return float(reward)


def _delta(
    before: dict[str, float],
    after: dict[str, float],
    field: str,
) -> float:
    return _value(after, field) - _value(before, field)


def _value(observation: dict[str, float], field: str) -> float:
    return float(observation.get(field, 0.0))
