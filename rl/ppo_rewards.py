"""Explainable reward shaping for strategy PPO transitions."""
from __future__ import annotations

from dataclasses import dataclass

from rl.ppo_types import StrategyPPOTransition


@dataclass(frozen=True)
class StrategyRewardConfig:
    """Explicit reward surface shared by training and offline evaluation."""

    terminal_victory: float = 1.0
    terminal_defeat: float = -1.0
    terminal_tie: float = 0.0
    army_delta_weight: float = 0.01
    worker_delta_weight: float = 0.002
    base_delta_weight: float = 0.10
    threat_relief_weight: float = 0.05
    objective_progress_weight: float = 1.0
    successful_execution_bonus: float = 0.01
    blocked_action_penalty: float = -0.01
    clip_abs: float | None = 2.0

    def __post_init__(self) -> None:
        if self.clip_abs is not None and self.clip_abs <= 0:
            raise ValueError("clip_abs must be positive or None")


class StrategyRewardCalculator:
    """Calculate rewards and expose their attribution for traces and audits."""

    def __init__(self, config: StrategyRewardConfig | None = None) -> None:
        self.config = config or StrategyRewardConfig()

    def calculate(self, transition: StrategyPPOTransition) -> float:
        return float(sum(self.calculate_components(transition).values()))

    def calculate_components(
        self,
        transition: StrategyPPOTransition,
    ) -> dict[str, float]:
        before = transition.state_before
        after = transition.state_after
        config = self.config
        components = {
            "army_delta": _delta(before, after, "army_count")
            * config.army_delta_weight,
            "worker_delta": _delta(before, after, "workers")
            * config.worker_delta_weight,
            "base_delta": _delta(before, after, "own_bases")
            * config.base_delta_weight,
            "threat_relief": (
                _value(before, "base_under_threat")
                - _value(after, "base_under_threat")
            )
            * config.threat_relief_weight,
            "objective_progress": float(
                transition.info.get("objective_progress", 0.0)
            )
            * config.objective_progress_weight,
            "execution": 0.0,
            "terminal": 0.0,
        }

        execution = transition.execution_result
        if execution.attempted and execution.effect != "noop":
            components["execution"] = config.successful_execution_bonus
        elif execution.blocker:
            components["execution"] = config.blocked_action_penalty

        outcome = (transition.outcome or "").lower()
        if "victory" in outcome:
            components["terminal"] = config.terminal_victory
        elif "defeat" in outcome:
            components["terminal"] = config.terminal_defeat
        elif "tie" in outcome:
            components["terminal"] = config.terminal_tie

        unclipped = sum(components.values())
        if config.clip_abs is not None:
            clipped = max(-config.clip_abs, min(config.clip_abs, unclipped))
            if clipped != unclipped:
                components["clip_adjustment"] = clipped - unclipped
        return {name: float(value) for name, value in components.items()}


def _delta(
    before: dict[str, float],
    after: dict[str, float],
    field: str,
) -> float:
    return _value(after, field) - _value(before, field)


def _value(observation: dict[str, float], field: str) -> float:
    return float(observation.get(field, 0.0))
