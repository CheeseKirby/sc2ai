"""Typed contracts shared by the strategy PPO scaffold."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rl.strategy_actions import StrategyAction


@dataclass(frozen=True)
class StrategyExecutionFeedback:
    """Serializable summary of what happened after a strategy action."""

    attempted: bool
    effect: str
    blocker: str | None = None
    unit_type: str | None = None
    target: str | None = None


@dataclass(frozen=True)
class StrategyPPOTransition:
    """One transition with explicit pre-action and post-action state."""

    state_before: dict[str, float]
    action: StrategyAction
    execution_result: StrategyExecutionFeedback
    state_after: dict[str, float]
    terminated: bool = False
    truncated: bool = False
    outcome: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StrategyEnvBackend(Protocol):
    """Bridge implemented by a future live SC2 or replay-backed environment."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Reset one episode and return the initial strategy observation."""

    def step(self, action: StrategyAction) -> StrategyPPOTransition:
        """Execute one action and return its complete transition."""

    def close(self) -> None:
        """Release backend resources."""
