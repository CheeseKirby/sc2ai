"""Model interfaces for high-level army policies."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bot.managers.army_policy import ArmyAction
from rl.observations import OBSERVATION_FIELDS


@dataclass(frozen=True)
class PolicyModelSpec:
    """Architecture contract saved with policy checkpoints."""

    observation_dim: int = len(OBSERVATION_FIELDS)
    action_dim: int = len(ArmyAction)
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "relu"


class ArmyPolicyNetwork(nn.Module):
    """Small MLP for high-level army action logits."""

    def __init__(self, spec: PolicyModelSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or PolicyModelSpec()
        layers: list[nn.Module] = []
        input_dim = self.spec.observation_dim
        for hidden_size in self.spec.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(_activation(self.spec.activation))
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, self.spec.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations.float())

    @torch.no_grad()
    def predict_action(self, observation: torch.Tensor) -> int:
        """Return the greedy action id for one observation vector."""
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        logits = self.forward(observation)
        return int(torch.argmax(logits, dim=-1).item())


def build_policy_model(spec: PolicyModelSpec | None = None) -> ArmyPolicyNetwork:
    """Construct the default policy model."""
    return ArmyPolicyNetwork(spec)


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")

