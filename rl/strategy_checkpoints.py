"""Checkpoint helpers for low-frequency strategy policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rl.models import ArmyPolicyNetwork, PolicyModelSpec, build_policy_model
from rl.normalization import ObservationNormalizer
from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_observations import (
    STRATEGY_OBSERVATION_FIELDS,
    STRATEGY_OBSERVATION_SCHEMA_VERSION,
)


STRATEGY_CHECKPOINT_VERSION = 1
STRATEGY_POLICY_FAMILY = "strategy"


@dataclass(frozen=True)
class StrategyPolicyCheckpointMetadata:
    """Metadata stored with every strategy policy state dict."""

    checkpoint_version: int
    policy_family: str
    observation_schema_version: str
    observation_fields: tuple[str, ...]
    action_names: tuple[str, ...]
    model_spec: PolicyModelSpec
    normalizer: dict[str, Any] | None
    extra: dict[str, Any]


@dataclass(frozen=True)
class LoadedStrategyPolicyCheckpoint:
    """Loaded strategy model plus metadata."""

    model: ArmyPolicyNetwork
    metadata: StrategyPolicyCheckpointMetadata


def save_strategy_policy_checkpoint(
    path: str | Path,
    model: ArmyPolicyNetwork,
    *,
    normalizer: ObservationNormalizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a strategy policy checkpoint with strategy schema metadata."""
    metadata = StrategyPolicyCheckpointMetadata(
        checkpoint_version=STRATEGY_CHECKPOINT_VERSION,
        policy_family=STRATEGY_POLICY_FAMILY,
        observation_schema_version=STRATEGY_OBSERVATION_SCHEMA_VERSION,
        observation_fields=STRATEGY_OBSERVATION_FIELDS,
        action_names=tuple(
            STRATEGY_ACTION_NAMES[index] for index in sorted(STRATEGY_ACTION_NAMES)
        ),
        model_spec=model.spec,
        normalizer=normalizer.to_dict() if normalizer is not None else None,
        extra=extra or {},
    )
    payload = {
        "metadata": _metadata_to_dict(metadata),
        "state_dict": model.state_dict(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_strategy_policy_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedStrategyPolicyCheckpoint:
    """Load and validate a strategy policy checkpoint."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    metadata = _metadata_from_dict(payload["metadata"])
    _validate_metadata(metadata)
    model = build_policy_model(metadata.model_spec)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return LoadedStrategyPolicyCheckpoint(model=model, metadata=metadata)


def _metadata_to_dict(metadata: StrategyPolicyCheckpointMetadata) -> dict[str, Any]:
    data = asdict(metadata)
    data["model_spec"]["hidden_sizes"] = list(metadata.model_spec.hidden_sizes)
    data["observation_fields"] = list(metadata.observation_fields)
    data["action_names"] = list(metadata.action_names)
    return data


def _metadata_from_dict(data: dict[str, Any]) -> StrategyPolicyCheckpointMetadata:
    model_spec_data = data["model_spec"]
    return StrategyPolicyCheckpointMetadata(
        checkpoint_version=int(data["checkpoint_version"]),
        policy_family=str(data.get("policy_family", "")),
        observation_schema_version=str(data["observation_schema_version"]),
        observation_fields=tuple(data["observation_fields"]),
        action_names=tuple(data["action_names"]),
        model_spec=PolicyModelSpec(
            observation_dim=int(model_spec_data["observation_dim"]),
            action_dim=int(model_spec_data["action_dim"]),
            hidden_sizes=tuple(int(v) for v in model_spec_data["hidden_sizes"]),
            activation=str(model_spec_data["activation"]),
        ),
        normalizer=data.get("normalizer"),
        extra=dict(data.get("extra", {})),
    )


def _validate_metadata(metadata: StrategyPolicyCheckpointMetadata) -> None:
    if metadata.checkpoint_version != STRATEGY_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported strategy checkpoint version: {metadata.checkpoint_version}"
        )
    if metadata.policy_family != STRATEGY_POLICY_FAMILY:
        raise ValueError("Checkpoint is not a strategy policy checkpoint")
    if metadata.observation_schema_version != STRATEGY_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            "Strategy observation schema version mismatch: "
            f"checkpoint={metadata.observation_schema_version} "
            f"runtime={STRATEGY_OBSERVATION_SCHEMA_VERSION}. "
            "Retrain the strategy policy with the current strategy schema."
        )
    if metadata.observation_fields != STRATEGY_OBSERVATION_FIELDS:
        raise ValueError("Strategy observation field order mismatch")
    expected_actions = tuple(
        STRATEGY_ACTION_NAMES[index] for index in sorted(STRATEGY_ACTION_NAMES)
    )
    if metadata.action_names != expected_actions:
        raise ValueError("Strategy action names mismatch")
    if metadata.normalizer is not None:
        ObservationNormalizer.from_dict(
            metadata.normalizer,
            expected_fields=STRATEGY_OBSERVATION_FIELDS,
            expected_schema_version=STRATEGY_OBSERVATION_SCHEMA_VERSION,
        )
