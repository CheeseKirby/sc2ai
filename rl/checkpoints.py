"""Checkpoint save/load helpers with schema metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rl.actions import ACTION_NAMES
from rl.models import ArmyPolicyNetwork, PolicyModelSpec, build_policy_model
from rl.normalization import ObservationNormalizer
from rl.observations import OBSERVATION_FIELDS, OBSERVATION_SCHEMA_VERSION


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class PolicyCheckpointMetadata:
    """Metadata stored next to every policy state dict."""

    checkpoint_version: int
    observation_schema_version: int
    observation_fields: tuple[str, ...]
    action_names: tuple[str, ...]
    model_spec: PolicyModelSpec
    normalizer: dict[str, Any] | None
    extra: dict[str, Any]


@dataclass(frozen=True)
class LoadedPolicyCheckpoint:
    """Loaded model plus metadata."""

    model: ArmyPolicyNetwork
    metadata: PolicyCheckpointMetadata


def save_policy_checkpoint(
    path: str | Path,
    model: ArmyPolicyNetwork,
    *,
    normalizer: ObservationNormalizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a policy checkpoint with schema/action metadata."""
    metadata = PolicyCheckpointMetadata(
        checkpoint_version=CHECKPOINT_VERSION,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        observation_fields=OBSERVATION_FIELDS,
        action_names=tuple(ACTION_NAMES[index] for index in sorted(ACTION_NAMES)),
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


def load_policy_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedPolicyCheckpoint:
    """Load a policy checkpoint and validate schema compatibility."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    metadata = _metadata_from_dict(payload["metadata"])
    _validate_metadata(metadata)
    model = build_policy_model(metadata.model_spec)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return LoadedPolicyCheckpoint(model=model, metadata=metadata)


def _metadata_to_dict(metadata: PolicyCheckpointMetadata) -> dict[str, Any]:
    data = asdict(metadata)
    data["model_spec"]["hidden_sizes"] = list(metadata.model_spec.hidden_sizes)
    data["observation_fields"] = list(metadata.observation_fields)
    data["action_names"] = list(metadata.action_names)
    return data


def _metadata_from_dict(data: dict[str, Any]) -> PolicyCheckpointMetadata:
    model_spec_data = data["model_spec"]
    return PolicyCheckpointMetadata(
        checkpoint_version=int(data["checkpoint_version"]),
        observation_schema_version=int(data["observation_schema_version"]),
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


def _validate_metadata(metadata: PolicyCheckpointMetadata) -> None:
    if metadata.checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {metadata.checkpoint_version}"
        )
    if metadata.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            "Observation schema version mismatch: "
            f"checkpoint={metadata.observation_schema_version} "
            f"runtime={OBSERVATION_SCHEMA_VERSION}. "
            "Retrain the policy with the current observation schema, or use "
            "the archived code/runtime that matches the checkpoint."
        )
    if metadata.observation_fields != OBSERVATION_FIELDS:
        raise ValueError("Observation field order mismatch")
    expected_actions = tuple(ACTION_NAMES[index] for index in sorted(ACTION_NAMES))
    if metadata.action_names != expected_actions:
        raise ValueError("Action names mismatch")
    if metadata.normalizer is not None:
        ObservationNormalizer.from_dict(metadata.normalizer)
