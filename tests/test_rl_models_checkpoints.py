from __future__ import annotations

import pytest
import torch

from bot.managers.army_policy import ArmyAction
from rl.actions import ACTION_NAMES
from rl.checkpoints import CHECKPOINT_VERSION
from rl.checkpoints import load_policy_checkpoint, save_policy_checkpoint
from rl.models import PolicyModelSpec, build_policy_model
from rl.normalization import fit_observation_normalizer
from rl.observations import OBSERVATION_FIELDS, OBSERVATION_FIELDS_V1


@pytest.mark.unit
def test_policy_model_forward_shape() -> None:
    model = build_policy_model(
        PolicyModelSpec(
            observation_dim=len(OBSERVATION_FIELDS),
            action_dim=len(ArmyAction),
            hidden_sizes=(8,),
        )
    )
    observations = torch.zeros((3, len(OBSERVATION_FIELDS)), dtype=torch.float32)

    logits = model(observations)

    assert logits.shape == (3, len(ArmyAction))


@pytest.mark.unit
def test_policy_model_predict_action_returns_valid_id() -> None:
    model = build_policy_model(PolicyModelSpec(hidden_sizes=(8,)))
    observation = torch.zeros((len(OBSERVATION_FIELDS),), dtype=torch.float32)

    action = model.predict_action(observation)

    assert 0 <= action < len(ArmyAction)


@pytest.mark.unit
def test_policy_checkpoint_round_trip_preserves_logits(tmp_path) -> None:
    torch.manual_seed(7)
    model = build_policy_model(PolicyModelSpec(hidden_sizes=(8,)))
    observation = torch.randn((1, len(OBSERVATION_FIELDS)))
    before = model(observation).detach().clone()
    checkpoint = tmp_path / "policy.pt"

    save_policy_checkpoint(checkpoint, model, extra={"source": "unit"})
    loaded = load_policy_checkpoint(checkpoint)
    after = loaded.model(observation).detach()

    assert torch.allclose(before, after)
    assert loaded.metadata.extra == {"source": "unit"}
    assert loaded.metadata.model_spec.hidden_sizes == (8,)


@pytest.mark.unit
def test_policy_checkpoint_preserves_normalizer_metadata(tmp_path) -> None:
    model = build_policy_model(PolicyModelSpec(hidden_sizes=(8,)))
    observations = torch.arange(
        2 * len(OBSERVATION_FIELDS),
        dtype=torch.float32,
    ).reshape(2, len(OBSERVATION_FIELDS))
    normalizer = fit_observation_normalizer(observations.numpy())
    checkpoint = tmp_path / "policy.pt"

    save_policy_checkpoint(checkpoint, model, normalizer=normalizer)
    loaded = load_policy_checkpoint(checkpoint)

    assert loaded.metadata.normalizer is not None
    assert loaded.metadata.normalizer["fields"] == list(OBSERVATION_FIELDS)


@pytest.mark.unit
def test_policy_checkpoint_rejects_old_observation_schema(tmp_path) -> None:
    checkpoint = tmp_path / "old_schema.pt"
    payload = {
        "metadata": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "observation_schema_version": 1,
            "observation_fields": list(OBSERVATION_FIELDS_V1),
            "action_names": [
                ACTION_NAMES[index] for index in sorted(ACTION_NAMES)
            ],
            "model_spec": {
                "observation_dim": len(OBSERVATION_FIELDS_V1),
                "action_dim": len(ArmyAction),
                "hidden_sizes": [8],
                "activation": "relu",
            },
            "normalizer": None,
            "extra": {},
        },
        "state_dict": {},
    }
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="Retrain the policy"):
        load_policy_checkpoint(checkpoint)
