"""Minimal behavior cloning training loop."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bot.managers.army_policy import ArmyAction
from rl.actions import ACTION_NAMES
from rl.checkpoints import save_policy_checkpoint
from rl.datasets import TrajectoryDataset, load_trajectory_dataset
from rl.experiments import ExperimentRun, write_json
from rl.models import PolicyModelSpec, build_policy_model
from rl.normalization import ObservationNormalizer, fit_observation_normalizer


@dataclass(frozen=True)
class ImitationTrainConfig:
    """Configuration for one small behavior cloning run."""

    inputs: tuple[str, ...]
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 64)
    device: str = "cpu"
    include_terminal: bool = False
    validation_fraction: float = 0.2
    seed: int = 7
    normalize: bool = True
    class_weighting: str = "none"


@dataclass(frozen=True)
class ImitationTrainMetrics:
    """Summary metrics for a behavior cloning run."""

    examples: int
    train_examples: int
    validation_examples: int
    observation_dim: int
    observation_schema_counts: dict[str, int]
    rows_defaulted_observation_fields: int
    action_counts: dict[int, int]
    action_names: dict[int, str]
    action_counts_by_name: dict[str, int]
    missing_action_names: list[str]
    class_weighting: str
    class_weights_by_name: dict[str, float]
    train_loss: float
    train_accuracy: float
    validation_loss: float | None
    validation_accuracy: float | None
    confusion_matrix: list[list[int]]
    per_action_accuracy_by_name: dict[str, float | None]
    checkpoint_path: str
    normalizer_path: str | None


@dataclass(frozen=True)
class DatasetSplit:
    train_observations: np.ndarray
    train_actions: np.ndarray
    validation_observations: np.ndarray
    validation_actions: np.ndarray


def train_imitation_policy(
    *,
    config: ImitationTrainConfig,
    run: ExperimentRun,
) -> ImitationTrainMetrics:
    """Train a small policy model and save a checkpoint."""
    dataset = load_trajectory_dataset(
        config.inputs,
        include_terminal=config.include_terminal,
    )
    if dataset.size == 0:
        raise ValueError("No trajectory examples found for imitation training")

    split = split_dataset(
        dataset,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    normalizer: ObservationNormalizer | None = None
    normalizer_path: Path | None = None
    train_observations_np = split.train_observations
    validation_observations_np = split.validation_observations
    if config.normalize:
        normalizer = fit_observation_normalizer(split.train_observations)
        train_observations_np = normalizer.transform(split.train_observations)
        if split.validation_observations.size:
            validation_observations_np = normalizer.transform(
                split.validation_observations
            )
        normalizer_path = run.artifacts_dir / "normalizer.json"
        normalizer.save(normalizer_path)

    device = torch.device(config.device)
    model = build_policy_model(
        PolicyModelSpec(
            observation_dim=dataset.observation_dim,
            hidden_sizes=config.hidden_sizes,
        )
    ).to(device)

    train_observations = torch.from_numpy(train_observations_np)
    train_actions = torch.from_numpy(split.train_actions)
    loader = DataLoader(
        TensorDataset(train_observations, train_actions),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    class_weights = build_class_weights(
        split.train_actions,
        action_dim=len(ArmyAction),
        strategy=config.class_weighting,
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=(
            torch.from_numpy(class_weights).to(device)
            if config.class_weighting != "none"
            else None
        )
    )
    train_loss = 0.0
    train_accuracy = 0.0
    validation_loss: float | None = None
    validation_accuracy: float | None = None
    validation_predictions: torch.Tensor | None = None

    for _epoch in range(config.epochs):
        model.train()
        for batch_observations, batch_actions in loader:
            batch_observations = batch_observations.to(device)
            batch_actions = batch_actions.to(device)
            logits = model(batch_observations)
            loss = loss_fn(logits, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy, _ = evaluate_imitation_model(
            model,
            torch.from_numpy(train_observations_np).to(device),
            torch.from_numpy(split.train_actions).to(device),
            loss_fn,
        )
        if split.validation_actions.size:
            (
                validation_loss,
                validation_accuracy,
                validation_predictions,
            ) = evaluate_imitation_model(
                model,
                torch.from_numpy(validation_observations_np).to(device),
                torch.from_numpy(split.validation_actions).to(device),
                loss_fn,
            )

    checkpoint_path = run.checkpoints_dir / "policy.pt"
    save_policy_checkpoint(
        checkpoint_path,
        model,
        normalizer=normalizer,
        extra={
            "training_kind": "imitation",
            "examples": dataset.size,
            "inputs": list(config.inputs),
            "class_weighting": config.class_weighting,
        },
    )
    if validation_predictions is not None:
        confusion_matrix = build_confusion_matrix(
            validation_predictions.cpu().numpy(),
            split.validation_actions,
            action_dim=len(ArmyAction),
        )
    else:
        _, _, train_predictions = evaluate_imitation_model(
            model,
            torch.from_numpy(train_observations_np).to(device),
            torch.from_numpy(split.train_actions).to(device),
            loss_fn,
        )
        confusion_matrix = build_confusion_matrix(
            train_predictions.cpu().numpy(),
            split.train_actions,
            action_dim=len(ArmyAction),
        )
    per_action_accuracy = per_action_accuracy_by_name(confusion_matrix)

    metrics = ImitationTrainMetrics(
        examples=dataset.size,
        train_examples=int(split.train_actions.shape[0]),
        validation_examples=int(split.validation_actions.shape[0]),
        observation_dim=dataset.observation_dim,
        observation_schema_counts=dataset.observation_schema_counts,
        rows_defaulted_observation_fields=dataset.rows_defaulted_observation_fields,
        action_counts=dataset.action_counts,
        action_names=ACTION_NAMES,
        action_counts_by_name={
            ACTION_NAMES[action_id]: count
            for action_id, count in sorted(dataset.action_counts.items())
        },
        missing_action_names=[
            ACTION_NAMES[action_id]
            for action_id in sorted(ACTION_NAMES)
            if action_id not in dataset.action_counts
        ],
        class_weighting=config.class_weighting,
        class_weights_by_name={
            ACTION_NAMES[action_id]: float(class_weights[action_id])
            for action_id in sorted(ACTION_NAMES)
        },
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        validation_loss=validation_loss,
        validation_accuracy=validation_accuracy,
        confusion_matrix=confusion_matrix,
        per_action_accuracy_by_name=per_action_accuracy,
        checkpoint_path=str(checkpoint_path),
        normalizer_path=str(normalizer_path) if normalizer_path is not None else None,
    )
    write_json(run.artifacts_dir / "metrics.json", asdict(metrics))
    return metrics


@torch.no_grad()
def evaluate_imitation_model(
    model: nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    loss_fn: nn.Module,
) -> tuple[float, float, torch.Tensor]:
    """Return loss and exact action accuracy on the provided tensors."""
    model.eval()
    logits = model(observations)
    loss = float(loss_fn(logits, actions).item())
    predictions = torch.argmax(logits, dim=-1)
    accuracy = float((predictions == actions).float().mean().item())
    return loss, accuracy, predictions


def split_dataset(
    dataset: TrajectoryDataset,
    *,
    validation_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Create a deterministic train/validation split."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")
    size = dataset.size
    indices = np.arange(size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if size > 1 and validation_fraction > 0:
        validation_size = max(1, int(round(size * validation_fraction)))
        validation_size = min(validation_size, size - 1)
    else:
        validation_size = 0
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    return DatasetSplit(
        train_observations=dataset.observations[train_indices],
        train_actions=dataset.actions[train_indices],
        validation_observations=dataset.observations[validation_indices],
        validation_actions=dataset.actions[validation_indices],
    )


def build_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    action_dim: int,
) -> list[list[int]]:
    """Return rows=true labels, columns=predicted labels."""
    matrix = np.zeros((action_dim, action_dim), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1
    return matrix.tolist()


def build_class_weights(
    labels: np.ndarray,
    *,
    action_dim: int,
    strategy: str,
) -> np.ndarray:
    """Return per-action class weights for CrossEntropyLoss."""
    if strategy not in {"none", "balanced"}:
        raise ValueError("class_weighting must be one of: none, balanced")
    if strategy == "none":
        return np.ones((action_dim,), dtype=np.float32)

    weights = np.zeros((action_dim,), dtype=np.float32)
    counts = np.bincount(labels.astype(np.int64), minlength=action_dim)
    present = counts > 0
    present_count = int(present.sum())
    if present_count == 0:
        return weights
    total = float(counts[present].sum())
    weights[present] = total / (present_count * counts[present].astype(np.float32))
    return weights


def per_action_accuracy_by_name(
    confusion_matrix: list[list[int]],
) -> dict[str, float | None]:
    """Return per-label accuracy from a rows=true, columns=predicted matrix."""
    accuracies: dict[str, float | None] = {}
    for action_id, name in ACTION_NAMES.items():
        row = confusion_matrix[action_id]
        total = sum(row)
        accuracies[name] = (row[action_id] / total) if total else None
    return accuracies
