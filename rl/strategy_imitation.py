"""Behavior cloning for low-frequency strategy policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rl.experiments import ExperimentRun, write_json
from rl.experiments import read_json
from rl.imitation import (
    build_class_weights,
    build_confusion_matrix,
    evaluate_imitation_model,
    split_dataset,
)
from rl.models import PolicyModelSpec, build_policy_model
from rl.normalization import ObservationNormalizer, fit_observation_normalizer
from rl.strategy_actions import STRATEGY_ACTION_NAMES, StrategyAction
from rl.strategy_checkpoints import save_strategy_policy_checkpoint
from rl.strategy_datasets import load_strategy_trajectory_dataset
from rl.strategy_filtered_datasets import (
    SIGNAL_FILTER_PRESETS,
    StrategySignalFilterSummary,
    load_signal_filtered_strategy_trajectory_dataset,
)
from rl.strategy_observations import (
    STRATEGY_OBSERVATION_FIELDS,
    STRATEGY_OBSERVATION_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class StrategyImitationTrainConfig:
    """Configuration for one strategy behavior cloning run."""

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
    signal_filter: str = "off"
    observation_detail_gate_path: str | None = None


@dataclass(frozen=True)
class StrategyImitationTrainMetrics:
    """Summary metrics for a strategy behavior cloning run."""

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
    signal_filter: str
    signal_filter_summary: dict | None
    train_loss: float
    train_accuracy: float
    validation_loss: float | None
    validation_accuracy: float | None
    confusion_matrix: list[list[int]]
    per_action_accuracy_by_name: dict[str, float | None]
    checkpoint_path: str
    normalizer_path: str | None
    observation_detail_gate_path: str | None
    observation_detail_gate_ready: bool | None
    observation_detail_gate_inputs: list[str] | None


@dataclass(frozen=True)
class _ObservationDetailGateCheck:
    ready: bool | None
    inputs: list[str] | None


def train_strategy_imitation_policy(
    *,
    config: StrategyImitationTrainConfig,
    run: ExperimentRun,
) -> StrategyImitationTrainMetrics:
    """Train a strategy policy model and save a strategy checkpoint."""
    observation_detail_gate = _require_observation_detail_gate_ready(
        config.observation_detail_gate_path,
        training_inputs=config.inputs,
    )
    signal_filter_summary: StrategySignalFilterSummary | None = None
    if config.signal_filter == "off":
        dataset = load_strategy_trajectory_dataset(
            config.inputs,
            include_terminal=config.include_terminal,
        )
    else:
        if config.signal_filter not in SIGNAL_FILTER_PRESETS:
            names = ", ".join(("off", *sorted(SIGNAL_FILTER_PRESETS)))
            raise ValueError(
                f"Unknown signal_filter {config.signal_filter!r}; expected one of {names}"
            )
        filtered = load_signal_filtered_strategy_trajectory_dataset(
            config.inputs,
            filter_name=config.signal_filter,
            include_terminal=config.include_terminal,
        )
        dataset = filtered.dataset
        signal_filter_summary = filtered.summary
    if dataset.size == 0:
        raise ValueError("No strategy trajectory examples found for imitation training")

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
        normalizer = fit_observation_normalizer(
            split.train_observations,
            fields=STRATEGY_OBSERVATION_FIELDS,
            schema_version=STRATEGY_OBSERVATION_SCHEMA_VERSION,
        )
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
            action_dim=len(StrategyAction),
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
        action_dim=len(StrategyAction),
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
    save_strategy_policy_checkpoint(
        checkpoint_path,
        model,
        normalizer=normalizer,
        extra={
            "training_kind": "strategy_imitation",
            "examples": dataset.size,
            "inputs": list(config.inputs),
            "class_weighting": config.class_weighting,
        },
    )
    if validation_predictions is not None:
        confusion_matrix = build_confusion_matrix(
            validation_predictions.cpu().numpy(),
            split.validation_actions,
            action_dim=len(StrategyAction),
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
            action_dim=len(StrategyAction),
        )

    per_action_accuracy = strategy_per_action_accuracy_by_name(confusion_matrix)
    metrics = StrategyImitationTrainMetrics(
        examples=dataset.size,
        train_examples=int(split.train_actions.shape[0]),
        validation_examples=int(split.validation_actions.shape[0]),
        observation_dim=dataset.observation_dim,
        observation_schema_counts=dataset.observation_schema_counts,
        rows_defaulted_observation_fields=dataset.rows_defaulted_observation_fields,
        action_counts=dataset.action_counts,
        action_names=STRATEGY_ACTION_NAMES,
        action_counts_by_name={
            STRATEGY_ACTION_NAMES[action_id]: count
            for action_id, count in sorted(dataset.action_counts.items())
        },
        missing_action_names=[
            STRATEGY_ACTION_NAMES[action_id]
            for action_id in sorted(STRATEGY_ACTION_NAMES)
            if action_id not in dataset.action_counts
        ],
        class_weighting=config.class_weighting,
        class_weights_by_name={
            STRATEGY_ACTION_NAMES[action_id]: float(class_weights[action_id])
            for action_id in sorted(STRATEGY_ACTION_NAMES)
        },
        signal_filter=config.signal_filter,
        signal_filter_summary=(
            asdict(signal_filter_summary)
            if signal_filter_summary is not None
            else None
        ),
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        validation_loss=validation_loss,
        validation_accuracy=validation_accuracy,
        confusion_matrix=confusion_matrix,
        per_action_accuracy_by_name=per_action_accuracy,
        checkpoint_path=str(checkpoint_path),
        normalizer_path=str(normalizer_path) if normalizer_path is not None else None,
        observation_detail_gate_path=config.observation_detail_gate_path,
        observation_detail_gate_ready=observation_detail_gate.ready,
        observation_detail_gate_inputs=observation_detail_gate.inputs,
    )
    write_json(run.artifacts_dir / "metrics.json", asdict(metrics))
    return metrics


def strategy_per_action_accuracy_by_name(
    confusion_matrix: list[list[int]],
) -> dict[str, float | None]:
    """Return per-label strategy accuracy from a rows=true matrix."""
    accuracies: dict[str, float | None] = {}
    for action_id, name in STRATEGY_ACTION_NAMES.items():
        row = confusion_matrix[action_id]
        total = sum(row)
        accuracies[name] = (row[action_id] / total) if total else None
    return accuracies


def _require_observation_detail_gate_ready(
    gate_path: str | None,
    *,
    training_inputs: tuple[str, ...],
) -> _ObservationDetailGateCheck:
    if gate_path is None:
        return _ObservationDetailGateCheck(ready=None, inputs=None)
    gate = read_json(gate_path)
    if bool(gate.get("ready", False)):
        gate_inputs = gate.get("analysis_inputs")
        if not isinstance(gate_inputs, list):
            raise ValueError("Observation detail gate inputs missing")
        actual_inputs = [str(value) for value in gate_inputs]
        expected_inputs = [str(value) for value in training_inputs]
        if actual_inputs != expected_inputs:
            raise ValueError(
                "Observation detail gate inputs mismatch: "
                f"gate={actual_inputs} training={expected_inputs}"
            )
        return _ObservationDetailGateCheck(ready=True, inputs=actual_inputs)
    reasons = gate.get("blocking_reasons") or []
    if isinstance(reasons, list):
        reason_text = ", ".join(str(reason) for reason in reasons) or "<none>"
    else:
        reason_text = str(reasons)
    raise ValueError(f"Observation detail gate failed: {reason_text}")
