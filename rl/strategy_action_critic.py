"""Trainable binary critic for strategy action safety signals."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bot.managers.rl_strategy_policy import RLStrategyPolicy as _RLStrategyPolicy
from rl.experiments import ExperimentRun, read_json, write_json
from rl.normalization import ObservationNormalizer, fit_observation_normalizer
from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_datasets import StrategyTrajectoryPathInput
from rl.strategy_signal_dataset import (
    START_METRICS,
    StrategySignalRecord,
    build_strategy_signal_dataset,
)


ACTION_CRITIC_CHECKPOINT_VERSION = 1
ACTION_CRITIC_FAMILY = "strategy_action_critic"
ACTION_CRITIC_FEATURE_SCHEMA_VERSION = "strategy_action_critic_v1"
ACTION_CRITIC_FEATURE_SCHEMA_V2 = "strategy_action_critic_v2"
SAFE_LABEL = 0
UNSAFE_LABEL = 1
LABEL_NAMES: dict[int, str] = {SAFE_LABEL: "safe", UNSAFE_LABEL: "unsafe"}
SAFE_TRAINING_USES = frozenset(
    {"accept_positive", "drop_ambiguous", "weak_context"}
)
CONSERVATIVE_SAFE_TRAINING_USES = frozenset({"accept_positive"})
OUTCOME_CONSERVATIVE_SAFE_TRAINING_USES = frozenset({"accept_positive"})
UNSAFE_TRAINING_USES = frozenset({"drop_non_executable", "veto_negative"})
OUTCOME_CONSERVATIVE_UNSAFE_TRAINING_USES = frozenset({"veto_negative"})
ACTION_CRITIC_LABEL_POLICIES: tuple[str, ...] = (
    "trainable",
    "conservative",
    "outcome-conservative",
)
ACTION_CRITIC_V1_FEATURE_FIELDS: tuple[str, ...] = (
    *START_METRICS,
    *(f"action:{STRATEGY_ACTION_NAMES[action_id]}" for action_id in sorted(STRATEGY_ACTION_NAMES)),
)
ACTION_CRITIC_THREAT_FEATURE_FIELDS: tuple[str, ...] = (
    "threat:any",
    "threat:air",
    "threat:ground",
)
ACTION_CRITIC_ACTION_THREAT_FEATURE_FIELDS: tuple[str, ...] = tuple(
    f"action_threat:{STRATEGY_ACTION_NAMES[action_id]}:{threat_kind}"
    for action_id in sorted(STRATEGY_ACTION_NAMES)
    for threat_kind in ("any", "air", "ground")
)
ACTION_CRITIC_V2_FEATURE_FIELDS: tuple[str, ...] = (
    *ACTION_CRITIC_V1_FEATURE_FIELDS,
    *ACTION_CRITIC_THREAT_FEATURE_FIELDS,
    *ACTION_CRITIC_ACTION_THREAT_FEATURE_FIELDS,
)
ACTION_CRITIC_FEATURE_FIELDS_BY_SCHEMA: dict[str, tuple[str, ...]] = {
    ACTION_CRITIC_FEATURE_SCHEMA_VERSION: ACTION_CRITIC_V1_FEATURE_FIELDS,
    ACTION_CRITIC_FEATURE_SCHEMA_V2: ACTION_CRITIC_V2_FEATURE_FIELDS,
}
ACTION_CRITIC_FEATURE_SCHEMA_VERSIONS: tuple[str, ...] = tuple(
    ACTION_CRITIC_FEATURE_FIELDS_BY_SCHEMA
)
ACTION_CRITIC_FEATURE_FIELDS: tuple[str, ...] = ACTION_CRITIC_V1_FEATURE_FIELDS
_ = _RLStrategyPolicy


@dataclass(frozen=True)
class StrategyActionCriticRecord:
    """One training example for the action safety critic."""

    source: str
    path: str
    step: int
    game_time: float
    candidate_action: str
    candidate_blocker: str | None
    candidate_blocker_group: str | None
    recommended_training_use: str
    label_quality: str
    label: int
    label_name: str


@dataclass(frozen=True)
class StrategyActionCriticDataset:
    """Feature matrix and labels for strategy action safety training."""

    inputs: list[str]
    features: np.ndarray
    labels: np.ndarray
    example_weights: np.ndarray
    records: tuple[StrategyActionCriticRecord, ...]
    feature_fields: tuple[str, ...]
    feature_schema_version: str
    label_policy: str
    drop_non_executable_weight: float
    non_executable_blocker_weights: dict[str, float]
    label_counts: dict[int, int]
    label_counts_by_name: dict[str, int]
    training_use_counts: dict[str, int]
    training_use_weight_sums: dict[str, float]
    non_executable_blocker_counts: dict[str, int]
    non_executable_blocker_group_counts: dict[str, int]
    non_executable_blocker_weight_sums: dict[str, float]
    non_executable_blocker_group_weight_sums: dict[str, float]
    dropped_records_by_training_use: dict[str, int]

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])

    @property
    def feature_dim(self) -> int:
        if self.size:
            return int(self.features.shape[1])
        return len(self.feature_fields)


@dataclass(frozen=True)
class StrategyActionCriticTrainConfig:
    """Configuration for one action critic training run."""

    inputs: tuple[str, ...]
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 64)
    device: str = "cpu"
    validation_fraction: float = 0.2
    seed: int = 7
    normalize: bool = True
    class_weighting: str = "none"
    threshold: float = 0.5
    label_policy: str = "trainable"
    feature_schema_version: str = ACTION_CRITIC_FEATURE_SCHEMA_VERSION
    drop_non_executable_weight: float = 1.0
    non_executable_blocker_weights: dict[str, float] = field(default_factory=dict)
    observation_detail_gate_path: str | None = None


@dataclass(frozen=True)
class StrategyActionCriticTrainMetrics:
    """Summary metrics for one action critic training run."""

    examples: int
    safe_examples: int
    unsafe_examples: int
    train_examples: int
    validation_examples: int
    feature_dim: int
    feature_fields: list[str]
    feature_schema_version: str
    label_counts: dict[int, int]
    label_counts_by_name: dict[str, int]
    training_use_counts: dict[str, int]
    training_use_weight_sums: dict[str, float]
    non_executable_blocker_counts: dict[str, int]
    non_executable_blocker_group_counts: dict[str, int]
    non_executable_blocker_weight_sums: dict[str, float]
    non_executable_blocker_group_weight_sums: dict[str, float]
    dropped_records_by_training_use: dict[str, int]
    label_policy: str
    class_weighting: str
    class_weights_by_name: dict[str, float]
    drop_non_executable_weight: float
    train_weight_sum: float
    validation_weight_sum: float
    threshold: float
    train_loss: float
    train_accuracy: float
    train_precision: float | None
    train_recall: float | None
    validation_loss: float | None
    validation_accuracy: float | None
    validation_precision: float | None
    validation_recall: float | None
    confusion_matrix: list[list[int]]
    checkpoint_path: str
    normalizer_path: str | None
    observation_detail_gate_path: str | None
    observation_detail_gate_ready: bool | None
    observation_detail_gate_inputs: list[str] | None


@dataclass(frozen=True)
class ActionCriticModelSpec:
    """Architecture contract for strategy action critic checkpoints."""

    feature_dim: int
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "relu"


class StrategyActionCriticNetwork(nn.Module):
    """Small MLP that predicts unsafe probability logits for action candidates."""

    def __init__(self, spec: ActionCriticModelSpec) -> None:
        super().__init__()
        self.spec = spec
        layers: list[nn.Module] = []
        input_dim = spec.feature_dim
        for hidden_size in spec.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(_activation(spec.activation))
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)

    @torch.no_grad()
    def predict_unsafe_probability(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(features))


@dataclass(frozen=True)
class StrategyActionCriticCheckpointMetadata:
    """Metadata saved with every action critic checkpoint."""

    checkpoint_version: int
    critic_family: str
    feature_schema_version: str
    feature_fields: tuple[str, ...]
    label_names: dict[int, str]
    model_spec: ActionCriticModelSpec
    normalizer: dict[str, Any] | None
    extra: dict[str, Any]


@dataclass(frozen=True)
class LoadedStrategyActionCriticCheckpoint:
    """Loaded action critic model plus metadata."""

    model: StrategyActionCriticNetwork
    metadata: StrategyActionCriticCheckpointMetadata


@dataclass(frozen=True)
class _CriticSplit:
    train_features: np.ndarray
    train_labels: np.ndarray
    train_weights: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray
    validation_weights: np.ndarray


@dataclass(frozen=True)
class _Evaluation:
    loss: float
    accuracy: float
    precision: float | None
    recall: float | None
    predictions: np.ndarray
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class _ObservationDetailGateCheck:
    ready: bool | None
    inputs: list[str] | None


def load_strategy_action_critic_dataset(
    paths: StrategyTrajectoryPathInput,
    *,
    label_policy: str = "trainable",
    feature_schema_version: str = ACTION_CRITIC_FEATURE_SCHEMA_VERSION,
    drop_non_executable_weight: float = 1.0,
    non_executable_blocker_weights: dict[str, float] | None = None,
) -> StrategyActionCriticDataset:
    """Build a binary action-safety dataset from row-level strategy signals."""
    _validate_label_policy(label_policy)
    _validate_drop_non_executable_weight(drop_non_executable_weight)
    blocker_weights = _validate_non_executable_blocker_weights(
        non_executable_blocker_weights or {}
    )
    feature_fields = action_critic_feature_fields(feature_schema_version)
    signal_dataset = build_strategy_signal_dataset(
        paths,
        include_before_filter_candidates=False,
    )
    features: list[np.ndarray] = []
    records: list[StrategyActionCriticRecord] = []
    weights: list[float] = []
    dropped: Counter[str] = Counter()

    for record in signal_dataset.records:
        label = _label_for_training_use(
            record.recommended_training_use,
            label_policy=label_policy,
        )
        if label is None:
            dropped[record.recommended_training_use] += 1
            continue
        features.append(
            action_critic_feature_vector(record, feature_fields=feature_fields)
        )
        weights.append(
            _training_use_weight(
                record.recommended_training_use,
                candidate_blocker=record.candidate_blocker,
                drop_non_executable_weight=drop_non_executable_weight,
                non_executable_blocker_weights=blocker_weights,
            )
        )
        records.append(
            StrategyActionCriticRecord(
                source=record.source,
                path=record.path,
                step=record.step,
                game_time=record.game_time,
                candidate_action=record.candidate_action,
                candidate_blocker=record.candidate_blocker,
                candidate_blocker_group=non_executable_blocker_group(
                    record.candidate_blocker
                ),
                recommended_training_use=record.recommended_training_use,
                label_quality=record.label_quality,
                label=label,
                label_name=LABEL_NAMES[label],
            )
        )

    feature_matrix = (
        np.stack(features).astype(np.float32)
        if features
        else np.empty((0, len(feature_fields)), dtype=np.float32)
    )
    labels = np.asarray([record.label for record in records], dtype=np.float32)
    example_weights = np.asarray(weights, dtype=np.float32)
    label_counts = Counter(int(label) for label in labels.astype(np.int64))
    return StrategyActionCriticDataset(
        inputs=signal_dataset.inputs,
        features=feature_matrix,
        labels=labels,
        example_weights=example_weights,
        records=tuple(records),
        feature_fields=feature_fields,
        feature_schema_version=feature_schema_version,
        label_policy=label_policy,
        drop_non_executable_weight=float(drop_non_executable_weight),
        non_executable_blocker_weights=blocker_weights,
        label_counts=dict(sorted(label_counts.items())),
        label_counts_by_name={
            LABEL_NAMES[label]: int(label_counts[label])
            for label in sorted(LABEL_NAMES)
            if label_counts[label]
        },
        training_use_counts=_count(
            record.recommended_training_use for record in records
        ),
        training_use_weight_sums=_weight_sums_by_training_use(records, example_weights),
        non_executable_blocker_counts=_non_executable_counts(
            records,
            attr="candidate_blocker",
        ),
        non_executable_blocker_group_counts=_non_executable_counts(
            records,
            attr="candidate_blocker_group",
        ),
        non_executable_blocker_weight_sums=_non_executable_weight_sums(
            records,
            example_weights,
            attr="candidate_blocker",
        ),
        non_executable_blocker_group_weight_sums=_non_executable_weight_sums(
            records,
            example_weights,
            attr="candidate_blocker_group",
        ),
        dropped_records_by_training_use=_count(dropped.elements()),
    )


def action_critic_feature_fields(
    feature_schema_version: str = ACTION_CRITIC_FEATURE_SCHEMA_VERSION,
) -> tuple[str, ...]:
    """Return the feature field contract for an action critic schema."""
    try:
        return ACTION_CRITIC_FEATURE_FIELDS_BY_SCHEMA[feature_schema_version]
    except KeyError as exc:
        names = ", ".join(ACTION_CRITIC_FEATURE_SCHEMA_VERSIONS)
        raise ValueError(
            f"Unsupported action critic feature schema: {feature_schema_version}; "
            f"expected one of: {names}"
        ) from exc


def action_critic_feature_vector(
    record: StrategySignalRecord,
    *,
    feature_fields: tuple[str, ...] = ACTION_CRITIC_FEATURE_FIELDS,
) -> np.ndarray:
    """Return a feature vector for one signal record."""
    return action_critic_feature_vector_from_metrics(
        record.start_metrics,
        record.candidate_action,
        feature_fields=feature_fields,
    )


def action_critic_feature_vector_from_observation(
    observation: dict[str, float],
    candidate_action: str,
    *,
    feature_fields: tuple[str, ...] = ACTION_CRITIC_FEATURE_FIELDS,
) -> np.ndarray:
    """Return critic features for a candidate action at an observation row."""
    metrics = {field: float(observation.get(field, 0.0)) for field in START_METRICS}
    return action_critic_feature_vector_from_metrics(
        metrics,
        candidate_action,
        feature_fields=feature_fields,
    )


def action_critic_feature_vector_from_metrics(
    start_metrics: dict[str, float],
    candidate_action: str,
    *,
    feature_fields: tuple[str, ...] = ACTION_CRITIC_FEATURE_FIELDS,
) -> np.ndarray:
    """Return critic features from start metrics plus an action one-hot."""
    values = [
        _feature_value(
            field,
            start_metrics=start_metrics,
            candidate_action=candidate_action,
        )
        for field in feature_fields
    ]
    return np.asarray(values, dtype=np.float32)


def _feature_value(
    field: str,
    *,
    start_metrics: dict[str, float],
    candidate_action: str,
) -> float:
    if field in START_METRICS:
        return float(start_metrics.get(field, 0.0))
    if field.startswith("action:"):
        return 1.0 if field.removeprefix("action:") == candidate_action else 0.0
    if field.startswith("threat:"):
        return _threat_value(field.removeprefix("threat:"), start_metrics)
    if field.startswith("action_threat:"):
        _, action, threat_kind = field.split(":", maxsplit=2)
        if action != candidate_action:
            return 0.0
        return _threat_value(threat_kind, start_metrics)
    raise ValueError(f"Unknown action critic feature field: {field}")


def _threat_value(threat_kind: str, start_metrics: dict[str, float]) -> float:
    if threat_kind == "any":
        return float(start_metrics.get("base_under_threat", 0.0))
    if threat_kind == "air":
        return float(start_metrics.get("base_under_air_threat", 0.0))
    if threat_kind == "ground":
        return float(start_metrics.get("base_under_ground_threat", 0.0))
    raise ValueError(f"Unknown threat feature kind: {threat_kind}")


def train_strategy_action_critic(
    *,
    config: StrategyActionCriticTrainConfig,
    run: ExperimentRun,
) -> StrategyActionCriticTrainMetrics:
    """Train a binary strategy action critic and save a checkpoint."""
    observation_detail_gate = _require_observation_detail_gate_ready(
        config.observation_detail_gate_path,
        training_inputs=config.inputs,
    )
    dataset = load_strategy_action_critic_dataset(
        config.inputs,
        label_policy=config.label_policy,
        feature_schema_version=config.feature_schema_version,
        drop_non_executable_weight=config.drop_non_executable_weight,
        non_executable_blocker_weights=config.non_executable_blocker_weights,
    )
    if dataset.size == 0:
        raise ValueError("No strategy action critic examples found")
    split = _split_dataset(
        dataset,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    normalizer: ObservationNormalizer | None = None
    normalizer_path: Path | None = None
    train_features_np = split.train_features
    validation_features_np = split.validation_features
    if config.normalize:
        normalizer = fit_observation_normalizer(
            split.train_features,
            fields=dataset.feature_fields,
            schema_version=dataset.feature_schema_version,
        )
        train_features_np = normalizer.transform(split.train_features)
        if split.validation_features.size:
            validation_features_np = normalizer.transform(split.validation_features)
        normalizer_path = run.artifacts_dir / "normalizer.json"
        normalizer.save(normalizer_path)

    device = torch.device(config.device)
    model = StrategyActionCriticNetwork(
        ActionCriticModelSpec(
            feature_dim=dataset.feature_dim,
            hidden_sizes=config.hidden_sizes,
        )
    ).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features_np),
            torch.from_numpy(split.train_labels.astype(np.float32)),
            torch.from_numpy(split.train_weights.astype(np.float32)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    class_weights = _class_weights(split.train_labels, config.class_weighting)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(class_weights["unsafe"], dtype=torch.float32).to(device),
        reduction="none",
    )
    if float(split.train_weights.sum()) <= 0.0:
        raise ValueError("Training example weights must sum to a positive value")

    train_eval = _Evaluation(0.0, 0.0, None, None, np.empty((0,)), [[0, 0], [0, 0]])
    validation_eval: _Evaluation | None = None
    for _epoch in range(config.epochs):
        model.train()
        for batch_features, batch_labels, batch_weights in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)
            logits = model(batch_features)
            loss = _weighted_loss(logits, batch_labels, batch_weights, loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_eval = _evaluate_action_critic(
            model,
            torch.from_numpy(train_features_np).to(device),
            torch.from_numpy(split.train_labels.astype(np.float32)).to(device),
            torch.from_numpy(split.train_weights.astype(np.float32)).to(device),
            loss_fn,
            threshold=config.threshold,
        )
        if split.validation_labels.size:
            validation_eval = _evaluate_action_critic(
                model,
                torch.from_numpy(validation_features_np).to(device),
                torch.from_numpy(split.validation_labels.astype(np.float32)).to(device),
                torch.from_numpy(split.validation_weights.astype(np.float32)).to(device),
                loss_fn,
                threshold=config.threshold,
            )

    checkpoint_path = run.checkpoints_dir / "critic.pt"
    save_strategy_action_critic_checkpoint(
        checkpoint_path,
        model,
        normalizer=normalizer,
        extra={
            "training_kind": "strategy_action_critic",
            "examples": dataset.size,
            "inputs": list(config.inputs),
            "class_weighting": config.class_weighting,
            "threshold": config.threshold,
            "label_policy": config.label_policy,
            "feature_schema_version": dataset.feature_schema_version,
            "drop_non_executable_weight": config.drop_non_executable_weight,
            "non_executable_blocker_weights": config.non_executable_blocker_weights,
        },
        feature_schema_version=dataset.feature_schema_version,
        feature_fields=dataset.feature_fields,
    )
    confusion_matrix = (
        validation_eval.confusion_matrix
        if validation_eval is not None
        else train_eval.confusion_matrix
    )
    metrics = StrategyActionCriticTrainMetrics(
        examples=dataset.size,
        safe_examples=int(dataset.label_counts.get(SAFE_LABEL, 0)),
        unsafe_examples=int(dataset.label_counts.get(UNSAFE_LABEL, 0)),
        train_examples=int(split.train_labels.shape[0]),
        validation_examples=int(split.validation_labels.shape[0]),
        feature_dim=dataset.feature_dim,
        feature_fields=list(dataset.feature_fields),
        feature_schema_version=dataset.feature_schema_version,
        label_counts=dataset.label_counts,
        label_counts_by_name=dataset.label_counts_by_name,
        training_use_counts=dataset.training_use_counts,
        training_use_weight_sums=dataset.training_use_weight_sums,
        non_executable_blocker_counts=dataset.non_executable_blocker_counts,
        non_executable_blocker_group_counts=(
            dataset.non_executable_blocker_group_counts
        ),
        non_executable_blocker_weight_sums=(
            dataset.non_executable_blocker_weight_sums
        ),
        non_executable_blocker_group_weight_sums=(
            dataset.non_executable_blocker_group_weight_sums
        ),
        dropped_records_by_training_use=dataset.dropped_records_by_training_use,
        label_policy=dataset.label_policy,
        class_weighting=config.class_weighting,
        class_weights_by_name={
            "safe": float(class_weights["safe"]),
            "unsafe": float(class_weights["unsafe"]),
        },
        drop_non_executable_weight=float(config.drop_non_executable_weight),
        train_weight_sum=float(split.train_weights.sum()),
        validation_weight_sum=float(split.validation_weights.sum()),
        threshold=config.threshold,
        train_loss=train_eval.loss,
        train_accuracy=train_eval.accuracy,
        train_precision=train_eval.precision,
        train_recall=train_eval.recall,
        validation_loss=validation_eval.loss if validation_eval else None,
        validation_accuracy=validation_eval.accuracy if validation_eval else None,
        validation_precision=validation_eval.precision if validation_eval else None,
        validation_recall=validation_eval.recall if validation_eval else None,
        confusion_matrix=confusion_matrix,
        checkpoint_path=str(checkpoint_path),
        normalizer_path=str(normalizer_path) if normalizer_path is not None else None,
        observation_detail_gate_path=config.observation_detail_gate_path,
        observation_detail_gate_ready=observation_detail_gate.ready,
        observation_detail_gate_inputs=observation_detail_gate.inputs,
    )
    write_json(run.artifacts_dir / "metrics.json", asdict(metrics))
    return metrics


def save_strategy_action_critic_checkpoint(
    path: str | Path,
    model: StrategyActionCriticNetwork,
    *,
    normalizer: ObservationNormalizer | None = None,
    extra: dict[str, Any] | None = None,
    feature_schema_version: str = ACTION_CRITIC_FEATURE_SCHEMA_VERSION,
    feature_fields: tuple[str, ...] | None = None,
) -> None:
    """Save a strategy action critic checkpoint with feature metadata."""
    resolved_feature_fields = feature_fields or action_critic_feature_fields(
        feature_schema_version
    )
    metadata = StrategyActionCriticCheckpointMetadata(
        checkpoint_version=ACTION_CRITIC_CHECKPOINT_VERSION,
        critic_family=ACTION_CRITIC_FAMILY,
        feature_schema_version=feature_schema_version,
        feature_fields=resolved_feature_fields,
        label_names=LABEL_NAMES,
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


def load_strategy_action_critic_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedStrategyActionCriticCheckpoint:
    """Load and validate a strategy action critic checkpoint."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    metadata = _metadata_from_dict(payload["metadata"])
    _validate_metadata(metadata)
    model = StrategyActionCriticNetwork(metadata.model_spec)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return LoadedStrategyActionCriticCheckpoint(model=model, metadata=metadata)


def _split_dataset(
    dataset: StrategyActionCriticDataset,
    *,
    validation_fraction: float,
    seed: int,
) -> _CriticSplit:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")
    indices = np.arange(dataset.size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if dataset.size > 1 and validation_fraction > 0:
        validation_size = max(1, int(round(dataset.size * validation_fraction)))
        validation_size = min(validation_size, dataset.size - 1)
    else:
        validation_size = 0
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    return _CriticSplit(
        train_features=dataset.features[train_indices],
        train_labels=dataset.labels[train_indices],
        train_weights=dataset.example_weights[train_indices],
        validation_features=dataset.features[validation_indices],
        validation_labels=dataset.labels[validation_indices],
        validation_weights=dataset.example_weights[validation_indices],
    )


@torch.no_grad()
def _evaluate_action_critic(
    model: StrategyActionCriticNetwork,
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    loss_fn: nn.Module,
    *,
    threshold: float,
) -> _Evaluation:
    model.eval()
    logits = model(features)
    loss = float(_weighted_loss(logits, labels, weights, loss_fn).item())
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).to(torch.int64)
    labels_int = labels.to(torch.int64)
    accuracy = float((predictions == labels_int).float().mean().item())
    confusion = _confusion_matrix(
        predictions.detach().cpu().numpy(),
        labels_int.detach().cpu().numpy(),
    )
    return _Evaluation(
        loss=loss,
        accuracy=accuracy,
        precision=_precision(confusion),
        recall=_recall(confusion),
        predictions=predictions.detach().cpu().numpy(),
        confusion_matrix=confusion,
    )


def _weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    losses = loss_fn(logits, labels)
    weight_sum = weights.sum().clamp_min(1e-6)
    return (losses * weights).sum() / weight_sum


def _metadata_to_dict(metadata: StrategyActionCriticCheckpointMetadata) -> dict[str, Any]:
    data = asdict(metadata)
    data["feature_fields"] = list(metadata.feature_fields)
    data["label_names"] = {
        str(label): name for label, name in metadata.label_names.items()
    }
    data["model_spec"]["hidden_sizes"] = list(metadata.model_spec.hidden_sizes)
    return data


def _metadata_from_dict(data: dict[str, Any]) -> StrategyActionCriticCheckpointMetadata:
    spec = data["model_spec"]
    return StrategyActionCriticCheckpointMetadata(
        checkpoint_version=int(data["checkpoint_version"]),
        critic_family=str(data.get("critic_family", "")),
        feature_schema_version=str(data["feature_schema_version"]),
        feature_fields=tuple(data["feature_fields"]),
        label_names={int(label): name for label, name in data["label_names"].items()},
        model_spec=ActionCriticModelSpec(
            feature_dim=int(spec["feature_dim"]),
            hidden_sizes=tuple(int(value) for value in spec["hidden_sizes"]),
            activation=str(spec["activation"]),
        ),
        normalizer=data.get("normalizer"),
        extra=dict(data.get("extra", {})),
    )


def _validate_metadata(metadata: StrategyActionCriticCheckpointMetadata) -> None:
    if metadata.checkpoint_version != ACTION_CRITIC_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported action critic checkpoint version: {metadata.checkpoint_version}"
        )
    if metadata.critic_family != ACTION_CRITIC_FAMILY:
        raise ValueError("Checkpoint is not a strategy action critic checkpoint")
    expected_feature_fields = action_critic_feature_fields(
        metadata.feature_schema_version
    )
    if metadata.feature_fields != expected_feature_fields:
        raise ValueError("Strategy action critic feature fields mismatch")
    if metadata.label_names != LABEL_NAMES:
        raise ValueError("Strategy action critic label names mismatch")
    if metadata.normalizer is not None:
        ObservationNormalizer.from_dict(
            metadata.normalizer,
            expected_fields=metadata.feature_fields,
            expected_schema_version=metadata.feature_schema_version,
        )


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


def _label_for_training_use(training_use: str, *, label_policy: str) -> int | None:
    _validate_label_policy(label_policy)
    if label_policy == "outcome-conservative":
        if training_use in OUTCOME_CONSERVATIVE_SAFE_TRAINING_USES:
            return SAFE_LABEL
        if training_use in OUTCOME_CONSERVATIVE_UNSAFE_TRAINING_USES:
            return UNSAFE_LABEL
        return None
    safe_uses = (
        CONSERVATIVE_SAFE_TRAINING_USES
        if label_policy == "conservative"
        else SAFE_TRAINING_USES
    )
    if training_use in safe_uses:
        return SAFE_LABEL
    if training_use in UNSAFE_TRAINING_USES:
        return UNSAFE_LABEL
    return None


def _validate_label_policy(label_policy: str) -> None:
    if label_policy not in ACTION_CRITIC_LABEL_POLICIES:
        names = ", ".join(ACTION_CRITIC_LABEL_POLICIES)
        raise ValueError(f"label_policy must be one of: {names}")


def _validate_drop_non_executable_weight(value: float) -> None:
    if value < 0.0:
        raise ValueError("drop_non_executable_weight must be >= 0.0")


def _validate_non_executable_blocker_weights(
    values: dict[str, float],
) -> dict[str, float]:
    normalized = {str(name): float(weight) for name, weight in values.items()}
    invalid = {
        name: weight
        for name, weight in normalized.items()
        if weight < 0.0
    }
    if invalid:
        raise ValueError(
            "non_executable_blocker_weights must be >= 0.0: "
            f"{sorted(invalid)}"
        )
    return dict(sorted(normalized.items()))


def non_executable_blocker_group(blocker: str | None) -> str | None:
    """Return a coarse group for non-executable action blockers."""
    if not blocker:
        return None
    if blocker.startswith("cannot_afford_"):
        return "resource_short"
    if blocker.startswith("supply_blocked_"):
        return "supply_blocked"
    if blocker.startswith("missing_"):
        return "tech_missing"
    if blocker.startswith("no_ready_"):
        return "production_missing"
    if (
        blocker.endswith("_reached")
        or blocker.endswith("_already_started")
        or blocker.endswith("_already_pending")
        or blocker.endswith("_cap")
    ):
        return "cap_or_duplicate"
    return "other"


def _training_use_weight(
    training_use: str,
    *,
    candidate_blocker: str | None,
    drop_non_executable_weight: float,
    non_executable_blocker_weights: dict[str, float],
) -> float:
    if training_use == "drop_non_executable":
        group = non_executable_blocker_group(candidate_blocker)
        if candidate_blocker and candidate_blocker in non_executable_blocker_weights:
            return float(non_executable_blocker_weights[candidate_blocker])
        if group and group in non_executable_blocker_weights:
            return float(non_executable_blocker_weights[group])
        return float(drop_non_executable_weight)
    return 1.0


def _weight_sums_by_training_use(
    records: list[StrategyActionCriticRecord],
    weights: np.ndarray,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for record, weight in zip(records, weights):
        totals[record.recommended_training_use] = (
            totals.get(record.recommended_training_use, 0.0) + float(weight)
        )
    return {
        name: float(totals[name])
        for name in sorted(totals)
    }


def _non_executable_counts(
    records: list[StrategyActionCriticRecord],
    *,
    attr: str,
) -> dict[str, int]:
    return _count(
        str(getattr(record, attr) or "unknown")
        for record in records
        if record.recommended_training_use == "drop_non_executable"
    )


def _non_executable_weight_sums(
    records: list[StrategyActionCriticRecord],
    weights: np.ndarray,
    *,
    attr: str,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for record, weight in zip(records, weights):
        if record.recommended_training_use != "drop_non_executable":
            continue
        name = str(getattr(record, attr) or "unknown")
        totals[name] = totals.get(name, 0.0) + float(weight)
    return {
        name: float(totals[name])
        for name in sorted(totals)
    }


def _class_weights(labels: np.ndarray, class_weighting: str) -> dict[str, float]:
    if class_weighting not in {"none", "balanced"}:
        raise ValueError("class_weighting must be one of: none, balanced")
    if class_weighting == "none":
        return {"safe": 1.0, "unsafe": 1.0}
    safe = int((labels == SAFE_LABEL).sum())
    unsafe = int((labels == UNSAFE_LABEL).sum())
    if unsafe <= 0:
        return {"safe": 1.0, "unsafe": 1.0}
    return {"safe": 1.0, "unsafe": float(max(safe, 1)) / float(unsafe)}


def _confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1
    return matrix.tolist()


def _precision(confusion: list[list[int]]) -> float | None:
    true_positive = confusion[UNSAFE_LABEL][UNSAFE_LABEL]
    false_positive = confusion[SAFE_LABEL][UNSAFE_LABEL]
    denominator = true_positive + false_positive
    return (true_positive / denominator) if denominator else None


def _recall(confusion: list[list[int]]) -> float | None:
    true_positive = confusion[UNSAFE_LABEL][UNSAFE_LABEL]
    false_negative = confusion[UNSAFE_LABEL][SAFE_LABEL]
    denominator = true_positive + false_negative
    return (true_positive / denominator) if denominator else None


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _count(values: IterableABC[str]) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in Counter(values).items()))
