"""Train a binary strategy action critic from strategy signal labels."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import create_experiment_run, mark_experiment_status  # noqa: E402
from rl.strategy_action_critic import (  # noqa: E402
    ACTION_CRITIC_FEATURE_SCHEMA_VERSIONS,
    ACTION_CRITIC_LABEL_POLICIES,
    StrategyActionCriticTrainConfig,
    train_strategy_action_critic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strategy action critic")
    parser.add_argument("inputs", nargs="+", help="Strategy trajectory JSONL files or dirs")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--run-name", default="strategy_action_critic_smoke")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[64, 64],
        help="MLP hidden layer sizes.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable feature normalization.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="balanced",
        help="Optional positive-class weighting for unsafe labels.",
    )
    parser.add_argument(
        "--label-policy",
        choices=ACTION_CRITIC_LABEL_POLICIES,
        default="trainable",
        help="How strategy signal labels are converted into critic labels.",
    )
    parser.add_argument(
        "--feature-schema-version",
        choices=ACTION_CRITIC_FEATURE_SCHEMA_VERSIONS,
        default="strategy_action_critic_v1",
        help="Action critic feature schema version.",
    )
    parser.add_argument(
        "--drop-non-executable-weight",
        type=float,
        default=1.0,
        help="Per-example loss weight for drop_non_executable labels.",
    )
    parser.add_argument(
        "--non-executable-blocker-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help=(
            "Override drop_non_executable weight by exact blocker or blocker group. "
            "May be repeated, e.g. resource_short=0.25."
        ),
    )
    parser.add_argument(
        "--observation-detail-gate",
        type=Path,
        default=None,
        help=(
            "Optional strategy observation detail gate JSON. "
            "Training fails before loading data unless the gate is ready=true "
            "and matches the training inputs."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = StrategyActionCriticTrainConfig(
        inputs=tuple(args.inputs),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_sizes=tuple(args.hidden_sizes),
        device=args.device,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        normalize=not args.no_normalize,
        class_weighting=args.class_weighting,
        threshold=args.threshold,
        label_policy=args.label_policy,
        feature_schema_version=args.feature_schema_version,
        drop_non_executable_weight=args.drop_non_executable_weight,
        non_executable_blocker_weights=_parse_blocker_weights(
            args.non_executable_blocker_weight
        ),
        observation_detail_gate_path=(
            str(args.observation_detail_gate)
            if args.observation_detail_gate is not None
            else None
        ),
    )
    run = create_experiment_run(
        root=args.run_root,
        name=args.run_name,
        kind="strategy_action_critic",
        config=asdict(config),
        tags=args.tag,
    )
    try:
        metrics = train_strategy_action_critic(config=config, run=run)
    except Exception:
        mark_experiment_status(run, "failed")
        raise

    mark_experiment_status(run, "complete", summary=asdict(metrics))
    print(f"Run: {run.root}")
    print(f"Examples: {metrics.examples}")
    print(f"Label policy: {metrics.label_policy}")
    print(f"Feature schema: {metrics.feature_schema_version}")
    print(f"Unsafe examples: {metrics.unsafe_examples}")
    print(f"Train accuracy: {metrics.train_accuracy:.3f}")
    if metrics.train_precision is not None:
        print(f"Train precision: {metrics.train_precision:.3f}")
    if metrics.train_recall is not None:
        print(f"Train recall: {metrics.train_recall:.3f}")
    if metrics.validation_accuracy is not None:
        print(f"Validation accuracy: {metrics.validation_accuracy:.3f}")
    if metrics.validation_precision is not None:
        print(f"Validation precision: {metrics.validation_precision:.3f}")
    if metrics.validation_recall is not None:
        print(f"Validation recall: {metrics.validation_recall:.3f}")
    print(f"Checkpoint: {metrics.checkpoint_path}")
    return 0


def _parse_blocker_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--non-executable-blocker-weight must use NAME=WEIGHT format"
            )
        name, raw_weight = value.split("=", maxsplit=1)
        name = name.strip()
        if not name:
            raise ValueError("non-executable blocker weight name cannot be empty")
        weights[name] = float(raw_weight)
    return weights


if __name__ == "__main__":
    raise SystemExit(main())
