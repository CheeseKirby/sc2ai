"""Train a minimal behavior-cloning policy from trajectory JSONL files."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiments import create_experiment_run, mark_experiment_status  # noqa: E402
from rl.imitation import ImitationTrainConfig, train_imitation_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train imitation policy")
    parser.add_argument("inputs", nargs="+", help="Trajectory JSONL files or dirs")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--run-name", default="imitation-smoke")
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
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable observation normalization.",
    )
    parser.add_argument(
        "--include-terminal",
        action="store_true",
        help="Include done=true terminal trajectory rows.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="none",
        help="Optional class weighting for imbalanced action labels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ImitationTrainConfig(
        inputs=tuple(args.inputs),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_sizes=tuple(args.hidden_sizes),
        device=args.device,
        include_terminal=args.include_terminal,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        normalize=not args.no_normalize,
        class_weighting=args.class_weighting,
    )
    run = create_experiment_run(
        root=args.run_root,
        name=args.run_name,
        kind="imitation",
        config=asdict(config),
        tags=args.tag,
    )
    try:
        metrics = train_imitation_policy(config=config, run=run)
    except Exception:
        mark_experiment_status(run, "failed")
        raise

    mark_experiment_status(run, "complete", summary=asdict(metrics))
    print(f"Run: {run.root}")
    print(f"Examples: {metrics.examples}")
    print(f"Train accuracy: {metrics.train_accuracy:.3f}")
    if metrics.validation_accuracy is not None:
        print(f"Validation accuracy: {metrics.validation_accuracy:.3f}")
    print(f"Checkpoint: {metrics.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
