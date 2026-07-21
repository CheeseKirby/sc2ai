"""Train the strategy PPO policy against a surrogate or injected backend.

The built-in surrogate is for pipeline validation and portfolio demos. It does
not model live StarCraft II and must not be used to claim gameplay strength.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.experiments import create_experiment_run, mark_experiment_status  # noqa: E402
from rl.ppo_surrogate_backend import ScenarioStrategyBackend  # noqa: E402
from rl.ppo_training import PPOTrainConfig, train_ppo_policy  # noqa: E402
from rl.ppo_types import StrategyEnvBackend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy PPO training pipeline")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="strategy-ppo")
    parser.add_argument(
        "--backend",
        choices=("external", "surrogate"),
        default="external",
        help="Use the portable surrogate or an injected live/replay backend.",
    )
    parser.add_argument(
        "--backend-factory",
        default=None,
        help="External backend factory in package.module:callable form.",
    )
    parser.add_argument("--surrogate-max-steps", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without creating a run or training.",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_backend_factory(spec: str) -> Callable[[], StrategyEnvBackend]:
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("backend factory must use package.module:callable format")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"Backend factory is not callable: {spec}")
    return factory


def resolve_backend_factory(args: argparse.Namespace) -> Callable[[], StrategyEnvBackend]:
    if args.backend == "surrogate":
        if args.backend_factory is not None:
            raise ValueError("--backend surrogate cannot be combined with --backend-factory")
        if args.surrogate_max_steps < 1:
            raise ValueError("--surrogate-max-steps must be >= 1")
        return lambda: ScenarioStrategyBackend(max_steps=args.surrogate_max_steps)
    if args.backend_factory is None:
        raise SystemExit(
            "External PPO training requires --backend-factory "
            "package.module:callable; use --backend surrogate for a local smoke run."
        )
    return load_backend_factory(args.backend_factory)


def main() -> int:
    args = parse_args()
    config = PPOTrainConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
    )
    config.validate()
    backend_config = {
        "backend": args.backend,
        "backend_factory": args.backend_factory,
        "surrogate_max_steps": args.surrogate_max_steps,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "kind": "strategy_ppo",
                    **backend_config,
                    "training_started": False,
                    "config": asdict(config),
                    "disclaimer": (
                        "surrogate training validates the pipeline only and does "
                        "not estimate live SC2 performance"
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    factory = resolve_backend_factory(args)
    run = create_experiment_run(
        root=args.run_root,
        name=args.run_name,
        kind="strategy_ppo",
        config={**asdict(config), **backend_config},
        tags=["ppo", "strategy", args.backend],
    )
    try:
        checkpoint = train_ppo_policy(
            backend=factory(),
            run=run,
            config=config,
        )
    except Exception as exc:
        mark_experiment_status(run, "failed", summary={"error": repr(exc)})
        raise
    mark_experiment_status(
        run,
        "completed",
        summary={
            "checkpoint_path": str(checkpoint),
            "backend": args.backend,
            "promotion_ready": False,
        },
    )
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
