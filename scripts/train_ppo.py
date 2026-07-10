"""CLI scaffold for strategy PPO training.

No live SC2 backend ships with this file. Use --dry-run to inspect the config,
or provide a future backend factory as ``package.module:callable``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.experiments import create_experiment_run, mark_experiment_status  # noqa: E402
from rl.ppo_training import PPOTrainConfig, train_ppo_policy  # noqa: E402
from rl.ppo_types import StrategyEnvBackend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy PPO training scaffold")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="strategy-ppo-scaffold")
    parser.add_argument(
        "--backend-factory",
        default=None,
        help="Future backend factory in package.module:callable form.",
    )
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
    if args.dry_run:
        print(
            json.dumps(
                {
                    "kind": "strategy_ppo_scaffold",
                    "backend_factory": args.backend_factory,
                    "training_started": False,
                    "config": asdict(config),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.backend_factory is None:
        raise SystemExit(
            "Live PPO training is not wired yet; pass --dry-run or provide "
            "--backend-factory package.module:callable."
        )

    factory = load_backend_factory(args.backend_factory)
    run = create_experiment_run(
        root=args.run_root,
        name=args.run_name,
        kind="strategy_ppo",
        config={
            **asdict(config),
            "backend_factory": args.backend_factory,
        },
        tags=["ppo", "strategy", "scaffold"],
    )
    try:
        backend = factory()
        checkpoint = train_ppo_policy(backend=backend, run=run, config=config)
    except Exception as exc:
        mark_experiment_status(run, "failed", summary={"error": repr(exc)})
        raise
    mark_experiment_status(
        run,
        "completed",
        summary={"checkpoint_path": str(checkpoint)},
    )
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
