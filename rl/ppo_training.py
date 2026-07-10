"""Stable-Baselines3 PPO wiring for the strategy environment scaffold."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from rl.experiments import ExperimentRun, write_json
from rl.ppo_env import SC2StrategyPPOEnv
from rl.ppo_types import StrategyEnvBackend


class PPOAlgorithm(Protocol):
    def learn(self, *, total_timesteps: int) -> PPOAlgorithm:
        """Train for the requested number of timesteps."""

    def save(self, path: str) -> None:
        """Save a Stable-Baselines3-compatible checkpoint."""


PPOAlgorithmFactory = Callable[..., PPOAlgorithm]


@dataclass(frozen=True)
class PPOTrainConfig:
    """Minimal PPO configuration; values are placeholders, not tuned defaults."""

    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    gamma: float = 0.99
    seed: int = 7
    device: str = "cpu"
    verbose: int = 1

    def validate(self) -> None:
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.n_steps < 2:
            raise ValueError("n_steps must be >= 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0.0, 1.0]")


def train_ppo_policy(
    *,
    backend: StrategyEnvBackend,
    run: ExperimentRun,
    config: PPOTrainConfig,
    algorithm_factory: PPOAlgorithmFactory | None = None,
) -> Path:
    """Train and save a strategy PPO model using an injected backend.

    The repository intentionally does not provide a live SC2 backend yet. This
    function becomes operational once a caller supplies one.
    """

    config.validate()
    run.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run.artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_json(run.artifacts_dir / "ppo_config.json", asdict(config))

    env = SC2StrategyPPOEnv(backend)
    try:
        if algorithm_factory is None:
            from stable_baselines3 import PPO

            algorithm_factory = PPO
        model = algorithm_factory(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            seed=config.seed,
            device=config.device,
            verbose=config.verbose,
        )
        model.learn(total_timesteps=config.total_timesteps)
        checkpoint_base = run.checkpoints_dir / "strategy_ppo"
        model.save(str(checkpoint_base))
        checkpoint_path = Path(f"{checkpoint_base}.zip")
        write_json(
            run.artifacts_dir / "ppo_result.json",
            {
                "checkpoint_path": str(checkpoint_path),
                "total_timesteps": config.total_timesteps,
                "status": "completed",
            },
        )
        return checkpoint_path
    finally:
        env.close()
