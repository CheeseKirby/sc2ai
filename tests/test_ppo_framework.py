from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bot.managers.ppo_strategy_policy import PPOStrategyPolicy
from rl.ppo_env import SC2StrategyPPOEnv, TransitionContractError
from rl.ppo_rewards import StrategyRewardCalculator, StrategyRewardConfig
from rl.ppo_training import PPOTrainConfig, train_ppo_policy
from rl.ppo_types import (
    StrategyEnvBackend,
    StrategyExecutionFeedback,
    StrategyPPOTransition,
)
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS


def _observation(**overrides: float) -> dict[str, float]:
    observation = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    observation.update(overrides)
    return observation


class FakeBackend(StrategyEnvBackend):
    def __init__(
        self,
        *,
        initial: dict[str, float] | None = None,
        transition_before: dict[str, float] | None = None,
    ) -> None:
        self.initial = initial or _observation(army_count=3.0)
        self.transition_before = transition_before or self.initial
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        del seed, options
        return dict(self.initial)

    def step(self, action: StrategyAction) -> StrategyPPOTransition:
        return StrategyPPOTransition(
            state_before=dict(self.transition_before),
            action=action,
            execution_result=StrategyExecutionFeedback(
                attempted=True,
                effect="delegate_train_army",
            ),
            state_after=_observation(army_count=5.0),
            terminated=False,
            truncated=False,
            info={"backend": "fake"},
        )

    def close(self) -> None:
        self.closed = True


@pytest.mark.unit
def test_strategy_ppo_env_preserves_transition_contract_and_computes_reward() -> None:
    backend = FakeBackend()
    calculator = StrategyRewardCalculator(
        StrategyRewardConfig(
            terminal_victory=0.0,
            terminal_defeat=0.0,
            army_delta_weight=0.5,
            worker_delta_weight=0.0,
            base_delta_weight=0.0,
            threat_relief_weight=0.0,
            successful_execution_bonus=0.0,
            blocked_action_penalty=0.0,
            clip_abs=None,
        )
    )
    env = SC2StrategyPPOEnv(backend, reward_calculator=calculator)

    observation, reset_info = env.reset(seed=7)
    next_observation, reward, terminated, truncated, info = env.step(
        int(StrategyAction.PRODUCE_ARMY)
    )

    assert observation.shape == (len(STRATEGY_OBSERVATION_FIELDS),)
    assert observation.dtype == np.float32
    assert reset_info["schema"] == "strategy_v2"
    assert next_observation.shape == observation.shape
    assert reward == pytest.approx(1.0)
    assert terminated is False
    assert truncated is False
    assert info["action_name"] == "PRODUCE_ARMY"
    assert info["execution_result"]["effect"] == "delegate_train_army"

    env.close()
    assert backend.closed is True


@pytest.mark.unit
def test_strategy_ppo_env_rejects_backend_state_before_mismatch() -> None:
    backend = FakeBackend(
        initial=_observation(workers=20.0),
        transition_before=_observation(workers=21.0),
    )
    env = SC2StrategyPPOEnv(backend)
    env.reset()

    with pytest.raises(TransitionContractError, match="state_before"):
        env.step(int(StrategyAction.STAY_COURSE))


class FakePPOModel:
    def __init__(self, action: StrategyAction) -> None:
        self.action = action
        self.calls: list[tuple[np.ndarray, bool]] = []

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        self.calls.append((observation, deterministic))
        return np.asarray([int(self.action)], dtype=np.int64), None


@pytest.mark.unit
def test_ppo_strategy_policy_uses_deterministic_checkpoint_action() -> None:
    model = FakePPOModel(StrategyAction.TECH_ROBO)
    bot = SimpleNamespace()
    policy = PPOStrategyPolicy(
        model=model,
        observation_builder=lambda _: np.zeros(
            (len(STRATEGY_OBSERVATION_FIELDS),),
            dtype=np.float32,
        ),
    )

    action = policy.decide_strategy(bot)

    assert action is StrategyAction.TECH_ROBO
    assert model.calls[0][1] is True
    assert bot.last_strategy_decision_source == "ppo"
    assert "TECH_ROBO" in bot.last_strategy_decision_reason


class FakeAlgorithm:
    def __init__(self) -> None:
        self.learn_timesteps: int | None = None

    def learn(self, *, total_timesteps: int) -> FakeAlgorithm:
        self.learn_timesteps = total_timesteps
        return self

    def save(self, path: str) -> None:
        target = f"{path}.zip"
        with open(target, "wb") as output:
            output.write(b"fake-ppo-checkpoint")


@pytest.mark.unit
def test_train_ppo_policy_wires_algorithm_without_requiring_live_sc2(tmp_path) -> None:
    algorithm = FakeAlgorithm()
    factory_calls: list[dict[str, Any]] = []

    def algorithm_factory(policy: str, env: Any, **kwargs: Any) -> FakeAlgorithm:
        factory_calls.append({"policy": policy, "env": env, **kwargs})
        return algorithm

    run = SimpleNamespace(
        checkpoints_dir=tmp_path / "checkpoints",
        artifacts_dir=tmp_path / "artifacts",
    )
    run.checkpoints_dir.mkdir()
    run.artifacts_dir.mkdir()

    checkpoint = train_ppo_policy(
        backend=FakeBackend(),
        run=run,
        config=PPOTrainConfig(
            total_timesteps=12,
            n_steps=4,
            batch_size=2,
            device="cpu",
        ),
        algorithm_factory=algorithm_factory,
    )

    assert algorithm.learn_timesteps == 12
    assert factory_calls[0]["policy"] == "MlpPolicy"
    assert factory_calls[0]["n_steps"] == 4
    assert checkpoint.name == "strategy_ppo.zip"
    assert checkpoint.exists()
    assert (run.artifacts_dir / "ppo_config.json").exists()
