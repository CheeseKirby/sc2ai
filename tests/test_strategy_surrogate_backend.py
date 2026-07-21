from __future__ import annotations

import pytest

from rl.ppo_env import SC2StrategyPPOEnv
from rl.ppo_surrogate_backend import ScenarioStrategyBackend
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS


@pytest.mark.unit
def test_surrogate_backend_exposes_complete_deterministic_transition() -> None:
    backend = ScenarioStrategyBackend(max_steps=4)
    before = backend.reset(seed=11, options={"scenario": "ground_rush"})

    transition = backend.step(StrategyAction.BUILD_STATIC_DEFENSE)

    assert tuple(before) == STRATEGY_OBSERVATION_FIELDS
    assert transition.state_before == before
    assert transition.state_after["ready_static_defense"] == 1.0
    assert transition.state_after["base_under_threat"] < 1.0
    assert transition.execution_result.attempted is True
    assert transition.info["scenario"] == "ground_rush"
    assert transition.info["objective_progress"] > 0.0


@pytest.mark.unit
def test_surrogate_backend_reports_blocked_actions_without_mutating_costs() -> None:
    backend = ScenarioStrategyBackend(max_steps=4)
    before = backend.reset(seed=3, options={"scenario": "ground_rush"})
    before["minerals"] = 0.0
    backend.set_state_for_testing(before)

    transition = backend.step(StrategyAction.EXPAND)

    assert transition.execution_result.attempted is False
    assert transition.execution_result.blocker == "insufficient_minerals"
    assert transition.state_after["own_bases"] == before["own_bases"]


@pytest.mark.unit
def test_surrogate_environment_includes_explainable_reward_components() -> None:
    env = SC2StrategyPPOEnv(ScenarioStrategyBackend(max_steps=2))
    env.reset(seed=5, options={"scenario": "economic_expansion"})

    _observation, reward, _terminated, _truncated, info = env.step(
        int(StrategyAction.EXPAND)
    )

    assert reward > 0.0
    assert info["reward_components"]["objective_progress"] > 0.0
    assert sum(info["reward_components"].values()) == pytest.approx(reward)
