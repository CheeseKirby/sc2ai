"""Portable strategy-policy benchmark for PPO, rules, and LLM planners."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from rl.ppo_env import SC2StrategyPPOEnv
from rl.ppo_surrogate_backend import SCENARIOS, ScenarioStrategyBackend
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS


STRATEGY_LAB_SCHEMA_VERSION = "strategy_lab_v1"


@dataclass(frozen=True)
class StrategyLabDecision:
    action: StrategyAction
    source: str
    reasoning: str
    confidence: float = 1.0


class StrategyLabPolicy(Protocol):
    def decide(self, observation: dict[str, float]) -> StrategyLabDecision:
        """Choose one macro action from a normalized strategy observation."""


class HeuristicStrategyLabPolicy:
    """Transparent reference planner used as a non-learning benchmark."""

    def decide(self, observation: dict[str, float]) -> StrategyLabDecision:
        action, reason = _heuristic_action(observation)
        return StrategyLabDecision(
            action=action,
            source="heuristic",
            reasoning=reason,
            confidence=1.0,
        )


class StayCourseStrategyLabPolicy:
    """Behavior-preserving baseline matching the production default."""

    def decide(self, observation: dict[str, float]) -> StrategyLabDecision:
        del observation
        return StrategyLabDecision(
            action=StrategyAction.STAY_COURSE,
            source="rule",
            reasoning="default_stay_course",
            confidence=1.0,
        )


class RandomStrategyLabPolicy:
    """Seedable lower-bound baseline for sanity-checking benchmark signal."""

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def reset(self, *, seed: int) -> None:
        self._rng.seed(seed)

    def decide(self, observation: dict[str, float]) -> StrategyLabDecision:
        del observation
        action = self._rng.choice(list(StrategyAction))
        return StrategyLabDecision(
            action=action,
            source="random",
            reasoning="seeded_random_baseline",
            confidence=0.0,
        )


class PPOStrategyLabPolicy:
    """Stable-Baselines3 checkpoint adapter for the portable benchmark."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        from stable_baselines3 import PPO

        self.model = PPO.load(str(checkpoint_path), device=device)

    def decide(self, observation: dict[str, float]) -> StrategyLabDecision:
        vector = np.asarray(
            [observation[field] for field in STRATEGY_OBSERVATION_FIELDS],
            dtype=np.float32,
        )
        raw_action, _state = self.model.predict(vector, deterministic=True)
        action = StrategyAction(int(np.asarray(raw_action).reshape(-1)[0]))
        return StrategyLabDecision(
            action=action,
            source="ppo",
            reasoning=f"ppo_deterministic_action:{action.name}",
            confidence=1.0,
        )


class LLMStrategyLabPolicy:
    """Adapter for the production structured-output LLM decision client."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.previous_action = StrategyAction.STAY_COURSE

    def reset(self, *, seed: int) -> None:
        del seed
        self.previous_action = StrategyAction.STAY_COURSE

    def decide(self, observation: dict[str, float]) -> StrategyLabDecision:
        decision = self.client.request_decision(
            observation=observation,
            previous_action=self.previous_action,
        )
        self.previous_action = decision.action
        return StrategyLabDecision(
            action=decision.action,
            source=decision.source,
            reasoning=decision.reasoning,
            confidence=decision.confidence,
        )


def benchmark_strategy_policies(
    policies: dict[str, StrategyLabPolicy],
    *,
    episodes_per_scenario: int = 4,
    seed: int = 7,
    max_steps: int = 8,
    trace_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate policies on identical deterministic macro scenarios."""
    if not policies:
        raise ValueError("at least one policy is required")
    if episodes_per_scenario < 1:
        raise ValueError("episodes_per_scenario must be >= 1")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    target = Path(trace_path) if trace_path is not None else None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")

    reports: dict[str, Any] = {}
    for policy_name, policy in policies.items():
        episode_rows: list[dict[str, Any]] = []
        action_counts: Counter[str] = Counter()
        latency_values: list[float] = []
        blocked_actions = 0
        executed_actions = 0
        fallback_count = 0
        total_decisions = 0
        scenario_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for scenario_index, scenario in enumerate(SCENARIOS):
            for repetition in range(episodes_per_scenario):
                episode_seed = (
                    seed + scenario_index * 1_000 + repetition
                )
                reset = getattr(policy, "reset", None)
                if callable(reset):
                    reset(seed=episode_seed)
                env = SC2StrategyPPOEnv(ScenarioStrategyBackend(max_steps=max_steps))
                vector, _reset_info = env.reset(
                    seed=episode_seed,
                    options={"scenario": scenario.name},
                )
                episode_reward = 0.0
                outcome = "unknown"
                steps = 0
                try:
                    while True:
                        observation = _vector_to_observation(vector)
                        started = perf_counter()
                        error: str | None = None
                        try:
                            decision = policy.decide(observation)
                        except Exception as exc:
                            error = repr(exc)
                            fallback_count += 1
                            decision = StrategyLabDecision(
                                action=StrategyAction.STAY_COURSE,
                                source="benchmark_fallback",
                                reasoning="policy_error_fallback",
                                confidence=0.0,
                            )
                        latency_ms = (perf_counter() - started) * 1_000.0
                        vector, reward, terminated, truncated, info = env.step(
                            int(decision.action)
                        )
                        steps += 1
                        total_decisions += 1
                        episode_reward += reward
                        latency_values.append(latency_ms)
                        action_counts[decision.action.name] += 1
                        blocker = info["execution_result"].get("blocker")
                        if blocker:
                            blocked_actions += 1
                        elif info["execution_result"].get("effect") != "noop":
                            executed_actions += 1
                        trace = {
                            "schema_version": STRATEGY_LAB_SCHEMA_VERSION,
                            "policy": policy_name,
                            "scenario": scenario.name,
                            "episode": repetition,
                            "step": steps,
                            "action": decision.action.name,
                            "source": decision.source,
                            "reasoning": decision.reasoning,
                            "confidence": decision.confidence,
                            "reward": reward,
                            "reward_components": info["reward_components"],
                            "objective_score": info["objective_score"],
                            "blocked_by": blocker,
                            "latency_ms": latency_ms,
                            "error": error,
                        }
                        _append_trace(target, trace)
                        if terminated or truncated:
                            outcome = info.get("outcome") or (
                                "truncated" if truncated else "unknown"
                            )
                            break
                finally:
                    env.close()

                episode = {
                    "scenario": scenario.name,
                    "reward": episode_reward,
                    "steps": steps,
                    "outcome": outcome,
                }
                episode_rows.append(episode)
                scenario_rows[scenario.name].append(episode)

        victories = sum(row["outcome"] == "victory" for row in episode_rows)
        reports[policy_name] = {
            "episodes": len(episode_rows),
            "victories": victories,
            "win_rate": victories / len(episode_rows),
            "mean_reward": _mean([row["reward"] for row in episode_rows]),
            "mean_steps": _mean([row["steps"] for row in episode_rows]),
            "decisions": total_decisions,
            "execution_rate": executed_actions / max(1, total_decisions),
            "blocked_action_rate": blocked_actions / max(1, total_decisions),
            "fallback_count": fallback_count,
            "latency_ms": {
                "mean": _mean(latency_values),
                "p50": _percentile(latency_values, 0.50),
                "p95": _percentile(latency_values, 0.95),
            },
            "action_distribution": {
                action.name: action_counts[action.name] for action in StrategyAction
            },
            "scenarios": {
                name: {
                    "episodes": len(rows),
                    "win_rate": sum(row["outcome"] == "victory" for row in rows)
                    / len(rows),
                    "mean_reward": _mean([row["reward"] for row in rows]),
                }
                for name, rows in scenario_rows.items()
            },
        }

    return {
        "schema_version": STRATEGY_LAB_SCHEMA_VERSION,
        "backend": "strategy_surrogate_v1",
        "disclaimer": (
            "Portable engineering benchmark only; results do not estimate live "
            "StarCraft II win rate."
        ),
        "seed": seed,
        "episodes_per_scenario": episodes_per_scenario,
        "scenario_count": len(SCENARIOS),
        "scenarios": [
            {"name": scenario.name, "description": scenario.description}
            for scenario in SCENARIOS
        ],
        "policies": reports,
    }


def _heuristic_action(observation: dict[str, float]) -> tuple[StrategyAction, str]:
    minerals = observation["minerals"]
    vespene = observation["vespene"]
    under_threat = observation["base_under_threat"] > 0.05
    armored = observation["enemy_armored_units_known"] > 0.0

    if armored and observation["ready_robo"] < 1.0 and minerals >= 200 and vespene >= 100:
        return StrategyAction.TECH_ROBO, "counter_armored_pressure_with_robo"
    if under_threat and observation["ready_static_defense"] < 2.0 and minerals >= 100:
        return StrategyAction.BUILD_STATIC_DEFENSE, "stabilize_active_base_threat"
    if under_threat and observation["army_count"] < 12.0:
        army_minerals = 275.0 if armored and observation["ready_robo"] >= 1.0 else 125.0
        army_vespene = 100.0 if armored and observation["ready_robo"] >= 1.0 else 50.0
        if minerals >= army_minerals and vespene >= army_vespene:
            return StrategyAction.PRODUCE_ARMY, "reinforce_until_threat_is_contained"
        return StrategyAction.STAY_COURSE, "bank_resources_for_emergency_army"
    if observation["own_bases"] < 2.0 and minerals >= 400:
        return StrategyAction.EXPAND, "safe_bank_supports_second_base"
    if observation["workers"] < observation["own_bases"] * 16.0 and minerals >= 100:
        return StrategyAction.BOOST_WORKERS, "fill_current_base_saturation"
    desired_gateways = max(2.0, observation["own_bases"] + 1.0)
    if observation["ready_gateways"] < desired_gateways and minerals >= 150:
        return StrategyAction.ADD_GATEWAYS, "scale_production_with_economy"
    if (
        observation["enemy_units_known"] == 0.0
        and observation["workers"] < observation["own_bases"] * 18.0
        and minerals >= 100
    ):
        return StrategyAction.BOOST_WORKERS, "continue_safe_economic_scaling"
    if observation["army_count"] >= 10.0:
        if observation["ready_forge"] < 1.0 and minerals >= 150:
            return StrategyAction.FORGE_UPGRADES, "establish_forge_for_upgrades"
        if (
            observation["ready_forge"] >= 1.0
            and observation["ground_weapon_level"] < 1.0
            and minerals >= 100
            and vespene >= 100
        ):
            return StrategyAction.FORGE_UPGRADES, "take_safe_upgrade_window"
    if observation["army_count"] < 12.0 and minerals >= 125 and vespene >= 50:
        return StrategyAction.PRODUCE_ARMY, "build_minimum_standing_army"
    if observation["workers"] < observation["own_bases"] * 18.0 and minerals >= 100:
        return StrategyAction.BOOST_WORKERS, "continue_economic_scaling"
    return StrategyAction.STAY_COURSE, "no_high_value_macro_change"


def _vector_to_observation(vector: np.ndarray) -> dict[str, float]:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size != len(STRATEGY_OBSERVATION_FIELDS):
        raise ValueError("strategy observation vector has the wrong size")
    return {
        field: float(values[index])
        for index, field in enumerate(STRATEGY_OBSERVATION_FIELDS)
    }


def _append_trace(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return float(ordered[index])
