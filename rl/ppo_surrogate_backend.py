"""Deterministic scenario backend for local PPO and policy-evaluation smoke tests.

This module is intentionally a surrogate, not a StarCraft II simulator. It
provides cheap, reproducible macro dynamics so the policy pipeline can be run
without launching SC2, while preserving the production transition contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable

from rl.ppo_types import StrategyExecutionFeedback, StrategyPPOTransition
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS


@dataclass(frozen=True)
class StrategyScenario:
    name: str
    description: str
    overrides: dict[str, float]


SCENARIOS: tuple[StrategyScenario, ...] = (
    StrategyScenario(
        "economic_expansion",
        "Convert a safe mineral bank into a second base and worker growth.",
        {
            "minerals": 650.0,
            "vespene": 100.0,
            "workers": 28.0,
            "own_bases": 1.0,
            "ready_gateways": 2.0,
            "army_count": 7.0,
        },
    ),
    StrategyScenario(
        "ground_rush",
        "Stabilize an early ground attack with defense and army production.",
        {
            "minerals": 450.0,
            "workers": 24.0,
            "ready_gateways": 2.0,
            "army_count": 6.0,
            "enemy_units_known": 18.0,
            "base_under_ground_threat": 1.0,
            "base_under_threat": 1.0,
            "enemy_to_home_distance": 8.0,
        },
    ),
    StrategyScenario(
        "armored_assault",
        "Unlock Robotics tech and field Immortals against armored pressure.",
        {
            "minerals": 550.0,
            "vespene": 300.0,
            "workers": 30.0,
            "own_bases": 2.0,
            "ready_gateways": 3.0,
            "army_count": 8.0,
            "enemy_units_known": 16.0,
            "enemy_armored_units_known": 8.0,
            "base_under_ground_threat": 1.0,
            "base_under_threat": 1.0,
            "enemy_to_home_distance": 12.0,
        },
    ),
    StrategyScenario(
        "production_scaling",
        "Turn a two-base economy into gateway capacity and standing army.",
        {
            "minerals": 800.0,
            "vespene": 220.0,
            "workers": 32.0,
            "own_bases": 2.0,
            "ready_gateways": 1.0,
            "army_count": 5.0,
            "enemy_units_known": 12.0,
        },
    ),
    StrategyScenario(
        "upgrade_window",
        "Use a safe mid-game window to establish Forge upgrades.",
        {
            "minerals": 700.0,
            "vespene": 350.0,
            "workers": 34.0,
            "own_bases": 2.0,
            "ready_gateways": 3.0,
            "army_count": 14.0,
        },
    ),
)
SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


class ScenarioStrategyBackend:
    """Small deterministic macro game used for portable engineering demos."""

    def __init__(self, *, max_steps: int = 8) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.max_steps = max_steps
        self._rng = random.Random()
        self._scenario: StrategyScenario | None = None
        self._state: dict[str, float] | None = None
        self._step = 0
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if seed is not None:
            self._rng.seed(seed)
        requested = (options or {}).get("scenario")
        if requested is None:
            scenario = self._rng.choice(SCENARIOS)
        else:
            try:
                scenario = SCENARIO_BY_NAME[str(requested)]
            except KeyError as exc:
                raise ValueError(f"Unknown surrogate scenario: {requested}") from exc
        self._scenario = scenario
        self._state = _initial_observation(scenario)
        self._step = 0
        self.closed = False
        return dict(self._state)

    def step(self, action: StrategyAction) -> StrategyPPOTransition:
        if self._state is None or self._scenario is None:
            raise RuntimeError("reset() must be called before step()")
        if self._step >= self.max_steps:
            raise RuntimeError("episode is complete; call reset()")

        before = dict(self._state)
        score_before = _objective_score(self._scenario.name, before)
        feedback = self._apply_action(action)
        self._advance_clock()
        self._step += 1
        after = dict(self._state)
        score_after = _objective_score(self._scenario.name, after)
        terminated = self._step >= self.max_steps or score_after >= 1.0
        outcome = None
        if terminated:
            outcome = "victory" if score_after >= 0.8 else "defeat"
        return StrategyPPOTransition(
            state_before=before,
            action=action,
            execution_result=feedback,
            state_after=after,
            terminated=terminated,
            outcome=outcome,
            info={
                "backend": "strategy_surrogate_v1",
                "scenario": self._scenario.name,
                "scenario_description": self._scenario.description,
                "step": self._step,
                "objective_score": score_after,
                "objective_progress": score_after - score_before,
            },
        )

    def close(self) -> None:
        self.closed = True
        self._state = None
        self._scenario = None

    def set_state_for_testing(self, observation: dict[str, float]) -> None:
        """Inject a complete state for contract and edge-case tests only."""
        if set(observation) != set(STRATEGY_OBSERVATION_FIELDS):
            raise ValueError("test state must contain the complete observation schema")
        self._state = {field: float(observation[field]) for field in STRATEGY_OBSERVATION_FIELDS}

    def _apply_action(self, action: StrategyAction) -> StrategyExecutionFeedback:
        assert self._state is not None
        state = self._state
        if action is StrategyAction.STAY_COURSE:
            return StrategyExecutionFeedback(attempted=True, effect="noop")
        if action is StrategyAction.EXPAND:
            return self._spend_and_apply(
                minerals=400.0,
                effect="expand",
                mutate=lambda: _increment(state, "own_bases", 1.0),
            )
        if action is StrategyAction.ADD_GATEWAYS:
            return self._spend_and_apply(
                minerals=150.0,
                effect="add_gateway",
                mutate=lambda: _increment(state, "ready_gateways", 1.0),
            )
        if action is StrategyAction.TECH_ROBO:
            if state["has_cybernetics_core"] < 1.0:
                return _blocked("missing_cybernetics_core")
            if state["ready_robo"] >= 1.0:
                return _blocked("robo_already_ready")
            return self._spend_and_apply(
                minerals=200.0,
                vespene=100.0,
                effect="tech_robo",
                mutate=lambda: _increment(state, "ready_robo", 1.0),
            )
        if action is StrategyAction.FORGE_UPGRADES:
            if state["ready_forge"] < 1.0:
                return self._spend_and_apply(
                    minerals=150.0,
                    effect="build_forge",
                    mutate=lambda: _increment(state, "ready_forge", 1.0),
                )
            return self._spend_and_apply(
                minerals=100.0,
                vespene=100.0,
                effect="ground_weapon_upgrade",
                mutate=lambda: _increment(state, "ground_weapon_level", 1.0),
            )
        if action is StrategyAction.BUILD_STATIC_DEFENSE:
            return self._spend_and_apply(
                minerals=100.0,
                effect="build_static_defense",
                mutate=self._add_static_defense,
            )
        if action is StrategyAction.PRODUCE_ARMY:
            if state["ready_gateways"] + state["ready_robo"] < 1.0:
                return _blocked("no_ready_production")
            if state["ready_robo"] >= 1.0 and state["enemy_armored_units_known"] > 0.0:
                return self._spend_and_apply(
                    minerals=275.0,
                    vespene=100.0,
                    effect="train_immortal",
                    mutate=self._add_immortal,
                )
            return self._spend_and_apply(
                minerals=125.0,
                vespene=50.0,
                effect="train_gateway_army",
                mutate=self._add_gateway_army,
            )
        if action is StrategyAction.BOOST_WORKERS:
            if state["supply_left"] < 2.0:
                return _blocked("supply_blocked")
            return self._spend_and_apply(
                minerals=100.0,
                effect="train_workers",
                mutate=self._add_workers,
            )
        raise ValueError(f"Unsupported strategy action: {action}")

    def _spend_and_apply(
        self,
        *,
        minerals: float,
        effect: str,
        mutate: Callable[[], None],
        vespene: float = 0.0,
    ) -> StrategyExecutionFeedback:
        assert self._state is not None
        if self._state["minerals"] < minerals:
            return _blocked("insufficient_minerals")
        if self._state["vespene"] < vespene:
            return _blocked("insufficient_vespene")
        self._state["minerals"] -= minerals
        self._state["vespene"] -= vespene
        mutate()
        return StrategyExecutionFeedback(attempted=True, effect=effect)

    def _add_static_defense(self) -> None:
        assert self._state is not None
        _increment(self._state, "ready_static_defense", 1.0)
        self._relieve_threat(0.35)

    def _add_gateway_army(self) -> None:
        assert self._state is not None
        _increment(self._state, "stalkers", 2.0)
        _increment(self._state, "army_count", 2.0)
        _increment(self._state, "supply_used", 4.0)
        self._relieve_threat(0.18)

    def _add_immortal(self) -> None:
        assert self._state is not None
        _increment(self._state, "immortals", 1.0)
        _increment(self._state, "army_count", 2.0)
        _increment(self._state, "supply_used", 4.0)
        self._relieve_threat(0.25)

    def _add_workers(self) -> None:
        assert self._state is not None
        _increment(self._state, "workers", 2.0)
        _increment(self._state, "supply_used", 2.0)

    def _relieve_threat(self, amount: float) -> None:
        assert self._state is not None
        for field in (
            "base_under_threat",
            "base_under_ground_threat",
            "base_under_air_threat",
        ):
            self._state[field] = max(0.0, self._state[field] - amount)

    def _advance_clock(self) -> None:
        assert self._state is not None
        state = self._state
        state["game_time"] += 30.0
        state["minerals"] += 45.0 + state["workers"] * 0.75
        state["vespene"] += 18.0 * min(state["own_bases"], 2.0)
        state["worker_saturation_ratio"] = min(
            1.25, state["workers"] / max(1.0, state["own_bases"] * 16.0)
        )
        state["supply_left"] = max(0.0, state["supply_cap"] - state["supply_used"])


def _initial_observation(scenario: StrategyScenario) -> dict[str, float]:
    state = {field: 0.0 for field in STRATEGY_OBSERVATION_FIELDS}
    state.update(
        {
            "game_time": 240.0,
            "minerals": 500.0,
            "vespene": 150.0,
            "supply_used": 30.0,
            "supply_cap": 70.0,
            "supply_left": 40.0,
            "workers": 24.0,
            "own_bases": 1.0,
            "ready_gateways": 2.0,
            "has_cybernetics_core": 1.0,
            "stalkers": 6.0,
            "army_count": 6.0,
            "worker_saturation_ratio": 1.0,
            "gateway_idle_count": 1.0,
            "enemy_to_home_distance": 60.0,
        }
    )
    state.update(scenario.overrides)
    state["worker_saturation_ratio"] = min(
        1.25, state["workers"] / max(1.0, state["own_bases"] * 16.0)
    )
    return {field: float(state[field]) for field in STRATEGY_OBSERVATION_FIELDS}


def _objective_score(scenario: str, state: dict[str, float]) -> float:
    relief = 1.0 - min(1.0, state["base_under_threat"])
    if scenario == "economic_expansion":
        return min(1.0, 0.55 * state["own_bases"] / 2.0 + 0.45 * state["workers"] / 34.0)
    if scenario == "ground_rush":
        return min(
            1.0,
            0.35 * state["ready_static_defense"] / 2.0
            + 0.40 * state["army_count"] / 12.0
            + 0.25 * relief,
        )
    if scenario == "armored_assault":
        return min(
            1.0,
            0.30 * min(1.0, state["ready_robo"])
            + 0.45 * state["immortals"] / 2.0
            + 0.25 * relief,
        )
    if scenario == "production_scaling":
        return min(
            1.0,
            0.45 * state["ready_gateways"] / 3.0
            + 0.55 * state["army_count"] / 12.0,
        )
    if scenario == "upgrade_window":
        return min(
            1.0,
            0.40 * min(1.0, state["ready_forge"])
            + 0.60 * min(1.0, state["ground_weapon_level"]),
        )
    raise ValueError(f"Unknown surrogate scenario: {scenario}")


def _increment(state: dict[str, float], field: str, amount: float) -> None:
    state[field] += amount


def _blocked(reason: str) -> StrategyExecutionFeedback:
    return StrategyExecutionFeedback(
        attempted=False,
        effect="noop",
        blocker=reason,
    )
