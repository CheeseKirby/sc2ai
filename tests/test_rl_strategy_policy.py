from __future__ import annotations

import asyncio

import pytest
import torch
from sc2.ids.unit_typeid import UnitTypeId

from bot.managers.rl_strategy_policy import RLStrategyPolicy
from rl.models import PolicyModelSpec, build_policy_model
from rl.strategy_action_critic import (
    ACTION_CRITIC_FEATURE_FIELDS,
    ActionCriticModelSpec,
    StrategyActionCriticNetwork,
    save_strategy_action_critic_checkpoint,
)
from rl.strategy_actions import StrategyAction
from rl.strategy_checkpoints import save_strategy_policy_checkpoint
from rl.strategy_observations import STRATEGY_OBSERVATION_FIELDS


class FakeUnit:
    def __init__(self, name: str = "unit") -> None:
        self.name = name
        self.train_calls: list[UnitTypeId] = []

    def train(self, unit_type: UnitTypeId) -> None:
        self.train_calls.append(unit_type)


class FakeUnits:
    def __init__(self, units: list[FakeUnit] | None = None) -> None:
        self.units = units or []

    @property
    def amount(self) -> int:
        return len(self.units)

    @property
    def exists(self) -> bool:
        return bool(self.units)

    @property
    def ready(self) -> FakeUnits:
        return self

    @property
    def idle(self) -> list[FakeUnit]:
        return self.units

    @property
    def first(self) -> FakeUnit:
        return self.units[0]

    @property
    def random(self) -> FakeUnit:
        return self.units[0]

    def __iter__(self):
        return iter(self.units)

    def of_type(self, _unit_types: object) -> FakeUnits:
        return FakeUnits([])


class FakeStructures:
    def __init__(self, counts: dict[UnitTypeId, list[FakeUnit]]) -> None:
        self.counts = counts

    def __call__(self, unit_type: UnitTypeId) -> FakeUnits:
        return FakeUnits(self.counts.get(unit_type, []))


class FakeBot:
    TARGET_WORKERS = 22
    STRATEGY_DECISION_INTERVAL = 64
    STRATEGY_TARGET_BASES = 2
    STRATEGY_GATEWAYS_PER_BASE = 4
    STRATEGY_MAX_STATIC_DEFENSE_PER_BASE = 2
    BASE_THREAT_RADIUS = 25.0
    time = 500.0
    minerals = 800
    vespene = 200
    supply_used = 60
    supply_cap = 80
    supply_left = 20

    def __init__(self) -> None:
        self.pylon = FakeUnit("pylon")
        self.nexus = FakeUnit("nexus")
        self.structure_counts: dict[UnitTypeId, list[FakeUnit]] = {
            UnitTypeId.PYLON: [self.pylon],
            UnitTypeId.NEXUS: [self.nexus],
            UnitTypeId.CYBERNETICSCORE: [FakeUnit("cyber")],
        }
        self.units = FakeUnits([])
        self.workers = FakeUnits([FakeUnit("probe") for _ in range(22)])
        self.townhalls = FakeUnits([self.nexus])
        self.structures = FakeStructures(self.structure_counts)
        self.enemy_units = FakeUnits([])
        self.enemy_structures = FakeUnits([FakeUnit("enemy")])
        self.pending: dict[UnitTypeId, int] = {}
        self.completed_upgrades = set()
        self.state = type("FakeState", (), {"upgrades": self.completed_upgrades})()
        self.affordable: set[object] = {UnitTypeId.GATEWAY}
        self.build_calls: list[tuple[UnitTypeId, FakeUnit]] = []

    def already_pending(self, unit_type: UnitTypeId) -> int:
        return self.pending.get(unit_type, 0)

    def already_pending_upgrade(self, _upgrade: object) -> bool:
        return False

    def can_afford(self, item: object) -> bool:
        return item in self.affordable

    async def build(self, unit_type: UnitTypeId, *, near: FakeUnit) -> None:
        self.build_calls.append((unit_type, near))


def _checkpoint_for_strategy_action(tmp_path, action: StrategyAction):
    model = build_policy_model(
        PolicyModelSpec(
            observation_dim=len(STRATEGY_OBSERVATION_FIELDS),
            action_dim=len(StrategyAction),
            hidden_sizes=(),
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        final = model.net[-1]
        final.bias[int(action)] = 10.0
    checkpoint = tmp_path / f"{action.name.lower()}.pt"
    save_strategy_policy_checkpoint(checkpoint, model)
    return checkpoint


def _checkpoint_for_strategy_biases(tmp_path, biases: dict[StrategyAction, float]):
    model = build_policy_model(
        PolicyModelSpec(
            observation_dim=len(STRATEGY_OBSERVATION_FIELDS),
            action_dim=len(StrategyAction),
            hidden_sizes=(),
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        final = model.net[-1]
        for action, bias in biases.items():
            final.bias[int(action)] = bias
    checkpoint = tmp_path / "strategy_biased.pt"
    save_strategy_policy_checkpoint(checkpoint, model)
    return checkpoint


def _action_critic_checkpoint(tmp_path, *, unsafe_action: StrategyAction):
    model = StrategyActionCriticNetwork(
        ActionCriticModelSpec(
            feature_dim=len(ACTION_CRITIC_FEATURE_FIELDS),
            hidden_sizes=(),
        )
    )
    with torch.no_grad():
        final = model.net[-1]
        final.weight.zero_()
        final.bias.fill_(-10.0)
        action_index = ACTION_CRITIC_FEATURE_FIELDS.index(f"action:{unsafe_action.name}")
        final.weight[0, action_index] = 20.0
    checkpoint = tmp_path / "action_critic.pt"
    save_strategy_action_critic_checkpoint(checkpoint, model)
    return checkpoint


@pytest.mark.unit
def test_rl_strategy_policy_executes_checkpoint_action(tmp_path) -> None:
    checkpoint = _checkpoint_for_strategy_action(tmp_path, StrategyAction.ADD_GATEWAYS)
    bot = FakeBot()

    action = asyncio.run(RLStrategyPolicy(checkpoint).decide_and_execute(bot))

    assert action is StrategyAction.ADD_GATEWAYS
    assert bot.build_calls == [(UnitTypeId.GATEWAY, bot.pylon)]
    assert bot.last_strategy_decision_source == "checkpoint"
    assert bot.last_strategy_decision_reason == "checkpoint_greedy_action"


@pytest.mark.unit
def test_rl_strategy_policy_can_mask_with_action_critic(tmp_path) -> None:
    checkpoint = _checkpoint_for_strategy_biases(
        tmp_path,
        {
            StrategyAction.ADD_GATEWAYS: 10.0,
            StrategyAction.STAY_COURSE: 9.0,
        },
    )
    critic_checkpoint = _action_critic_checkpoint(
        tmp_path,
        unsafe_action=StrategyAction.ADD_GATEWAYS,
    )
    bot = FakeBot()

    action = RLStrategyPolicy(
        checkpoint,
        action_critic_checkpoint_path=critic_checkpoint,
        action_critic_threshold=0.5,
    ).decide_strategy(bot)

    assert action is StrategyAction.STAY_COURSE
    assert bot.last_strategy_decision_source == "checkpoint"
    assert "checkpoint_action_critic_mask" in bot.last_strategy_decision_reason
    assert "raw=ADD_GATEWAYS" in bot.last_strategy_decision_reason
    assert "selected=STAY_COURSE" in bot.last_strategy_decision_reason
