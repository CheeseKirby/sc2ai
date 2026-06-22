from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from sc2.ids.unit_typeid import UnitTypeId

from bot.managers.army_policy import ArmyAction
from bot.managers.rl_army_policy import RLArmyPolicy
from rl.checkpoints import save_policy_checkpoint
from rl.models import PolicyModelSpec, build_policy_model


@dataclass(frozen=True)
class FakePosition:
    x: float
    y: float
    name: str = "pos"

    def distance_to(self, other: FakePosition) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def towards(self, other: FakePosition, distance: int) -> FakePosition:
        return FakePosition(
            self.x + distance,
            self.y + distance,
            f"{self.name}->towards({other.name},{distance})",
        )


class FakeUnit:
    def __init__(self) -> None:
        self.commands: list[tuple[str, FakePosition]] = []

    def attack(self, target: FakePosition) -> None:
        self.commands.append(("attack", target))

    def move(self, target: FakePosition) -> None:
        self.commands.append(("move", target))


class FakeUnits:
    def __init__(
        self,
        amount: int,
        center: FakePosition | None = None,
        units: list[FakeUnit] | None = None,
    ) -> None:
        self.amount = amount
        self.center = center or FakePosition(0, 0)
        self._units = units or []

    @property
    def exists(self) -> bool:
        return self.amount > 0

    @property
    def idle(self) -> list[FakeUnit]:
        return self._units


class FakeUnitCollection:
    def __init__(self, units: list[FakeUnit]) -> None:
        self.units = units
        self.counts = {
            UnitTypeId.ZEALOT: len(units),
            UnitTypeId.STALKER: 0,
        }

    def of_type(self, unit_types: set[UnitTypeId] | frozenset[UnitTypeId]) -> FakeUnits:
        amount = sum(self.counts.get(unit_type, 0) for unit_type in unit_types)
        return FakeUnits(amount, FakePosition(6, 8), self.units)


class FakeStructures:
    def __call__(self, unit_type: UnitTypeId) -> FakeUnits:
        if unit_type is UnitTypeId.GATEWAY:
            return FakeUnits(4)
        if unit_type is UnitTypeId.CYBERNETICSCORE:
            return FakeUnits(1)
        return FakeUnits(0)


@dataclass
class FakeTownhall:
    position: FakePosition


class FakeTownhalls:
    amount = 1

    @property
    def exists(self) -> bool:
        return True

    @property
    def first(self) -> FakeTownhall:
        return FakeTownhall(FakePosition(0, 0, "home"))


@dataclass
class FakeGameInfo:
    map_center: FakePosition


class FakeBot:
    time = 100.0
    minerals = 300
    vespene = 100
    supply_used = 40
    supply_cap = 50
    supply_left = 10
    enemy_start_locations = [FakePosition(10, 10, "enemy-main")]

    def __init__(self) -> None:
        self.is_attacking = False
        self.units_list = [FakeUnit(), FakeUnit()]
        self.units = FakeUnitCollection(self.units_list)
        self.workers = FakeUnits(22)
        self.townhalls = FakeTownhalls()
        self.structures = FakeStructures()
        self.enemy_units = FakeUnits(0)
        self.enemy_structures = FakeUnits(1)
        self.game_info = FakeGameInfo(FakePosition(5, 5, "center"))


def _checkpoint_for_action(tmp_path, action: ArmyAction):
    model = build_policy_model(PolicyModelSpec(hidden_sizes=()))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        final = model.net[-1]
        final.bias[int(action)] = 10.0
    checkpoint = tmp_path / f"{action.name.lower()}.pt"
    save_policy_checkpoint(checkpoint, model)
    return checkpoint


@pytest.mark.unit
def test_rl_army_policy_attacks_with_checkpoint_action(tmp_path) -> None:
    checkpoint = _checkpoint_for_action(tmp_path, ArmyAction.ATTACK_MAIN)
    bot = FakeBot()

    action = RLArmyPolicy(checkpoint).manage_army(bot)

    assert action is ArmyAction.ATTACK_MAIN
    assert bot.is_attacking is True
    assert {command[0] for unit in bot.units_list for command in unit.commands} == {
        "attack"
    }
    assert {command[1].name for unit in bot.units_list for command in unit.commands} == {
        "enemy-main"
    }


@pytest.mark.unit
def test_rl_army_policy_retreats_with_checkpoint_action(tmp_path) -> None:
    checkpoint = _checkpoint_for_action(tmp_path, ArmyAction.RETREAT_HOME)
    bot = FakeBot()
    bot.is_attacking = True

    action = RLArmyPolicy(checkpoint).manage_army(bot)

    assert action is ArmyAction.RETREAT_HOME
    assert bot.is_attacking is False
    assert {command[0] for unit in bot.units_list for command in unit.commands} == {
        "move"
    }

