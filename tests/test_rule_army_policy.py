from __future__ import annotations

from dataclasses import dataclass

import pytest

from bot.managers.army_policy import ArmyAction
from bot.managers.army_memory import ArmyMemory
from bot.managers.coverage_army_policy import CoverageArmyPolicy
from bot.managers.rule_army_policy import RuleArmyPolicy


@dataclass(frozen=True)
class FakePosition:
    name: str

    def towards(self, other: FakePosition, distance: int) -> FakePosition:
        return FakePosition(f"{self.name}->towards({other.name},{distance})")


class FakeUnit:
    def __init__(self) -> None:
        self.commands: list[tuple[str, FakePosition]] = []

    def attack(self, target: FakePosition) -> None:
        self.commands.append(("attack", target))

    def move(self, target: FakePosition) -> None:
        self.commands.append(("move", target))


class FakeArmy:
    def __init__(self, units: list[FakeUnit], idle_units: list[FakeUnit] | None = None) -> None:
        self._units = units
        self._idle_units = units if idle_units is None else idle_units
        self.amount = len(units)

    def of_type(self, _unit_types: object) -> FakeArmy:
        return self

    @property
    def idle(self) -> list[FakeUnit]:
        return self._idle_units


@dataclass
class FakeTownhall:
    position: FakePosition


class FakeTownhalls:
    def __init__(self, townhall: FakeTownhall | None) -> None:
        self._townhall = townhall

    @property
    def exists(self) -> bool:
        return self._townhall is not None

    @property
    def first(self) -> FakeTownhall:
        if self._townhall is None:
            raise AssertionError("No fake townhall exists")
        return self._townhall

    def __iter__(self):
        if self._townhall is None:
            return iter(())
        return iter((self._townhall,))


class FakeEnemyUnits:
    def __init__(self, target: FakePosition | None = None) -> None:
        self.target = target

    @property
    def exists(self) -> bool:
        return self.target is not None

    def closer_than(self, _radius: float, _townhall: FakeTownhall) -> FakeEnemyUnits:
        return self

    @property
    def first(self) -> FakePosition:
        if self.target is None:
            raise AssertionError("No fake enemy target exists")
        return self.target


@dataclass
class FakeGameInfo:
    map_center: FakePosition


class FakeBot:
    ARMY_ATTACK_THRESHOLD = 15
    ARMY_RETREAT_THRESHOLD = 5
    RETREAT_PEAK_LOSS_RATIO = 0.25
    RETREAT_MIN_PEAK_ARMY = 8
    RETREAT_MIN_LOST_FROM_PEAK = 3

    def __init__(
        self,
        *,
        army_count: int,
        is_attacking: bool = False,
        has_townhall: bool = True,
        enemy_threat: bool = False,
        idle_units: list[FakeUnit] | None = None,
    ) -> None:
        self.is_attacking = is_attacking
        self.units_list = [FakeUnit() for _ in range(army_count)]
        self.units = FakeArmy(self.units_list, idle_units=idle_units)
        self.enemy_start_locations = [FakePosition("enemy-main")]
        townhall = FakeTownhall(FakePosition("home")) if has_townhall else None
        self.townhalls = FakeTownhalls(townhall)
        self.game_info = FakeGameInfo(FakePosition("center"))
        self.enemy_units = FakeEnemyUnits(
            FakePosition("enemy-near-home") if enemy_threat else None
        )
        self.army_memory = ArmyMemory()
        self.army_memory.update(army_count, is_attacking=is_attacking)


@pytest.mark.unit
def test_army_memory_tracks_peak_loss_and_recent_delta() -> None:
    memory = ArmyMemory()

    memory.update(10, is_attacking=False)
    memory.start_attack()
    memory.update(14, is_attacking=True)
    memory.update(8, is_attacking=True)

    assert memory.previous_army_count == 14
    assert memory.army_count == 8
    assert memory.army_count_delta == -6
    assert memory.attack_army_peak == 14
    assert memory.army_lost_from_peak == 6
    assert memory.army_lost_from_peak_ratio == pytest.approx(6 / 14)


@pytest.mark.unit
def test_rule_policy_rallies_when_below_attack_threshold() -> None:
    bot = FakeBot(army_count=10)

    action = RuleArmyPolicy().manage_army(bot)

    assert action is ArmyAction.RALLY
    assert bot.is_attacking is False
    assert {command[0] for unit in bot.units_list for command in unit.commands} == {
        "move"
    }


@pytest.mark.unit
def test_rule_policy_attacks_when_threshold_is_reached() -> None:
    bot = FakeBot(army_count=15)

    action = RuleArmyPolicy().manage_army(bot)

    assert action is ArmyAction.ATTACK_MAIN
    assert bot.is_attacking is True
    assert {command[0] for unit in bot.units_list for command in unit.commands} == {
        "attack"
    }
    assert {command[1].name for unit in bot.units_list for command in unit.commands} == {
        "enemy-main"
    }


@pytest.mark.unit
def test_rule_policy_retreats_when_attacking_army_falls_to_threshold() -> None:
    bot = FakeBot(army_count=5, is_attacking=True)

    action = RuleArmyPolicy().manage_army(bot)

    assert action is ArmyAction.RETREAT_HOME
    assert bot.is_attacking is False
    assert {command[0] for unit in bot.units_list for command in unit.commands} == {
        "move"
    }
    assert {
        command[1].name for unit in bot.units_list for command in unit.commands
    } == {"home->towards(center,8)"}


@pytest.mark.unit
def test_rule_policy_does_not_issue_rally_without_townhall() -> None:
    bot = FakeBot(army_count=10, has_townhall=False)

    action = RuleArmyPolicy().manage_army(bot)

    assert action is ArmyAction.RALLY
    assert bot.is_attacking is False
    assert all(unit.commands == [] for unit in bot.units_list)


@pytest.mark.unit
def test_rule_policy_defend_base_attacks_visible_home_threat() -> None:
    bot = FakeBot(army_count=2, enemy_threat=True)
    army = bot.units.of_type(object())

    RuleArmyPolicy().execute(bot, ArmyAction.DEFEND_BASE, army)

    assert {command[0] for unit in bot.units_list for command in unit.commands} == {
        "attack"
    }
    assert {command[1].name for unit in bot.units_list for command in unit.commands} == {
        "enemy-near-home"
    }


@pytest.mark.unit
def test_coverage_policy_holds_when_no_army_exists() -> None:
    bot = FakeBot(army_count=0, is_attacking=True)

    action = CoverageArmyPolicy().manage_army(bot)

    assert action is ArmyAction.HOLD
    assert bot.is_attacking is False


@pytest.mark.unit
def test_coverage_policy_defends_visible_home_threat() -> None:
    bot = FakeBot(army_count=4, enemy_threat=True)

    action = CoverageArmyPolicy().manage_army(bot)

    assert action is ArmyAction.DEFEND_BASE
    assert bot.is_attacking is False


@pytest.mark.unit
def test_coverage_policy_holds_when_attacking_army_is_busy() -> None:
    units = [FakeUnit() for _ in range(10)]
    bot = FakeBot(army_count=10, is_attacking=True, idle_units=[])
    bot.units_list = units
    bot.units = FakeArmy(units, idle_units=[])

    action = CoverageArmyPolicy().manage_army(bot)

    assert action is ArmyAction.HOLD
    assert bot.is_attacking is True
    assert all(unit.commands == [] for unit in units)


@pytest.mark.unit
def test_coverage_policy_retreats_after_large_attack_peak_loss() -> None:
    bot = FakeBot(army_count=9, is_attacking=True)
    bot.army_memory = ArmyMemory()
    bot.army_memory.update(16, is_attacking=True)
    bot.army_memory.update(9, is_attacking=True)

    action = CoverageArmyPolicy().manage_army(bot)

    assert action is ArmyAction.RETREAT_HOME
    assert bot.is_attacking is False
    assert {command[0] for unit in bot.units_list for command in unit.commands} == {
        "move"
    }
