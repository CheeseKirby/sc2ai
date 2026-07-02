from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from bot.managers.rule_strategy_policy import RuleStrategyPolicy
from bot.managers.strategy_executor import StrategyExecutionResult, StrategyExecutor
from rl.strategy_actions import (
    STRATEGY_ACTION_NAMES,
    StrategyAction,
    strategy_action_from_int,
    strategy_action_name,
    strategy_action_to_int,
)


class FakeUnit:
    def __init__(self, name: str = "unit") -> None:
        self.name = name
        self.train_calls: list[UnitTypeId] = []
        self.research_calls: list[UpgradeId] = []

    def train(self, unit_type: UnitTypeId) -> None:
        self.train_calls.append(unit_type)

    def research(self, upgrade: UpgradeId) -> None:
        self.research_calls.append(upgrade)


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


class FakeStructures:
    def __init__(self, counts: dict[UnitTypeId, list[FakeUnit]]) -> None:
        self.counts = counts

    def __call__(self, unit_type: UnitTypeId) -> FakeUnits:
        return FakeUnits(self.counts.get(unit_type, []))


class FakeUnitCollection:
    def __init__(self, counts: dict[UnitTypeId, list[FakeUnit]]) -> None:
        self.counts = counts

    def of_type(self, unit_types: set[UnitTypeId] | frozenset[UnitTypeId]) -> FakeUnits:
        units: list[FakeUnit] = []
        for unit_type in unit_types:
            units.extend(self.counts.get(unit_type, []))
        return FakeUnits(units)


class FakeBot:
    TARGET_GATEWAYS = 4
    STRATEGY_TARGET_BASES = 2
    STRATEGY_GATEWAYS_PER_BASE = 4
    STRATEGY_MAX_STATIC_DEFENSE_PER_BASE = 2

    def __init__(self) -> None:
        self.pylon = FakeUnit("pylon")
        self.nexus = FakeUnit("nexus")
        self.robo = FakeUnit("robo")
        self.forge = FakeUnit("forge")
        self.structure_counts: dict[UnitTypeId, list[FakeUnit]] = {
            UnitTypeId.PYLON: [self.pylon],
            UnitTypeId.NEXUS: [self.nexus],
        }
        self.unit_counts: dict[UnitTypeId, list[FakeUnit]] = {}
        self.units = FakeUnitCollection(self.unit_counts)
        self.structures = FakeStructures(self.structure_counts)
        self.townhalls = FakeUnits([self.nexus])
        self.pending: dict[UnitTypeId, int] = {}
        self.pending_upgrades: set[UpgradeId] = set()
        self.completed_upgrades: set[UpgradeId] = set()
        self.state = type("FakeState", (), {"upgrades": self.completed_upgrades})()
        self.affordable: set[object] = {
            UnitTypeId.NEXUS,
            UnitTypeId.GATEWAY,
            UnitTypeId.ROBOTICSFACILITY,
            UnitTypeId.OBSERVER,
            UnitTypeId.IMMORTAL,
            UnitTypeId.FORGE,
            UnitTypeId.SHIELDBATTERY,
            UnitTypeId.PHOTONCANNON,
            UnitTypeId.PROBE,
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
            UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
        }
        self.supply_left = 20
        self.build_calls: list[tuple[UnitTypeId, FakeUnit]] = []
        self.expand_calls = 0
        self.train_army_calls = 0

    def already_pending(self, unit_type: UnitTypeId) -> int:
        return self.pending.get(unit_type, 0)

    def already_pending_upgrade(self, upgrade: UpgradeId) -> bool:
        return upgrade in self.pending_upgrades

    def can_afford(self, item: object) -> bool:
        return item in self.affordable

    async def build(self, unit_type: UnitTypeId, *, near: FakeUnit) -> None:
        self.build_calls.append((unit_type, near))

    async def expand_now(self) -> None:
        self.expand_calls += 1

    async def _train_army(self) -> None:
        self.train_army_calls += 1


@pytest.mark.unit
def test_strategy_action_helpers_are_stable() -> None:
    assert strategy_action_to_int(StrategyAction.EXPAND) == 1
    assert strategy_action_from_int(3) is StrategyAction.TECH_ROBO
    assert strategy_action_name(StrategyAction.BUILD_STATIC_DEFENSE) == (
        "BUILD_STATIC_DEFENSE"
    )
    assert STRATEGY_ACTION_NAMES[0] == "STAY_COURSE"


@pytest.mark.unit
def test_rule_strategy_policy_preserves_default_noop() -> None:
    bot = SimpleNamespace()
    policy = RuleStrategyPolicy()

    assert policy.decide_strategy(bot) is StrategyAction.STAY_COURSE
    assert policy.last_decision_source == "rule"
    assert policy.last_decision_reason == "default_stay_course"
    assert bot.last_strategy_decision_source == "rule"
    assert bot.last_strategy_decision_reason == "default_stay_course"


@pytest.mark.unit
def test_strategy_executor_stay_course_is_noop() -> None:
    bot = FakeBot()

    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.STAY_COURSE))

    assert bot.build_calls == []
    assert bot.expand_calls == 0


@pytest.mark.unit
def test_strategy_executor_expands_when_allowed() -> None:
    bot = FakeBot()

    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.EXPAND))

    assert bot.expand_calls == 1


@pytest.mark.unit
def test_strategy_executor_adds_gateway_near_power() -> None:
    bot = FakeBot()

    result = asyncio.run(StrategyExecutor().execute(bot, StrategyAction.ADD_GATEWAYS))

    assert bot.build_calls == [(UnitTypeId.GATEWAY, bot.pylon)]
    assert result == StrategyExecutionResult(
        action=StrategyAction.ADD_GATEWAYS,
        attempted=True,
        effect="build_structure",
        unit_type="GATEWAY",
        target="power_field",
    )


@pytest.mark.unit
def test_strategy_executor_tech_robo_builds_robo_after_cyber() -> None:
    bot = FakeBot()
    bot.structure_counts[UnitTypeId.CYBERNETICSCORE] = [FakeUnit("cyber")]

    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.TECH_ROBO))

    assert bot.build_calls == [(UnitTypeId.ROBOTICSFACILITY, bot.pylon)]


@pytest.mark.unit
def test_strategy_executor_tech_robo_trains_observer_before_immortal() -> None:
    bot = FakeBot()
    bot.structure_counts[UnitTypeId.CYBERNETICSCORE] = [FakeUnit("cyber")]
    bot.structure_counts[UnitTypeId.ROBOTICSFACILITY] = [bot.robo]

    result = asyncio.run(StrategyExecutor().execute(bot, StrategyAction.TECH_ROBO))

    assert bot.robo.train_calls == [UnitTypeId.OBSERVER]
    assert result.effect == "train_robo_unit"
    assert result.unit_type == "OBSERVER"
    assert result.blocker is None


@pytest.mark.unit
def test_strategy_executor_tech_robo_trains_immortal_when_observer_exists() -> None:
    bot = FakeBot()
    bot.structure_counts[UnitTypeId.CYBERNETICSCORE] = [FakeUnit("cyber")]
    bot.structure_counts[UnitTypeId.ROBOTICSFACILITY] = [bot.robo]
    bot.unit_counts[UnitTypeId.OBSERVER] = [FakeUnit("observer")]

    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.TECH_ROBO))

    assert bot.robo.train_calls == [UnitTypeId.IMMORTAL]


@pytest.mark.unit
def test_strategy_executor_produce_army_uses_ready_robo_for_observer() -> None:
    bot = FakeBot()
    bot.structure_counts[UnitTypeId.ROBOTICSFACILITY] = [bot.robo]

    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.PRODUCE_ARMY))

    assert bot.robo.train_calls == [UnitTypeId.OBSERVER]
    assert bot.train_army_calls == 1


@pytest.mark.unit
def test_strategy_executor_produce_army_uses_ready_robo_for_immortal() -> None:
    bot = FakeBot()
    bot.structure_counts[UnitTypeId.ROBOTICSFACILITY] = [bot.robo]
    bot.unit_counts[UnitTypeId.OBSERVER] = [FakeUnit("observer")]

    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.PRODUCE_ARMY))

    assert bot.robo.train_calls == [UnitTypeId.IMMORTAL]
    assert bot.train_army_calls == 1


@pytest.mark.unit
def test_strategy_executor_forge_upgrades_builds_forge_then_researches() -> None:
    bot = FakeBot()

    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.FORGE_UPGRADES))

    assert bot.build_calls == [(UnitTypeId.FORGE, bot.pylon)]

    bot.build_calls.clear()
    bot.structure_counts[UnitTypeId.FORGE] = [bot.forge]
    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.FORGE_UPGRADES))

    assert bot.forge.research_calls == [UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1]


@pytest.mark.unit
def test_strategy_executor_forge_upgrades_skips_completed_upgrades() -> None:
    bot = FakeBot()
    bot.structure_counts[UnitTypeId.FORGE] = [bot.forge]
    bot.completed_upgrades.add(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1)

    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.FORGE_UPGRADES))

    assert bot.forge.research_calls == [UpgradeId.PROTOSSGROUNDARMORSLEVEL1]


@pytest.mark.unit
def test_strategy_executor_static_defense_requires_power_and_tech() -> None:
    bot = FakeBot()
    bot.structure_counts[UnitTypeId.PYLON] = []

    result = asyncio.run(
        StrategyExecutor().execute(bot, StrategyAction.BUILD_STATIC_DEFENSE)
    )

    assert bot.build_calls == []
    assert result == StrategyExecutionResult(
        action=StrategyAction.BUILD_STATIC_DEFENSE,
        attempted=False,
        effect="noop",
        blocker="missing_static_defense_tech",
    )

    bot.structure_counts[UnitTypeId.PYLON] = [bot.pylon]
    bot.structure_counts[UnitTypeId.CYBERNETICSCORE] = [FakeUnit("cyber")]
    result = asyncio.run(
        StrategyExecutor().execute(bot, StrategyAction.BUILD_STATIC_DEFENSE)
    )

    assert bot.build_calls == [(UnitTypeId.SHIELDBATTERY, bot.pylon)]
    assert result.effect == "build_structure"
    assert result.unit_type == "SHIELDBATTERY"


@pytest.mark.unit
def test_strategy_executor_prefers_photon_cannon_when_forge_is_ready() -> None:
    bot = FakeBot()
    bot.structure_counts[UnitTypeId.CYBERNETICSCORE] = [FakeUnit("cyber")]
    bot.structure_counts[UnitTypeId.FORGE] = [FakeUnit("forge")]

    result = asyncio.run(
        StrategyExecutor().execute(bot, StrategyAction.BUILD_STATIC_DEFENSE)
    )

    assert bot.build_calls == [(UnitTypeId.PHOTONCANNON, bot.pylon)]
    assert result.effect == "build_structure"
    assert result.unit_type == "PHOTONCANNON"


@pytest.mark.unit
def test_strategy_executor_falls_back_to_battery_when_cannon_unaffordable() -> None:
    bot = FakeBot()
    bot.structure_counts[UnitTypeId.CYBERNETICSCORE] = [FakeUnit("cyber")]
    bot.structure_counts[UnitTypeId.FORGE] = [FakeUnit("forge")]
    bot.affordable.remove(UnitTypeId.PHOTONCANNON)

    result = asyncio.run(
        StrategyExecutor().execute(bot, StrategyAction.BUILD_STATIC_DEFENSE)
    )

    assert bot.build_calls == [(UnitTypeId.SHIELDBATTERY, bot.pylon)]
    assert result.effect == "build_structure"
    assert result.unit_type == "SHIELDBATTERY"


@pytest.mark.unit
def test_strategy_executor_reports_produce_army_robo_blocker_when_delegating() -> None:
    bot = FakeBot()

    result = asyncio.run(StrategyExecutor().execute(bot, StrategyAction.PRODUCE_ARMY))

    assert bot.train_army_calls == 1
    assert result == StrategyExecutionResult(
        action=StrategyAction.PRODUCE_ARMY,
        attempted=True,
        effect="delegate_train_army",
        blocker="no_ready_robo",
    )


@pytest.mark.unit
def test_strategy_executor_production_intents_delegate_safely() -> None:
    bot = FakeBot()

    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.BOOST_WORKERS))
    asyncio.run(StrategyExecutor().execute(bot, StrategyAction.PRODUCE_ARMY))

    assert bot.nexus.train_calls == [UnitTypeId.PROBE]
    assert bot.train_army_calls == 1
