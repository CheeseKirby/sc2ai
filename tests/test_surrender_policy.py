from __future__ import annotations

import asyncio

import pytest

from bot.managers.surrender_policy import SurrenderPolicy, maybe_surrender


class FakeUnitGroup:
    def __init__(self, amount: int = 0, *, threat: bool = False) -> None:
        self.amount = amount
        self._threat = threat

    @property
    def exists(self) -> bool:
        return self.amount > 0

    @property
    def ready(self) -> FakeUnitGroup:
        return self

    def of_type(self, _unit_types: object) -> FakeUnitGroup:
        return self

    def closer_than(self, _radius: float, _target: object) -> FakeUnitGroup:
        if self._threat:
            return self
        return FakeUnitGroup(0)

    @property
    def first(self) -> object:
        return object()

    def __iter__(self):
        return iter([object()] * self.amount)


class FakeStructures:
    def __init__(self, counts: dict[object, int]) -> None:
        self.counts = counts

    def __call__(self, unit_type: object) -> FakeUnitGroup:
        return FakeUnitGroup(self.counts.get(unit_type, 0))


class FakeClient:
    def __init__(self) -> None:
        self.leave_calls = 0

    async def leave(self) -> None:
        self.leave_calls += 1


class FakeBot:
    def __init__(
        self,
        *,
        time: float,
        townhalls: int,
        workers: int,
        army: int,
        production_structures: int,
        enemy_units: int,
        pending_nexus: int = 0,
        pending_production: int = 0,
        base_threat: bool = False,
    ) -> None:
        self.time = time
        self.townhalls = FakeUnitGroup(townhalls)
        self.workers = FakeUnitGroup(workers)
        self.units = FakeUnitGroup(army)
        self.enemy_units = FakeUnitGroup(enemy_units, threat=base_threat)
        self.enemy_structures = FakeUnitGroup(1)
        self.structures = FakeStructures({})
        self.pending: dict[object, int] = {}
        self.chat_messages: list[tuple[str, bool]] = []
        self.client = FakeClient()

        from sc2.ids.unit_typeid import UnitTypeId

        self.pending[UnitTypeId.NEXUS] = pending_nexus
        for unit_type in (
            UnitTypeId.GATEWAY,
            UnitTypeId.WARPGATE,
            UnitTypeId.ROBOTICSFACILITY,
            UnitTypeId.STARGATE,
        ):
            self.structures.counts[unit_type] = production_structures
            self.pending[unit_type] = pending_production

    def already_pending(self, unit_type: object) -> int:
        return self.pending.get(unit_type, 0)

    async def chat_send(self, message: str, team_only: bool = False) -> None:
        self.chat_messages.append((message, team_only))


@pytest.mark.unit
def test_surrender_policy_waits_until_minimum_game_time() -> None:
    bot = FakeBot(
        time=120.0,
        townhalls=0,
        workers=1,
        army=0,
        production_structures=0,
        enemy_units=20,
    )

    assert SurrenderPolicy().should_surrender(bot) is False


@pytest.mark.unit
def test_surrender_policy_keeps_recoverable_game_alive() -> None:
    bot = FakeBot(
        time=720.0,
        townhalls=1,
        workers=18,
        army=1,
        production_structures=4,
        enemy_units=30,
        base_threat=True,
    )

    assert SurrenderPolicy().should_surrender(bot) is False


@pytest.mark.unit
def test_surrender_policy_allows_pending_nexus_recovery() -> None:
    bot = FakeBot(
        time=720.0,
        townhalls=0,
        workers=8,
        army=0,
        production_structures=0,
        enemy_units=12,
        pending_nexus=1,
    )

    assert SurrenderPolicy().should_surrender(bot) is False


@pytest.mark.unit
def test_surrender_policy_surrenders_with_one_dead_base_under_heavy_pressure() -> None:
    bot = FakeBot(
        time=720.0,
        townhalls=1,
        workers=0,
        army=0,
        production_structures=0,
        enemy_units=12,
        base_threat=True,
    )

    assert SurrenderPolicy().should_surrender(bot) is True


@pytest.mark.unit
def test_maybe_surrender_sends_gg_and_leaves_in_hopeless_state() -> None:
    bot = FakeBot(
        time=720.0,
        townhalls=0,
        workers=2,
        army=0,
        production_structures=0,
        enemy_units=12,
    )

    assert asyncio.run(maybe_surrender(bot)) is True
    assert bot.chat_messages == [("gg", False)]
    assert bot.client.leave_calls == 1


@pytest.mark.unit
def test_maybe_surrender_is_one_shot() -> None:
    bot = FakeBot(
        time=720.0,
        townhalls=0,
        workers=2,
        army=0,
        production_structures=0,
        enemy_units=12,
    )

    assert asyncio.run(maybe_surrender(bot)) is True
    assert asyncio.run(maybe_surrender(bot)) is True
    assert bot.chat_messages == [("gg", False)]
    assert bot.client.leave_calls == 1
