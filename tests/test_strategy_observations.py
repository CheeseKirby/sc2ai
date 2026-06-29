from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from rl.strategy_observations import (
    STRATEGY_OBSERVATION_FIELDS,
    STRATEGY_OBSERVATION_DETAIL_FIELDS,
    STRATEGY_OBSERVATION_FIELDS_V1,
    STRATEGY_OBSERVATION_SCHEMA_VERSION,
    StrategyObservation,
    build_strategy_observation_details,
    build_strategy_observation,
    infer_strategy_observation_schema_version,
    normalize_strategy_observation_dict,
    strategy_observation_dict_to_vector,
    validate_strategy_observation_dict,
)


@dataclass(frozen=True)
class FakePosition:
    x: float
    y: float

    def distance_to(self, other: FakePosition) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class FakeUnit:
    def __init__(
        self,
        unit_type: UnitTypeId,
        position: FakePosition | None = None,
    ) -> None:
        self.type_id = unit_type
        self.position = position or FakePosition(0, 0)

    def distance_to(self, other: FakePosition) -> float:
        return self.position.distance_to(other)


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
    def idle(self) -> FakeUnits:
        return self

    @property
    def first(self) -> FakeUnit:
        return self.units[0]

    @property
    def center(self) -> FakePosition:
        if not self.units:
            return FakePosition(0, 0)
        x = sum(unit.position.x for unit in self.units) / len(self.units)
        y = sum(unit.position.y for unit in self.units) / len(self.units)
        return FakePosition(x, y)

    def __iter__(self):
        return iter(self.units)

    def of_type(self, unit_types: set[UnitTypeId] | frozenset[UnitTypeId]) -> FakeUnits:
        return FakeUnits([unit for unit in self.units if unit.type_id in unit_types])

    def closer_than(self, radius: float, target: object) -> FakeUnits:
        position = getattr(target, "position", target)
        return FakeUnits(
            [
                unit
                for unit in self.units
                if unit.position.distance_to(position) < radius
            ]
        )

    def closest_to(self, target: object) -> FakeUnit:
        position = getattr(target, "position", target)
        return min(self.units, key=lambda unit: unit.position.distance_to(position))


class FakeUnitCollection:
    def __init__(self, counts: dict[UnitTypeId, int]) -> None:
        self.counts = counts

    def of_type(self, unit_types: set[UnitTypeId] | frozenset[UnitTypeId]) -> FakeUnits:
        units: list[FakeUnit] = []
        for unit_type in unit_types:
            units.extend(
                FakeUnit(unit_type, FakePosition(6, 8))
                for _ in range(self.counts.get(unit_type, 0))
            )
        return FakeUnits(units)


class FakeStructures:
    def __init__(
        self,
        counts: dict[UnitTypeId, int],
        idle_counts: dict[UnitTypeId, int] | None = None,
    ) -> None:
        self.counts = counts
        self.idle_counts = idle_counts or {}

    def __call__(self, unit_type: UnitTypeId) -> FakeUnits:
        amount = self.counts.get(unit_type, 0)
        units = [FakeUnit(unit_type) for _ in range(amount)]
        ready = FakeUnits(units)
        idle_amount = min(self.idle_counts.get(unit_type, 0), amount)
        ready.idle_units = FakeUnits(units[:idle_amount])
        return FakeReadyUnits(ready, ready.idle_units)


class FakeReadyUnits(FakeUnits):
    def __init__(self, ready_units: FakeUnits, idle_units: FakeUnits) -> None:
        super().__init__(ready_units.units)
        self._idle = idle_units

    @property
    def ready(self) -> FakeReadyUnits:
        return self

    @property
    def idle(self) -> FakeUnits:
        return self._idle


class FakeTownhalls(FakeUnits):
    @property
    def ready(self) -> FakeTownhalls:
        return self


class FakeState:
    def __init__(self, upgrades: set[UpgradeId]) -> None:
        self.upgrades = upgrades


class FakeBot:
    TARGET_WORKERS = 22
    BASE_THREAT_RADIUS = 25.0
    time = 321.0
    minerals = 850
    vespene = 275
    supply_used = 55
    supply_cap = 78
    supply_left = 23

    def __init__(self) -> None:
        self.units = FakeUnitCollection(
            {
                UnitTypeId.ZEALOT: 4,
                UnitTypeId.STALKER: 6,
                UnitTypeId.IMMORTAL: 2,
                UnitTypeId.OBSERVER: 1,
                UnitTypeId.SENTRY: 1,
            }
        )
        self.workers = FakeUnits([FakeUnit(UnitTypeId.PROBE) for _ in range(38)])
        self.townhalls = FakeTownhalls(
            [
                FakeUnit(UnitTypeId.NEXUS, FakePosition(0, 0)),
                FakeUnit(UnitTypeId.NEXUS, FakePosition(30, 0)),
            ]
        )
        self.structures = FakeStructures(
            {
                UnitTypeId.GATEWAY: 5,
                UnitTypeId.ROBOTICSFACILITY: 1,
                UnitTypeId.FORGE: 1,
                UnitTypeId.CYBERNETICSCORE: 1,
                UnitTypeId.SHIELDBATTERY: 2,
                UnitTypeId.PHOTONCANNON: 1,
            },
            idle_counts={
                UnitTypeId.GATEWAY: 2,
                UnitTypeId.ROBOTICSFACILITY: 1,
            },
        )
        self.enemy_units = FakeUnits(
            [
                FakeUnit(UnitTypeId.PHOENIX, FakePosition(5, 0)),
                FakeUnit(UnitTypeId.IMMORTAL, FakePosition(8, 0)),
                FakeUnit(UnitTypeId.DARKTEMPLAR, FakePosition(60, 0)),
            ]
        )
        self.enemy_structures = FakeUnits([FakeUnit(UnitTypeId.NEXUS)])
        self.state = FakeState({UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1})
        self.pending = {
            UnitTypeId.NEXUS: 1,
            UnitTypeId.GATEWAY: 2,
            UnitTypeId.ROBOTICSFACILITY: 1,
            UnitTypeId.FORGE: 1,
            UnitTypeId.SHIELDBATTERY: 1,
        }
        self.pending_upgrades = {UpgradeId.PROTOSSGROUNDARMORSLEVEL1}

    def already_pending(self, unit_type: UnitTypeId) -> int:
        return self.pending.get(unit_type, 0)

    def already_pending_upgrade(self, upgrade: UpgradeId) -> bool:
        return upgrade in self.pending_upgrades


@pytest.mark.unit
def test_strategy_observation_serializes_to_dict_and_vector() -> None:
    observation = build_strategy_observation(FakeBot())

    data = observation.to_dict()
    vector = observation.to_vector()

    assert data["game_time"] == 321.0
    assert data["minerals"] == 850.0
    assert data["vespene"] == 275.0
    assert data["own_bases"] == 2.0
    assert data["pending_bases"] == 1.0
    assert data["ready_gateways"] == 5.0
    assert data["pending_gateways"] == 2.0
    assert data["ready_robo"] == 1.0
    assert data["pending_robo"] == 1.0
    assert data["ready_forge"] == 1.0
    assert data["pending_forge"] == 1.0
    assert data["ready_static_defense"] == 3.0
    assert data["pending_static_defense"] == 1.0
    assert data["has_cybernetics_core"] == 1.0
    assert data["immortals"] == 2.0
    assert data["observers"] == 1.0
    assert data["sentries"] == 1.0
    assert data["army_count"] == 14.0
    assert data["ground_weapon_level"] == 1.0
    assert data["ground_weapon_upgrade_pending"] == 0.0
    assert data["ground_armor_level"] == 0.0
    assert data["ground_armor_upgrade_pending"] == 1.0
    assert data["enemy_units_known"] == 3.0
    assert data["enemy_structures_known"] == 1.0
    assert data["enemy_air_units_known"] == 1.0
    assert data["enemy_armored_units_known"] == 1.0
    assert data["enemy_cloaked_units_seen"] == 1.0
    assert data["worker_saturation_ratio"] == pytest.approx(38 / 44)
    assert data["gateway_idle_count"] == 2.0
    assert data["robo_idle_count"] == 1.0
    assert data["base_under_air_threat"] == 1.0
    assert data["base_under_ground_threat"] == 1.0
    assert data["base_under_threat"] == 1.0
    assert data["enemy_to_home_distance"] == 5.0
    assert vector.dtype == np.float32
    assert vector.shape == (len(STRATEGY_OBSERVATION_FIELDS),)
    assert tuple(StrategyObservation.__dataclass_fields__) == STRATEGY_OBSERVATION_FIELDS


@pytest.mark.unit
def test_strategy_observation_details_split_static_defense_types() -> None:
    details = build_strategy_observation_details(FakeBot()).to_dict()

    assert details == {
        "ready_photon_cannons": 1.0,
        "pending_photon_cannons": 0.0,
        "ready_shield_batteries": 2.0,
        "pending_shield_batteries": 1.0,
    }
    assert tuple(details) == STRATEGY_OBSERVATION_DETAIL_FIELDS


@pytest.mark.unit
def test_strategy_observation_vectorization_uses_stable_field_order() -> None:
    observation = {
        field: float(index)
        for index, field in enumerate(STRATEGY_OBSERVATION_FIELDS)
    }

    vector = strategy_observation_dict_to_vector(observation)

    assert vector.tolist() == [
        float(index) for index in range(len(STRATEGY_OBSERVATION_FIELDS))
    ]
    assert infer_strategy_observation_schema_version(observation) == (
        STRATEGY_OBSERVATION_SCHEMA_VERSION
    )


@pytest.mark.unit
def test_strategy_observation_validation_rejects_missing_fields() -> None:
    observation = {field: 1.0 for field in STRATEGY_OBSERVATION_FIELDS}
    del observation["worker_saturation_ratio"]

    with pytest.raises(ValueError, match="worker_saturation_ratio"):
        validate_strategy_observation_dict(observation)


@pytest.mark.unit
def test_strategy_v1_observation_can_default_pending_fields() -> None:
    observation = {
        field: float(index)
        for index, field in enumerate(STRATEGY_OBSERVATION_FIELDS_V1)
    }

    normalized = normalize_strategy_observation_dict(
        observation,
        allow_missing_defaults=True,
    )
    vector = strategy_observation_dict_to_vector(
        observation,
        allow_missing_defaults=True,
    )

    assert infer_strategy_observation_schema_version(observation) == "strategy_v1"
    assert normalized["pending_forge"] == 0.0
    assert normalized["ground_weapon_upgrade_pending"] == 0.0
    assert normalized["ground_armor_upgrade_pending"] == 0.0
    assert vector.shape == (len(STRATEGY_OBSERVATION_FIELDS),)
