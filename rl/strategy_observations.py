"""Strategy-specific observation extraction for low-frequency macro policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId


def _unit_type(name: str) -> UnitTypeId | None:
    return getattr(UnitTypeId, name, None)


STRATEGY_OBSERVATION_SCHEMA_VERSION = "strategy_v2"
STRATEGY_OBSERVATION_FIELDS_V1: tuple[str, ...] = (
    "game_time",
    "minerals",
    "vespene",
    "supply_used",
    "supply_cap",
    "supply_left",
    "workers",
    "own_bases",
    "ready_gateways",
    "ready_robo",
    "ready_forge",
    "ready_static_defense",
    "has_cybernetics_core",
    "zealots",
    "stalkers",
    "immortals",
    "observers",
    "sentries",
    "army_count",
    "ground_weapon_level",
    "ground_armor_level",
    "enemy_units_known",
    "enemy_structures_known",
    "enemy_air_units_known",
    "enemy_armored_units_known",
    "enemy_cloaked_units_seen",
    "worker_saturation_ratio",
    "gateway_idle_count",
    "robo_idle_count",
    "base_under_air_threat",
    "base_under_ground_threat",
    "base_under_threat",
    "enemy_to_home_distance",
)
STRATEGY_OBSERVATION_FIELDS: tuple[str, ...] = (
    "game_time",
    "minerals",
    "vespene",
    "supply_used",
    "supply_cap",
    "supply_left",
    "workers",
    "own_bases",
    "pending_bases",
    "ready_gateways",
    "pending_gateways",
    "ready_robo",
    "pending_robo",
    "ready_forge",
    "pending_forge",
    "ready_static_defense",
    "pending_static_defense",
    "has_cybernetics_core",
    "zealots",
    "stalkers",
    "immortals",
    "observers",
    "sentries",
    "army_count",
    "ground_weapon_level",
    "ground_weapon_upgrade_pending",
    "ground_armor_level",
    "ground_armor_upgrade_pending",
    "enemy_units_known",
    "enemy_structures_known",
    "enemy_air_units_known",
    "enemy_armored_units_known",
    "enemy_cloaked_units_seen",
    "worker_saturation_ratio",
    "gateway_idle_count",
    "robo_idle_count",
    "base_under_air_threat",
    "base_under_ground_threat",
    "base_under_threat",
    "enemy_to_home_distance",
)
STRATEGY_OBSERVATION_DEFAULTS: dict[str, float] = {
    "pending_bases": 0.0,
    "pending_gateways": 0.0,
    "pending_robo": 0.0,
    "pending_forge": 0.0,
    "pending_static_defense": 0.0,
    "ground_weapon_upgrade_pending": 0.0,
    "ground_armor_upgrade_pending": 0.0,
}
STRATEGY_OBSERVATION_DETAIL_FIELDS: tuple[str, ...] = (
    "ready_photon_cannons",
    "pending_photon_cannons",
    "ready_shield_batteries",
    "pending_shield_batteries",
)

STRATEGY_ARMY_UNIT_TYPES = frozenset(
    {
        UnitTypeId.ZEALOT,
        UnitTypeId.STALKER,
        UnitTypeId.IMMORTAL,
        UnitTypeId.OBSERVER,
        UnitTypeId.SENTRY,
    }
)
ENEMY_AIR_UNIT_TYPES = frozenset(
    unit_type
    for unit_type in (
        _unit_type("PHOENIX"),
        _unit_type("VOIDRAY"),
        _unit_type("ORACLE"),
        _unit_type("TEMPEST"),
        _unit_type("CARRIER"),
        _unit_type("MOTHERSHIP"),
        _unit_type("OBSERVER"),
        _unit_type("WARPPRISM"),
        _unit_type("BANSHEE"),
        _unit_type("BATTLECRUISER"),
        _unit_type("LIBERATOR"),
        _unit_type("MEDIVAC"),
        _unit_type("RAVEN"),
        _unit_type("VIKINGFIGHTER"),
        _unit_type("MUTALISK"),
        _unit_type("CORRUPTOR"),
        _unit_type("BROODLORD"),
        _unit_type("VIPER"),
        _unit_type("OVERLORD"),
        _unit_type("OVERSEER"),
    )
    if unit_type is not None
)
ENEMY_ARMORED_UNIT_TYPES = frozenset(
    unit_type
    for unit_type in (
        _unit_type("STALKER"),
        _unit_type("IMMORTAL"),
        _unit_type("COLOSSUS"),
        _unit_type("ARCHON"),
        _unit_type("VOIDRAY"),
        _unit_type("TEMPEST"),
        _unit_type("CARRIER"),
        _unit_type("SIEGETANK"),
        _unit_type("THOR"),
        _unit_type("BATTLECRUISER"),
        _unit_type("ULTRALISK"),
        _unit_type("ROACH"),
        _unit_type("RAVAGER"),
        _unit_type("CORRUPTOR"),
    )
    if unit_type is not None
)
ENEMY_CLOAKED_UNIT_TYPES = frozenset(
    unit_type
    for unit_type in (
        _unit_type("DARKTEMPLAR"),
        _unit_type("OBSERVER"),
        _unit_type("WIDOWMINE"),
        _unit_type("BANSHEE"),
        _unit_type("GHOST"),
        _unit_type("LURKERMP"),
    )
    if unit_type is not None
)
GROUND_WEAPON_UPGRADES = (
    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
)
GROUND_ARMOR_UPGRADES = (
    UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
    UpgradeId.PROTOSSGROUNDARMORSLEVEL2,
    UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
)


@dataclass(frozen=True)
class StrategyObservation:
    """Compact numeric state for low-frequency macro strategy decisions."""

    game_time: float
    minerals: float
    vespene: float
    supply_used: float
    supply_cap: float
    supply_left: float
    workers: float
    own_bases: float
    pending_bases: float
    ready_gateways: float
    pending_gateways: float
    ready_robo: float
    pending_robo: float
    ready_forge: float
    pending_forge: float
    ready_static_defense: float
    pending_static_defense: float
    has_cybernetics_core: float
    zealots: float
    stalkers: float
    immortals: float
    observers: float
    sentries: float
    army_count: float
    ground_weapon_level: float
    ground_weapon_upgrade_pending: float
    ground_armor_level: float
    ground_armor_upgrade_pending: float
    enemy_units_known: float
    enemy_structures_known: float
    enemy_air_units_known: float
    enemy_armored_units_known: float
    enemy_cloaked_units_seen: float
    worker_saturation_ratio: float
    gateway_idle_count: float
    robo_idle_count: float
    base_under_air_threat: float
    base_under_ground_threat: float
    base_under_threat: float
    enemy_to_home_distance: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable mapping."""
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        """Return a float32 vector for ML code."""
        return strategy_observation_dict_to_vector(self.to_dict())


@dataclass(frozen=True)
class StrategyObservationDetails:
    """Non-vector strategy details for diagnostics and future schemas."""

    ready_photon_cannons: float
    pending_photon_cannons: float
    ready_shield_batteries: float
    pending_shield_batteries: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


def build_strategy_observation(bot: Any) -> StrategyObservation:
    """Build a strategy-specific abstract observation from a BotAI-like object."""
    zealots = _unit_count(bot, UnitTypeId.ZEALOT)
    stalkers = _unit_count(bot, UnitTypeId.STALKER)
    immortals = _unit_count(bot, UnitTypeId.IMMORTAL)
    observers = _unit_count(bot, UnitTypeId.OBSERVER)
    sentries = _unit_count(bot, UnitTypeId.SENTRY)
    ready_gateways = _ready_structure_count(bot, UnitTypeId.GATEWAY)
    ready_robo = _ready_structure_count(bot, UnitTypeId.ROBOTICSFACILITY)
    ready_forge = _ready_structure_count(bot, UnitTypeId.FORGE)
    ready_static_defense = (
        _ready_structure_count(bot, UnitTypeId.SHIELDBATTERY)
        + _ready_structure_count(bot, UnitTypeId.PHOTONCANNON)
    )
    pending_static_defense = (
        _already_pending(bot, UnitTypeId.SHIELDBATTERY)
        + _already_pending(bot, UnitTypeId.PHOTONCANNON)
    )
    enemy_units = getattr(bot, "enemy_units", [])
    enemy_structures = getattr(bot, "enemy_structures", [])
    near_home_enemies = _enemy_units_near_home(bot)
    near_home_air = _count_types(near_home_enemies, ENEMY_AIR_UNIT_TYPES)
    near_home_ground = max(_amount(near_home_enemies) - near_home_air, 0)

    return StrategyObservation(
        game_time=float(getattr(bot, "time", 0.0)),
        minerals=float(getattr(bot, "minerals", 0.0)),
        vespene=float(getattr(bot, "vespene", 0.0)),
        supply_used=float(getattr(bot, "supply_used", 0.0)),
        supply_cap=float(getattr(bot, "supply_cap", 0.0)),
        supply_left=float(getattr(bot, "supply_left", 0.0)),
        workers=float(_amount(getattr(bot, "workers", []))),
        own_bases=float(_ready_townhall_count(bot)),
        pending_bases=float(_already_pending(bot, UnitTypeId.NEXUS)),
        ready_gateways=float(ready_gateways),
        pending_gateways=float(_already_pending(bot, UnitTypeId.GATEWAY)),
        ready_robo=float(ready_robo),
        pending_robo=float(_already_pending(bot, UnitTypeId.ROBOTICSFACILITY)),
        ready_forge=float(ready_forge),
        pending_forge=float(_already_pending(bot, UnitTypeId.FORGE)),
        ready_static_defense=float(ready_static_defense),
        pending_static_defense=float(pending_static_defense),
        has_cybernetics_core=float(
            _ready_structure_count(bot, UnitTypeId.CYBERNETICSCORE) > 0
        ),
        zealots=float(zealots),
        stalkers=float(stalkers),
        immortals=float(immortals),
        observers=float(observers),
        sentries=float(sentries),
        army_count=float(zealots + stalkers + immortals + observers + sentries),
        ground_weapon_level=float(_upgrade_level(bot, GROUND_WEAPON_UPGRADES)),
        ground_weapon_upgrade_pending=float(
            _upgrade_pending(bot, GROUND_WEAPON_UPGRADES)
        ),
        ground_armor_level=float(_upgrade_level(bot, GROUND_ARMOR_UPGRADES)),
        ground_armor_upgrade_pending=float(
            _upgrade_pending(bot, GROUND_ARMOR_UPGRADES)
        ),
        enemy_units_known=float(_amount(enemy_units)),
        enemy_structures_known=float(_amount(enemy_structures)),
        enemy_air_units_known=float(_count_types(enemy_units, ENEMY_AIR_UNIT_TYPES)),
        enemy_armored_units_known=float(
            _count_types(enemy_units, ENEMY_ARMORED_UNIT_TYPES)
        ),
        enemy_cloaked_units_seen=float(
            _count_cloaked_units(enemy_units, ENEMY_CLOAKED_UNIT_TYPES)
        ),
        worker_saturation_ratio=_worker_saturation_ratio(bot),
        gateway_idle_count=float(
            _idle_count(_ready_structures(bot, UnitTypeId.GATEWAY))
        ),
        robo_idle_count=float(
            _idle_count(_ready_structures(bot, UnitTypeId.ROBOTICSFACILITY))
        ),
        base_under_air_threat=float(near_home_air > 0),
        base_under_ground_threat=float(near_home_ground > 0),
        base_under_threat=float(_amount(near_home_enemies) > 0),
        enemy_to_home_distance=_enemy_to_home_distance(bot),
    )


def build_strategy_observation_details(bot: Any) -> StrategyObservationDetails:
    """Build optional, non-vector strategy details for richer diagnostics."""
    return StrategyObservationDetails(
        ready_photon_cannons=float(_ready_structure_count(bot, UnitTypeId.PHOTONCANNON)),
        pending_photon_cannons=float(_already_pending(bot, UnitTypeId.PHOTONCANNON)),
        ready_shield_batteries=float(
            _ready_structure_count(bot, UnitTypeId.SHIELDBATTERY)
        ),
        pending_shield_batteries=float(
            _already_pending(bot, UnitTypeId.SHIELDBATTERY)
        ),
    )


def strategy_observation_dict_to_vector(
    observation: dict[str, float],
    *,
    allow_missing_defaults: bool = False,
) -> np.ndarray:
    """Vectorize a strategy observation dict using stable field order."""
    observation = normalize_strategy_observation_dict(
        observation,
        allow_missing_defaults=allow_missing_defaults,
    )
    return np.asarray(
        [float(observation[field]) for field in STRATEGY_OBSERVATION_FIELDS],
        dtype=np.float32,
    )


def normalize_strategy_observation_dict(
    observation: dict[str, float],
    *,
    allow_missing_defaults: bool = False,
) -> dict[str, float]:
    """Return a current strategy observation mapping."""
    validate_strategy_observation_dict(
        observation,
        allow_missing_defaults=allow_missing_defaults,
    )
    normalized = dict(observation)
    if allow_missing_defaults:
        for field, value in STRATEGY_OBSERVATION_DEFAULTS.items():
            normalized.setdefault(field, value)
    return normalized


def validate_strategy_observation_dict(
    observation: dict[str, float],
    *,
    allow_missing_defaults: bool = False,
) -> None:
    """Raise ValueError if a strategy observation is missing schema fields."""
    missing = [
        field
        for field in STRATEGY_OBSERVATION_FIELDS
        if field not in observation
        and not (allow_missing_defaults and field in STRATEGY_OBSERVATION_DEFAULTS)
    ]
    if missing:
        raise ValueError(f"Strategy observation missing fields: {', '.join(missing)}")


def infer_strategy_observation_schema_version(
    observation: dict[str, float],
) -> str | None:
    """Infer whether an observation row is strategy_v1 or strategy_v2."""
    if all(field in observation for field in STRATEGY_OBSERVATION_FIELDS):
        return STRATEGY_OBSERVATION_SCHEMA_VERSION
    if all(field in observation for field in STRATEGY_OBSERVATION_FIELDS_V1):
        return "strategy_v1"
    return None


def _unit_count(bot: Any, unit_type: UnitTypeId) -> int:
    return _amount(_units_of_type(getattr(bot, "units", []), {unit_type}))


def _units_of_type(units: Any, unit_types: set[UnitTypeId] | frozenset[UnitTypeId]) -> Any:
    of_type = getattr(units, "of_type", None)
    if of_type is None:
        return []
    try:
        return of_type(unit_types)
    except (TypeError, ValueError):
        return []


def _structures(bot: Any, unit_type: UnitTypeId) -> Any:
    structures = getattr(bot, "structures", None)
    if structures is None:
        return []
    try:
        return structures(unit_type)
    except (TypeError, ValueError):
        return []


def _ready_structures(bot: Any, unit_type: UnitTypeId) -> Any:
    structures = _structures(bot, unit_type)
    return getattr(structures, "ready", structures)


def _ready_structure_count(bot: Any, unit_type: UnitTypeId) -> int:
    return _amount(_ready_structures(bot, unit_type))


def _ready_townhalls(bot: Any) -> Any:
    townhalls = getattr(bot, "townhalls", [])
    return getattr(townhalls, "ready", townhalls)


def _ready_townhall_count(bot: Any) -> int:
    return _amount(_ready_townhalls(bot))


def _already_pending(bot: Any, unit_type: UnitTypeId) -> int:
    already_pending = getattr(bot, "already_pending", None)
    if already_pending is None:
        return 0
    try:
        return int(already_pending(unit_type))
    except (TypeError, ValueError):
        return 0


def _upgrade_pending(bot: Any, upgrades: tuple[UpgradeId, ...]) -> bool:
    already_pending_upgrade = getattr(bot, "already_pending_upgrade", None)
    if already_pending_upgrade is None:
        return False
    for upgrade in upgrades:
        try:
            if already_pending_upgrade(upgrade):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _idle_count(units: Any) -> int:
    idle = getattr(units, "idle", None)
    if idle is None:
        return 0
    return _amount(idle)


def _amount(collection: Any) -> int:
    amount = getattr(collection, "amount", None)
    if amount is not None:
        try:
            return int(amount)
        except (TypeError, ValueError):
            return 0
    try:
        return len(collection)
    except TypeError:
        return 0


def _iter_units(collection: Any) -> list[Any]:
    try:
        return list(collection)
    except TypeError:
        return []


def _count_types(collection: Any, unit_types: frozenset[UnitTypeId]) -> int:
    of_type = getattr(collection, "of_type", None)
    if of_type is not None:
        try:
            return _amount(of_type(unit_types))
        except (TypeError, ValueError):
            pass
    return sum(
        1 for unit in _iter_units(collection) if getattr(unit, "type_id", None) in unit_types
    )


def _count_cloaked_units(collection: Any, unit_types: frozenset[UnitTypeId]) -> int:
    return sum(
        1
        for unit in _iter_units(collection)
        if getattr(unit, "type_id", None) in unit_types
        or bool(getattr(unit, "is_cloaked", False))
    )


def _worker_saturation_ratio(bot: Any) -> float:
    bases = max(_ready_townhall_count(bot), 1)
    target_per_base = max(int(getattr(bot, "TARGET_WORKERS", 22)), 1)
    return float(_amount(getattr(bot, "workers", []))) / float(
        bases * target_per_base
    )


def _upgrade_level(bot: Any, upgrades_by_level: tuple[UpgradeId, ...]) -> int:
    active_upgrades = getattr(getattr(bot, "state", None), "upgrades", set())
    for level, upgrade in reversed(list(enumerate(upgrades_by_level, start=1))):
        if upgrade in active_upgrades:
            return level
    return 0


def _enemy_units_near_home(bot: Any) -> list[Any]:
    enemy_units = getattr(bot, "enemy_units", [])
    if not getattr(enemy_units, "exists", bool(_amount(enemy_units))):
        return []
    townhalls = _ready_townhalls(bot)
    if not getattr(townhalls, "exists", bool(_amount(townhalls))):
        return []

    radius = float(getattr(bot, "BASE_THREAT_RADIUS", 25.0))
    seen: dict[int, Any] = {}
    closer_than = getattr(enemy_units, "closer_than", None)
    if closer_than is None:
        return []
    for townhall in _iter_units(townhalls):
        try:
            threats = closer_than(radius, townhall)
        except (TypeError, ValueError):
            continue
        for unit in _iter_units(threats):
            seen[id(unit)] = unit
    return list(seen.values())


def _enemy_to_home_distance(bot: Any) -> float:
    enemy_units = getattr(bot, "enemy_units", [])
    townhalls = _ready_townhalls(bot)
    if not getattr(townhalls, "exists", bool(_amount(townhalls))):
        return 0.0
    if not getattr(enemy_units, "exists", bool(_amount(enemy_units))):
        return 0.0

    home = getattr(getattr(townhalls, "first", None), "position", None)
    if home is None:
        return 0.0
    closest_to = getattr(enemy_units, "closest_to", None)
    if closest_to is not None:
        enemy = closest_to(home)
        distance_to = getattr(enemy, "distance_to", None)
        if distance_to is not None:
            return float(distance_to(home))
        position = getattr(enemy, "position", None)
        if position is not None and hasattr(position, "distance_to"):
            return float(position.distance_to(home))
    center = getattr(enemy_units, "center", None)
    if center is not None and hasattr(center, "distance_to"):
        return float(center.distance_to(home))
    return 0.0
