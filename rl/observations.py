"""Low-dimensional observation extraction for high-level army policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId

from bot.managers.rule_army_policy import COMBAT_UNIT_TYPES, base_under_threat


OBSERVATION_SCHEMA_VERSION = 3
OBSERVATION_FIELDS_V1: tuple[str, ...] = (
    "game_time",
    "minerals",
    "vespene",
    "supply_used",
    "supply_cap",
    "supply_left",
    "workers",
    "townhalls",
    "gateways",
    "has_cybernetics_core",
    "zealots",
    "stalkers",
    "army_count",
    "is_attacking",
    "enemy_units_known",
    "enemy_structures_known",
    "army_to_home_distance",
    "army_to_enemy_start_distance",
)
OBSERVATION_FIELDS_V2: tuple[str, ...] = (
    *OBSERVATION_FIELDS_V1,
    "base_under_threat",
    "enemy_to_home_distance",
    "army_idle_count",
    "army_busy_count",
)
OBSERVATION_FIELDS_V3: tuple[str, ...] = (
    *OBSERVATION_FIELDS_V2,
    "attack_army_peak",
    "army_lost_from_peak",
    "army_lost_from_peak_ratio",
    "army_count_delta",
)
OBSERVATION_FIELDS: tuple[str, ...] = OBSERVATION_FIELDS_V3
OBSERVATION_DEFAULTS_V2: dict[str, float] = {
    "base_under_threat": 0.0,
    "enemy_to_home_distance": 0.0,
    "army_idle_count": 0.0,
    "army_busy_count": 0.0,
}
OBSERVATION_DEFAULTS_V3: dict[str, float] = {
    "attack_army_peak": 0.0,
    "army_lost_from_peak": 0.0,
    "army_lost_from_peak_ratio": 0.0,
    "army_count_delta": 0.0,
}
OBSERVATION_DEFAULTS: dict[str, float] = {
    **OBSERVATION_DEFAULTS_V2,
    **OBSERVATION_DEFAULTS_V3,
}


@dataclass(frozen=True)
class ArmyObservation:
    """Compact numeric state used for imitation learning and PPO."""

    game_time: float
    minerals: float
    vespene: float
    supply_used: float
    supply_cap: float
    supply_left: float
    workers: float
    townhalls: float
    gateways: float
    has_cybernetics_core: float
    zealots: float
    stalkers: float
    army_count: float
    is_attacking: float
    enemy_units_known: float
    enemy_structures_known: float
    army_to_home_distance: float
    army_to_enemy_start_distance: float
    base_under_threat: float
    enemy_to_home_distance: float
    army_idle_count: float
    army_busy_count: float
    attack_army_peak: float
    army_lost_from_peak: float
    army_lost_from_peak_ratio: float
    army_count_delta: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable mapping."""
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        """Return a float32 vector for ML code."""
        return observation_dict_to_vector(self.to_dict())


def build_observation(bot: Any) -> ArmyObservation:
    """Build the first-version abstract observation from a BotAI-like object."""
    army = bot.units.of_type(COMBAT_UNIT_TYPES)
    zealots = bot.units.of_type({UnitTypeId.ZEALOT})
    stalkers = bot.units.of_type({UnitTypeId.STALKER})
    gateways = bot.structures(UnitTypeId.GATEWAY)
    cybernetics_cores = bot.structures(UnitTypeId.CYBERNETICSCORE)
    idle_count = _idle_count(army)
    army_count = float(army.amount)
    army_memory = getattr(bot, "army_memory", None)

    return ArmyObservation(
        game_time=float(getattr(bot, "time", 0.0)),
        minerals=float(getattr(bot, "minerals", 0.0)),
        vespene=float(getattr(bot, "vespene", 0.0)),
        supply_used=float(getattr(bot, "supply_used", 0.0)),
        supply_cap=float(getattr(bot, "supply_cap", 0.0)),
        supply_left=float(getattr(bot, "supply_left", 0.0)),
        workers=float(bot.workers.amount),
        townhalls=float(bot.townhalls.amount),
        gateways=float(gateways.amount),
        has_cybernetics_core=float(cybernetics_cores.amount > 0),
        zealots=float(zealots.amount),
        stalkers=float(stalkers.amount),
        army_count=army_count,
        is_attacking=float(bool(getattr(bot, "is_attacking", False))),
        enemy_units_known=float(getattr(bot.enemy_units, "amount", 0)),
        enemy_structures_known=float(getattr(bot.enemy_structures, "amount", 0)),
        army_to_home_distance=_distance_to_home(bot, army),
        army_to_enemy_start_distance=_distance_to_enemy_start(bot, army),
        base_under_threat=float(base_under_threat(bot)),
        enemy_to_home_distance=_enemy_to_home_distance(bot),
        army_idle_count=float(idle_count),
        army_busy_count=max(army_count - float(idle_count), 0.0),
        attack_army_peak=_memory_float(army_memory, "attack_army_peak"),
        army_lost_from_peak=_memory_float(army_memory, "army_lost_from_peak"),
        army_lost_from_peak_ratio=_memory_float(
            army_memory,
            "army_lost_from_peak_ratio",
        ),
        army_count_delta=_memory_float(army_memory, "army_count_delta"),
    )


def observation_dict_to_vector(
    observation: dict[str, float],
    *,
    allow_missing_defaults: bool = False,
) -> np.ndarray:
    """Vectorize an observation dict using the stable schema order."""
    observation = normalize_observation_dict(
        observation,
        allow_missing_defaults=allow_missing_defaults,
    )
    return np.asarray(
        [float(observation[field]) for field in OBSERVATION_FIELDS],
        dtype=np.float32,
    )


def normalize_observation_dict(
    observation: dict[str, float],
    *,
    allow_missing_defaults: bool = False,
) -> dict[str, float]:
    """Return a current-schema observation, optionally defaulting old rows."""
    validate_observation_dict(
        observation,
        allow_missing_defaults=allow_missing_defaults,
    )
    normalized = dict(observation)
    if allow_missing_defaults:
        for field, value in OBSERVATION_DEFAULTS.items():
            normalized.setdefault(field, value)
    return normalized


def validate_observation_dict(
    observation: dict[str, float],
    *,
    allow_missing_defaults: bool = False,
) -> None:
    """Raise ValueError if an observation is missing schema fields."""
    missing = [field for field in OBSERVATION_FIELDS if field not in observation]
    if allow_missing_defaults:
        missing = [
            field for field in missing if field not in OBSERVATION_DEFAULTS
        ]
    if missing:
        raise ValueError(f"Observation missing fields: {', '.join(missing)}")


def infer_observation_schema_version(observation: dict[str, float]) -> int | None:
    """Infer whether an observation row looks like schema v1, v2, or v3."""
    if all(field in observation for field in OBSERVATION_FIELDS_V3):
        return 3
    if all(field in observation for field in OBSERVATION_FIELDS_V2):
        return 2
    if all(field in observation for field in OBSERVATION_FIELDS_V1):
        return 1
    return None


def _distance_to_home(bot: Any, army: Any) -> float:
    if not getattr(army, "exists", False) or not bot.townhalls.exists:
        return 0.0
    return float(army.center.distance_to(bot.townhalls.first.position))


def _distance_to_enemy_start(bot: Any, army: Any) -> float:
    if not getattr(army, "exists", False) or not bot.enemy_start_locations:
        return 0.0
    return float(army.center.distance_to(bot.enemy_start_locations[0]))


def _enemy_to_home_distance(bot: Any) -> float:
    if not bot.townhalls.exists or not getattr(bot.enemy_units, "exists", False):
        return 0.0
    home = bot.townhalls.first.position
    enemy_units = bot.enemy_units
    if hasattr(enemy_units, "closest_to"):
        enemy = enemy_units.closest_to(home)
        if hasattr(enemy, "distance_to"):
            return float(enemy.distance_to(home))
        if hasattr(enemy, "position"):
            return float(enemy.position.distance_to(home))
    if hasattr(enemy_units, "center"):
        return float(enemy_units.center.distance_to(home))
    return 0.0


def _idle_count(army: Any) -> int:
    idle = getattr(army, "idle", None)
    if idle is None:
        return 0
    amount = getattr(idle, "amount", None)
    if amount is not None:
        return int(amount)
    try:
        return len(idle)
    except TypeError:
        return 0


def _memory_float(memory: Any, field: str) -> float:
    if memory is None:
        return 0.0
    try:
        return float(getattr(memory, field, 0.0))
    except (TypeError, ValueError):
        return 0.0
