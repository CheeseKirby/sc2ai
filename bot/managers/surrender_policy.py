"""Conservative GG/surrender policy for unrecoverable game states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sc2.ids.unit_typeid import UnitTypeId

from bot.managers.rule_army_policy import base_under_threat


PROTOSS_ARMY_UNIT_TYPES = frozenset(
    {
        UnitTypeId.ZEALOT,
        UnitTypeId.STALKER,
        UnitTypeId.SENTRY,
        UnitTypeId.ADEPT,
        UnitTypeId.IMMORTAL,
        UnitTypeId.OBSERVER,
        UnitTypeId.WARPPRISM,
        UnitTypeId.VOIDRAY,
        UnitTypeId.PHOENIX,
        UnitTypeId.ORACLE,
        UnitTypeId.CARRIER,
        UnitTypeId.TEMPEST,
        UnitTypeId.COLOSSUS,
        UnitTypeId.DISRUPTOR,
        UnitTypeId.HIGHTEMPLAR,
        UnitTypeId.DARKTEMPLAR,
        UnitTypeId.ARCHON,
    }
)

PROTOSS_PRODUCTION_STRUCTURE_TYPES = frozenset(
    {
        UnitTypeId.GATEWAY,
        UnitTypeId.WARPGATE,
        UnitTypeId.ROBOTICSFACILITY,
        UnitTypeId.STARGATE,
    }
)


@dataclass(frozen=True)
class SurrenderPolicy:
    """Decide whether the bot should type gg and leave the game.

    The defaults intentionally only cover near-terminal states. Being behind,
    losing an attack, or getting pressured at one base should not trigger GG.
    """

    min_game_time: float = 360.0
    max_workers_without_base: int = 4
    max_army: int = 2
    max_production_structures: int = 1
    min_enemy_units_seen: int = 8
    rebuild_nexus_minerals: int = 400

    def should_surrender(self, bot: Any) -> bool:
        """Return True only when the visible state is almost unrecoverable."""
        if _float_attr(bot, "time") < self.min_game_time:
            return False

        ready_bases = _ready_amount(getattr(bot, "townhalls", None))
        pending_bases = _pending_count(bot, UnitTypeId.NEXUS)
        workers = _amount(getattr(bot, "workers", None))
        army = _army_amount(bot)
        production = _production_structure_count(bot)
        enemy_units = _amount(getattr(bot, "enemy_units", None))
        enemy_structures = _amount(getattr(bot, "enemy_structures", None))

        if (
            ready_bases == 0
            and pending_bases == 0
            and workers > 0
            and _int_attr(bot, "minerals") >= self.rebuild_nexus_minerals
        ):
            return False

        economy_gone = (
            ready_bases == 0
            and pending_bases == 0
            and workers <= self.max_workers_without_base
        )
        army_gone = army <= self.max_army
        production_gone = production <= self.max_production_structures
        has_enemy_pressure = _safe_base_under_threat(bot) or (
            enemy_units >= self.min_enemy_units_seen
        )

        if (
            ready_bases <= 1
            and pending_bases == 0
            and workers == 0
            and army_gone
            and production_gone
            and has_enemy_pressure
        ):
            return True

        if not (economy_gone and army_gone and production_gone):
            return False

        if workers == 0:
            return True

        has_scouted_enemy_position = enemy_structures > 0
        return has_enemy_pressure or has_scouted_enemy_position


async def maybe_surrender(
    bot: Any,
    policy: SurrenderPolicy | None = None,
    *,
    message: str = "gg",
) -> bool:
    """Send gg and leave once if the surrender policy fires.

    Returns True when the caller should skip the rest of the on-step logic.
    """
    if getattr(bot, "_gg_surrendered", False):
        return True

    active_policy = policy or getattr(bot, "surrender_policy", SurrenderPolicy())
    if not active_policy.should_surrender(bot):
        return False

    setattr(bot, "_gg_surrendered", True)
    try:
        await bot.chat_send(message)
    except Exception as exc:
        print(f"[{getattr(bot, 'NAME', 'Bot')}] Failed to send gg: {exc!r}")

    client = getattr(bot, "client", None)
    leave = getattr(client, "leave", None)
    if leave is None:
        print(f"[{getattr(bot, 'NAME', 'Bot')}] Cannot surrender: no client.leave()")
        return True

    try:
        await leave()
    except Exception as exc:
        print(f"[{getattr(bot, 'NAME', 'Bot')}] Failed to leave after gg: {exc!r}")
    return True


def _army_amount(bot: Any) -> int:
    units = getattr(bot, "units", None)
    of_type = getattr(units, "of_type", None)
    if of_type is None:
        return 0
    try:
        return _amount(of_type(PROTOSS_ARMY_UNIT_TYPES))
    except Exception:
        return 0


def _production_structure_count(bot: Any) -> int:
    ready = sum(
        _structure_amount(bot, unit_type)
        for unit_type in PROTOSS_PRODUCTION_STRUCTURE_TYPES
    )
    pending = sum(
        _pending_count(bot, unit_type)
        for unit_type in PROTOSS_PRODUCTION_STRUCTURE_TYPES
    )
    return ready + pending


def _structure_amount(bot: Any, unit_type: UnitTypeId) -> int:
    structures = getattr(bot, "structures", None)
    if not callable(structures):
        return 0
    try:
        return _ready_amount(structures(unit_type))
    except Exception:
        return 0


def _pending_count(bot: Any, unit_type: UnitTypeId) -> int:
    already_pending = getattr(bot, "already_pending", None)
    if already_pending is None:
        return 0
    try:
        return max(0, int(already_pending(unit_type)))
    except Exception:
        return 0


def _safe_base_under_threat(bot: Any) -> bool:
    try:
        return bool(base_under_threat(bot))
    except Exception:
        return False


def _ready_amount(collection: Any) -> int:
    ready = getattr(collection, "ready", collection)
    return _amount(ready)


def _amount(collection: Any) -> int:
    try:
        return max(0, int(getattr(collection, "amount", 0) or 0))
    except Exception:
        return 0


def _float_attr(obj: Any, name: str) -> float:
    try:
        return float(getattr(obj, name, 0.0) or 0.0)
    except Exception:
        return 0.0


def _int_attr(obj: Any, name: str) -> int:
    try:
        return int(getattr(obj, name, 0) or 0)
    except Exception:
        return 0
