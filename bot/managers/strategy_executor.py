"""Rule executor for low-frequency macro strategy intents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from rl.strategy_actions import StrategyAction


DEFAULT_TARGET_BASES = 2
DEFAULT_GATEWAYS_PER_BASE = 4
DEFAULT_MAX_STATIC_DEFENSE_PER_BASE = 2


@dataclass(frozen=True)
class StrategyExecutionResult:
    """What happened when one strategy intent was applied."""

    action: StrategyAction
    attempted: bool
    effect: str
    blocker: str | None = None
    unit_type: str | None = None
    target: str | None = None


class StrategyExecutor:
    """Translate macro strategy intents into safe rule-bot operations."""

    async def execute(self, bot: Any, action: StrategyAction) -> StrategyExecutionResult:
        """Execute one macro strategy intent if prerequisites allow it."""
        if action is StrategyAction.STAY_COURSE:
            return _execution_result(action, attempted=False, effect="noop")
        if action is StrategyAction.EXPAND:
            return await self._expand(bot, action)
        if action is StrategyAction.ADD_GATEWAYS:
            return await self._add_gateways_by_base_count(bot, action)
        if action is StrategyAction.TECH_ROBO:
            return await self._tech_robo(bot, action)
        if action is StrategyAction.FORGE_UPGRADES:
            return await self._forge_upgrades(bot, action)
        if action is StrategyAction.BUILD_STATIC_DEFENSE:
            return await self._build_static_defense(bot, action)
        if action is StrategyAction.PRODUCE_ARMY:
            robo_result = self._train_robo_units(bot, action=action)
            delegated = False
            train_army = getattr(bot, "_train_army", None)
            if train_army is not None:
                await train_army()
                delegated = True
            if robo_result.effect == "train_robo_unit":
                return robo_result
            if delegated:
                return _execution_result(
                    action,
                    attempted=True,
                    effect="delegate_train_army",
                    blocker=robo_result.blocker,
                )
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker=robo_result.blocker or "missing_train_army_api",
            )
        if action is StrategyAction.BOOST_WORKERS:
            return await self._boost_worker_production(bot, action)
        raise ValueError(f"Unsupported strategy action: {action!r}")

    async def _expand(
        self,
        bot: Any,
        action: StrategyAction,
    ) -> StrategyExecutionResult:
        target_bases = int(getattr(bot, "STRATEGY_TARGET_BASES", DEFAULT_TARGET_BASES))
        townhalls = getattr(bot, "townhalls", None)
        current_bases = int(getattr(townhalls, "amount", 0))
        pending = _already_pending(bot, UnitTypeId.NEXUS)
        if current_bases + pending >= target_bases:
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker="target_bases_reached",
            )
        if not _can_afford(bot, UnitTypeId.NEXUS):
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker="cannot_afford_nexus",
            )
        expand_now = getattr(bot, "expand_now", None)
        if expand_now is None:
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker="missing_expand_api",
            )
        await expand_now()
        return _execution_result(
            action,
            attempted=True,
            effect="expand",
            unit_type="NEXUS",
        )

    async def _add_gateways_by_base_count(
        self,
        bot: Any,
        action: StrategyAction,
    ) -> StrategyExecutionResult:
        bases = max(int(getattr(getattr(bot, "townhalls", None), "amount", 1)), 1)
        gateways_per_base = int(
            getattr(bot, "STRATEGY_GATEWAYS_PER_BASE", DEFAULT_GATEWAYS_PER_BASE)
        )
        target_gateways = max(
            int(getattr(bot, "TARGET_GATEWAYS", 0)),
            bases * gateways_per_base,
        )
        total_gateways = (
            _structure_count(bot, UnitTypeId.GATEWAY)
            + _already_pending(bot, UnitTypeId.GATEWAY)
        )
        if total_gateways >= target_gateways:
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker="target_gateways_reached",
            )
        return await _build_near_power(bot, action, UnitTypeId.GATEWAY)

    async def _tech_robo(
        self,
        bot: Any,
        action: StrategyAction,
    ) -> StrategyExecutionResult:
        if _ready_count(bot, UnitTypeId.CYBERNETICSCORE) <= 0:
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker="missing_cybernetics_core",
            )
        if (
            _structure_count(bot, UnitTypeId.ROBOTICSFACILITY)
            + _already_pending(bot, UnitTypeId.ROBOTICSFACILITY)
            <= 0
        ):
            return await _build_near_power(bot, action, UnitTypeId.ROBOTICSFACILITY)

        robos = _ready_structures(bot, UnitTypeId.ROBOTICSFACILITY)
        return self._train_robo_units(bot, action=action, robos=robos)

    def _train_robo_units(
        self,
        bot: Any,
        *,
        action: StrategyAction,
        robos: Any | None = None,
    ) -> StrategyExecutionResult:
        if robos is None:
            robos = _ready_structures(bot, UnitTypeId.ROBOTICSFACILITY)
        if int(getattr(robos, "amount", 0)) <= 0:
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker="no_ready_robo",
            )
        idle_robos = _idle_iterable(robos)
        if not idle_robos:
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker="no_idle_robo",
            )
        for robo in idle_robos:
            if _unit_count(bot, UnitTypeId.OBSERVER) <= 0 and _can_afford(
                bot, UnitTypeId.OBSERVER
            ):
                _train(robo, UnitTypeId.OBSERVER)
                return _execution_result(
                    action,
                    attempted=True,
                    effect="train_robo_unit",
                    unit_type="OBSERVER",
                    target="robotics_facility",
                )
            if _unit_count(bot, UnitTypeId.OBSERVER) <= 0:
                return _execution_result(
                    action,
                    attempted=False,
                    effect="noop",
                    blocker="cannot_afford_observer",
                )
            if _can_afford(bot, UnitTypeId.IMMORTAL) and _supply_left(bot) >= 4:
                _train(robo, UnitTypeId.IMMORTAL)
                return _execution_result(
                    action,
                    attempted=True,
                    effect="train_robo_unit",
                    unit_type="IMMORTAL",
                    target="robotics_facility",
                )
            if _supply_left(bot) < 4:
                return _execution_result(
                    action,
                    attempted=False,
                    effect="noop",
                    blocker="supply_blocked_immortal",
                )
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker="cannot_afford_immortal",
            )
        return _execution_result(
            action,
            attempted=False,
            effect="noop",
            blocker="no_idle_robo",
        )

    async def _forge_upgrades(
        self,
        bot: Any,
        action: StrategyAction,
    ) -> StrategyExecutionResult:
        if (
            _structure_count(bot, UnitTypeId.FORGE)
            + _already_pending(bot, UnitTypeId.FORGE)
            <= 0
        ):
            return await _build_near_power(bot, action, UnitTypeId.FORGE)

        for forge in _idle_iterable(_ready_structures(bot, UnitTypeId.FORGE)):
            for upgrade in (
                UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
                UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
            ):
                if _upgrade_researched(bot, upgrade) or _upgrade_pending(bot, upgrade):
                    continue
                if _can_afford(bot, upgrade):
                    _research(forge, upgrade)
                    return _execution_result(
                        action,
                        attempted=True,
                        effect="research_upgrade",
                        unit_type=_enum_name(upgrade),
                        target="forge",
                    )
        return _execution_result(
            action,
            attempted=False,
            effect="noop",
            blocker="no_affordable_upgrade",
        )

    async def _build_static_defense(
        self,
        bot: Any,
        action: StrategyAction,
    ) -> StrategyExecutionResult:
        bases = max(int(getattr(getattr(bot, "townhalls", None), "amount", 1)), 1)
        max_per_base = int(
            getattr(
                bot,
                "STRATEGY_MAX_STATIC_DEFENSE_PER_BASE",
                DEFAULT_MAX_STATIC_DEFENSE_PER_BASE,
            )
        )
        max_total = bases * max_per_base
        batteries = _structure_count(bot, UnitTypeId.SHIELDBATTERY)
        cannons = _structure_count(bot, UnitTypeId.PHOTONCANNON)
        if batteries + cannons >= max_total:
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker="static_defense_cap_reached",
            )

        if _ready_count(bot, UnitTypeId.CYBERNETICSCORE) > 0:
            battery_result = await _build_near_power(bot, action, UnitTypeId.SHIELDBATTERY)
            if battery_result.effect != "noop":
                return battery_result
            return battery_result
        if _ready_count(bot, UnitTypeId.FORGE) > 0:
            return await _build_near_power(bot, action, UnitTypeId.PHOTONCANNON)
        return _execution_result(
            action,
            attempted=False,
            effect="noop",
            blocker="missing_static_defense_tech",
        )

    async def _boost_worker_production(
        self,
        bot: Any,
        action: StrategyAction,
    ) -> StrategyExecutionResult:
        if _supply_left(bot) <= 0:
            return _execution_result(
                action,
                attempted=False,
                effect="noop",
                blocker="supply_blocked_probe",
            )
        for nexus in _idle_iterable(_ready_townhalls(bot)):
            if _can_afford(bot, UnitTypeId.PROBE):
                _train(nexus, UnitTypeId.PROBE)
                return _execution_result(
                    action,
                    attempted=True,
                    effect="train_worker",
                    unit_type="PROBE",
                    target="nexus",
                )
        return _execution_result(
            action,
            attempted=False,
            effect="noop",
            blocker="cannot_afford_probe_or_no_idle_nexus",
        )


async def _build_near_power(
    bot: Any,
    action: StrategyAction,
    unit_type: UnitTypeId,
) -> StrategyExecutionResult:
    if not _can_afford(bot, unit_type):
        return _execution_result(
            action,
            attempted=False,
            effect="noop",
            blocker=f"cannot_afford_{_enum_name(unit_type).lower()}",
            unit_type=_enum_name(unit_type),
        )
    pylon = _first_or_random(_ready_structures(bot, UnitTypeId.PYLON))
    if pylon is None:
        return _execution_result(
            action,
            attempted=False,
            effect="noop",
            blocker="missing_power_pylon",
            unit_type=_enum_name(unit_type),
        )
    build = getattr(bot, "build", None)
    if build is None:
        return _execution_result(
            action,
            attempted=False,
            effect="noop",
            blocker="missing_build_api",
            unit_type=_enum_name(unit_type),
        )
    await build(unit_type, near=pylon)
    return _execution_result(
        action,
        attempted=True,
        effect="build_structure",
        unit_type=_enum_name(unit_type),
        target="power_field",
    )


def _execution_result(
    action: StrategyAction,
    *,
    attempted: bool,
    effect: str,
    blocker: str | None = None,
    unit_type: str | None = None,
    target: str | None = None,
) -> StrategyExecutionResult:
    return StrategyExecutionResult(
        action=action,
        attempted=attempted,
        effect=effect,
        blocker=blocker,
        unit_type=unit_type,
        target=target,
    )


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _structures(bot: Any, unit_type: UnitTypeId) -> Any:
    structures = getattr(bot, "structures", None)
    if structures is None:
        return []
    return structures(unit_type)


def _ready_structures(bot: Any, unit_type: UnitTypeId) -> Any:
    structures = _structures(bot, unit_type)
    return getattr(structures, "ready", structures)


def _ready_townhalls(bot: Any) -> Any:
    townhalls = getattr(bot, "townhalls", [])
    return getattr(townhalls, "ready", townhalls)


def _ready_count(bot: Any, unit_type: UnitTypeId) -> int:
    return int(getattr(_ready_structures(bot, unit_type), "amount", 0))


def _structure_count(bot: Any, unit_type: UnitTypeId) -> int:
    return int(getattr(_structures(bot, unit_type), "amount", 0))


def _unit_count(bot: Any, unit_type: UnitTypeId) -> int:
    units = getattr(bot, "units", None)
    if units is None:
        return 0
    of_type = getattr(units, "of_type", None)
    if of_type is None:
        return 0
    try:
        return int(getattr(of_type({unit_type}), "amount", 0))
    except (TypeError, ValueError):
        return 0


def _already_pending(bot: Any, unit_type: UnitTypeId) -> int:
    already_pending = getattr(bot, "already_pending", None)
    if already_pending is None:
        return 0
    try:
        return int(already_pending(unit_type))
    except (TypeError, ValueError):
        return 0


def _upgrade_pending(bot: Any, upgrade: UpgradeId) -> bool:
    already_pending_upgrade = getattr(bot, "already_pending_upgrade", None)
    if already_pending_upgrade is None:
        return False
    try:
        return bool(already_pending_upgrade(upgrade))
    except (TypeError, ValueError):
        return False


def _upgrade_researched(bot: Any, upgrade: UpgradeId) -> bool:
    upgrades = getattr(getattr(bot, "state", None), "upgrades", set())
    try:
        return upgrade in upgrades
    except TypeError:
        return False


def _can_afford(bot: Any, item: Any) -> bool:
    can_afford = getattr(bot, "can_afford", None)
    if can_afford is None:
        return False
    return bool(can_afford(item))


def _supply_left(bot: Any) -> int:
    try:
        return int(getattr(bot, "supply_left", 0))
    except (TypeError, ValueError):
        return 0


def _first_or_random(units: Any) -> Any | None:
    if not getattr(units, "exists", False):
        return None
    random_unit = getattr(units, "random", None)
    if random_unit is not None:
        return random_unit
    first = getattr(units, "first", None)
    if first is not None:
        return first
    try:
        return next(iter(units))
    except StopIteration:
        return None


def _idle_iterable(units: Any) -> Any:
    idle = getattr(units, "idle", units)
    try:
        return list(idle)
    except TypeError:
        return []


def _train(unit: Any, unit_type: UnitTypeId) -> None:
    train = getattr(unit, "train", None)
    if train is not None:
        train(unit_type)


def _research(unit: Any, upgrade: UpgradeId) -> None:
    research = getattr(unit, "research", None)
    if research is not None:
        research(upgrade)
