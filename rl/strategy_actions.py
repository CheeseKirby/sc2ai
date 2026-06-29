"""Action helpers for low-frequency macro strategy decisions."""
from __future__ import annotations

from enum import IntEnum


class StrategyAction(IntEnum):
    """Discrete high-level macro strategy intents.

    This action space is intentionally separate from ArmyAction so strategy
    learning can evolve without invalidating existing army checkpoints.
    """

    STAY_COURSE = 0
    EXPAND = 1
    ADD_GATEWAYS = 2
    TECH_ROBO = 3
    FORGE_UPGRADES = 4
    BUILD_STATIC_DEFENSE = 5
    PRODUCE_ARMY = 6
    BOOST_WORKERS = 7


STRATEGY_ACTION_NAMES: dict[int, str] = {
    int(action): action.name for action in StrategyAction
}


def strategy_action_to_int(action: StrategyAction | int) -> int:
    """Normalize a strategy action to its discrete integer id."""
    return int(action)


def strategy_action_from_int(value: int) -> StrategyAction:
    """Return the StrategyAction for a discrete action id."""
    return StrategyAction(value)


def strategy_action_name(action: StrategyAction | int) -> str:
    """Return a stable strategy action name for logs and future datasets."""
    return strategy_action_from_int(int(action)).name
