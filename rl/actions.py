"""Action helpers shared by trajectory collection and future RL policies."""
from __future__ import annotations

from bot.managers.army_policy import ArmyAction


ACTION_NAMES: dict[int, str] = {int(action): action.name for action in ArmyAction}


def action_to_int(action: ArmyAction | int) -> int:
    """Normalize an army action to its discrete integer id."""
    return int(action)


def action_from_int(value: int) -> ArmyAction:
    """Return the ArmyAction for a discrete action id."""
    return ArmyAction(value)


def action_name(action: ArmyAction | int) -> str:
    """Return a stable action name for logs and datasets."""
    return action_from_int(int(action)).name

