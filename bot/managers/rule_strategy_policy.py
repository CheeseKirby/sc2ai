"""Default macro strategy policy preserving the current rule baseline."""
from __future__ import annotations

from typing import Any

from bot.managers.strategy_policy import write_strategy_decision_metadata
from rl.strategy_actions import StrategyAction


class RuleStrategyPolicy:
    """No-op strategy policy used by default.

    Existing macro behavior remains in ProtossRuleBot's rule methods. Returning
    STAY_COURSE keeps the new strategy layer present but behavior-preserving.
    """

    last_decision_source: str = "rule"
    last_decision_reason: str = "default_stay_course"

    def decide_strategy(self, bot: Any) -> StrategyAction:
        """Return the default no-op macro strategy action."""
        self.last_decision_source = "rule"
        self.last_decision_reason = "default_stay_course"
        write_strategy_decision_metadata(
            bot,
            source=self.last_decision_source,
            reason=self.last_decision_reason,
        )
        return StrategyAction.STAY_COURSE
