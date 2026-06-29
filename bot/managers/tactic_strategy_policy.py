"""Opt-in tactic-aware strategy policy wrapper."""
from __future__ import annotations

from typing import Any

from bot.managers.strategy_policy import write_strategy_decision_metadata
from bot.managers.tactic_selector import RuleTacticSelector
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import build_strategy_observation
from rl.tactics import TacticState, filter_strategy_action, get_tactic_spec


class TacticAwareStrategyPolicy:
    """Wrap a strategy teacher with a rule tactic selector and action filter.

    This class is deliberately opt-in. The default RuleStrategyPolicy path does
    not instantiate it, so rule/no-op behavior stays unchanged. The first online
    version records tactic metadata for all builds but only filters actions for
    selected opponent builds where data shows a shared weakness.
    """

    def __init__(
        self,
        base_policy: Any,
        *,
        selector: RuleTacticSelector | None = None,
        source: str = "rule",
        filter_opponent_ai_builds: tuple[str, ...] = ("Power",),
    ) -> None:
        self.base_policy = base_policy
        self.selector = selector or RuleTacticSelector()
        self.source = source
        self.filter_opponent_ai_builds = frozenset(filter_opponent_ai_builds)
        self.last_tactic_state: TacticState | None = None
        self.last_proposed_action: StrategyAction | None = None
        self.last_filtered_action: StrategyAction | None = None
        self.last_decision_source = f"tactic-aware-{self.source}"
        self.last_decision_reason = "uninitialized"

    def decide_strategy(self, bot: Any) -> StrategyAction:
        """Choose a filtered strategy action and attach tactic metadata to bot."""
        observation = build_strategy_observation(bot).to_dict()
        opponent_ai_build = str(
            getattr(bot, "episode_metadata", {}).get(
                "opponent_ai_build", "RandomBuild"
            )
        )
        action = self.decide_from_observation(
            observation,
            opponent_ai_build=opponent_ai_build,
        )
        setattr(bot, "last_tactic_state", self.last_tactic_state)
        setattr(bot, "last_tactic_source", self.source)
        setattr(bot, "last_strategy_action_before_tactic_filter", self.last_proposed_action)
        setattr(bot, "last_strategy_action_after_tactic_filter", self.last_filtered_action)
        write_strategy_decision_metadata(
            bot,
            source=self.last_decision_source,
            reason=self.last_decision_reason,
        )
        return action

    def decide_from_observation(
        self,
        observation: dict[str, float],
        *,
        opponent_ai_build: str = "RandomBuild",
    ) -> StrategyAction:
        """Choose a tactic-aware action from a strategy observation dict."""
        proposed = self._base_decide_from_observation(observation)
        tactic_state = self.selector.select(
            observation,
            opponent_ai_build=opponent_ai_build,
            previous_state=self.last_tactic_state,
        )
        spec = get_tactic_spec(tactic_state.current_tactic)
        if opponent_ai_build in self.filter_opponent_ai_builds:
            filtered = filter_strategy_action(spec, proposed, observation)
        else:
            filtered = proposed

        self.last_tactic_state = tactic_state
        self.last_proposed_action = proposed
        self.last_filtered_action = filtered
        self.last_decision_source = f"tactic-aware-{self.source}"
        self.last_decision_reason = _decision_reason(
            tactic_state=tactic_state,
            proposed=proposed,
            filtered=filtered,
        )
        return filtered

    def _base_decide_from_observation(
        self,
        observation: dict[str, float],
    ) -> StrategyAction:
        decide_from_observation = getattr(
            self.base_policy,
            "decide_from_observation",
            None,
        )
        if decide_from_observation is None:
            raise TypeError(
                "TacticAwareStrategyPolicy requires a base policy with "
                "decide_from_observation()."
            )
        return StrategyAction(decide_from_observation(observation))


def _decision_reason(
    *,
    tactic_state: TacticState,
    proposed: StrategyAction,
    filtered: StrategyAction,
) -> str:
    tactic_name = tactic_state.current_tactic.name
    if filtered is not proposed:
        return f"tactic_filter_{tactic_name}_{proposed.name}_to_{filtered.name}"
    return f"tactic_keep_{tactic_name}_{filtered.name}"
