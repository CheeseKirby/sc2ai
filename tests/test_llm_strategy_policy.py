from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from bot.managers.llm_army_policy import LLMPolicyConfig
from bot.managers.llm_strategy_policy import (
    LLMStrategyDecision,
    LLMStrategyPolicy,
    parse_llm_strategy_decision,
)
from rl.strategy_actions import StrategyAction


class FakeStrategyDecisionClient:
    def __init__(
        self,
        decision: LLMStrategyDecision | None = None,
        error: Exception | None = None,
    ) -> None:
        self.decision = decision
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def request_decision(
        self,
        *,
        observation: dict[str, float],
        previous_action: StrategyAction,
    ) -> LLMStrategyDecision:
        self.calls.append(
            {
                "observation": observation,
                "previous_action": previous_action,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.decision is not None
        return self.decision


def _config() -> LLMPolicyConfig:
    return LLMPolicyConfig(
        require_api_key=False,
        timeout_seconds=0.1,
        decision_interval=64,
    )


@pytest.mark.unit
def test_llm_strategy_policy_returns_structured_action_and_reason() -> None:
    client = FakeStrategyDecisionClient(
        LLMStrategyDecision(
            action=StrategyAction.TECH_ROBO,
            reasoning="Gas is high and no robotics facility is ready.",
            confidence=0.81,
        )
    )
    bot = SimpleNamespace(last_strategy_action=StrategyAction.STAY_COURSE)
    policy = LLMStrategyPolicy(
        _config(),
        client=client,
        observation_builder=lambda _: {"game_time": 420.0, "vespene": 300.0},
    )

    action = policy.decide_strategy(bot)

    assert action is StrategyAction.TECH_ROBO
    assert client.calls[0]["previous_action"] is StrategyAction.STAY_COURSE
    assert bot.last_strategy_decision_source == "llm"
    assert "robotics facility" in bot.last_strategy_decision_reason
    assert bot.last_llm_strategy_decision["confidence"] == pytest.approx(0.81)


@pytest.mark.unit
def test_llm_strategy_policy_falls_back_to_stay_course_on_client_error() -> None:
    client = FakeStrategyDecisionClient(error=TimeoutError("deadline exceeded"))
    bot = SimpleNamespace(last_strategy_action=StrategyAction.EXPAND)
    policy = LLMStrategyPolicy(
        _config(),
        client=client,
        observation_builder=lambda _: {"game_time": 120.0},
    )

    action = policy.decide_strategy(bot)

    assert action is StrategyAction.STAY_COURSE
    assert bot.last_strategy_decision_source == "rule-fallback"
    assert "deadline exceeded" in bot.last_strategy_decision_reason


@pytest.mark.unit
def test_llm_strategy_policy_falls_back_when_api_key_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_STRATEGY_LLM_KEY", raising=False)
    bot = SimpleNamespace(last_strategy_action=StrategyAction.STAY_COURSE)
    policy = LLMStrategyPolicy(
        LLMPolicyConfig(api_key_env="MISSING_STRATEGY_LLM_KEY"),
        observation_builder=lambda _: {"game_time": 120.0},
    )

    action = policy.decide_strategy(bot)

    assert action is StrategyAction.STAY_COURSE
    assert bot.last_strategy_decision_source == "rule-fallback"
    assert "not configured" in bot.last_strategy_decision_reason


@pytest.mark.unit
def test_parse_llm_strategy_decision_accepts_json_fence_and_clamps_confidence() -> None:
    decision = parse_llm_strategy_decision(
        "```json\n"
        '{"action":"build_static_defense","reasoning":"Air threat near home.",'
        '"confidence":1.4}\n'
        "```"
    )

    assert decision.action is StrategyAction.BUILD_STATIC_DEFENSE
    assert decision.reasoning == "Air threat near home."
    assert decision.confidence == 1.0
