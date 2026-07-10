"""Experimental LLM policy for low-frequency macro strategy decisions."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from bot.managers.llm_army_policy import (
    LLMPolicyConfig,
    OpenAICompatibleStructuredClient,
    resolve_api_key,
)
from bot.managers.rule_strategy_policy import RuleStrategyPolicy
from bot.managers.strategy_policy import write_strategy_decision_metadata
from rl.strategy_actions import StrategyAction
from rl.strategy_observations import build_strategy_observation


STRATEGY_ACTION_DESCRIPTIONS: dict[str, str] = {
    "STAY_COURSE": "Keep the deterministic macro plan unchanged.",
    "EXPAND": "Attempt to add another Nexus.",
    "ADD_GATEWAYS": "Add Gateway production capacity.",
    "TECH_ROBO": "Build or use Robotics Facility technology.",
    "FORGE_UPGRADES": "Build a Forge or research ground upgrades.",
    "BUILD_STATIC_DEFENSE": "Add Shield Battery or Photon Cannon defense.",
    "PRODUCE_ARMY": "Prioritize army production through existing facilities.",
    "BOOST_WORKERS": "Prioritize Probe production.",
}

STRATEGY_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "reasoning", "confidence"],
    "properties": {
        "action": {
            "type": "string",
            "enum": list(STRATEGY_ACTION_DESCRIPTIONS),
        },
        "reasoning": {
            "type": "string",
            "description": "One short explanation based only on the supplied state.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
}

STRATEGY_SYSTEM_PROMPT = """You are the low-frequency strategy planner for a Protoss StarCraft II bot.
Deterministic code owns unit micro and validates whether an action can execute.
Choose exactly one macro action from the supplied list. Prefer STAY_COURSE when
evidence is weak. Return only JSON matching the requested schema. Keep the
reasoning to one concise explanation, not hidden chain-of-thought."""


@dataclass(frozen=True)
class LLMStrategyDecision:
    action: StrategyAction
    reasoning: str
    confidence: float = 0.0
    source: str = "llm"


class LLMStrategyDecisionClient(Protocol):
    def request_decision(
        self,
        *,
        observation: dict[str, float],
        previous_action: StrategyAction,
    ) -> LLMStrategyDecision:
        """Return one normalized strategy decision."""


class OpenAICompatibleStrategyDecisionClient:
    """Structured OpenAI-compatible client for strategy decisions."""

    def __init__(self, config: LLMPolicyConfig) -> None:
        self.client = OpenAICompatibleStructuredClient(config)

    def request_decision(
        self,
        *,
        observation: dict[str, float],
        previous_action: StrategyAction,
    ) -> LLMStrategyDecision:
        payload = {
            "game": "StarCraft II",
            "race": "Protoss",
            "layer": "strategy",
            "observation": observation,
            "previous_action": previous_action.name,
            "available_actions": STRATEGY_ACTION_DESCRIPTIONS,
        }
        response = self.client.request_json(
            instructions=STRATEGY_SYSTEM_PROMPT,
            payload=payload,
            schema=STRATEGY_DECISION_SCHEMA,
            schema_name="sc2_strategy_decision",
        )
        return parse_llm_strategy_decision(response)


ObservationBuilder = Callable[[Any], dict[str, float]]


class LLMStrategyPolicy:
    """Use an LLM for explicitly enabled low-frequency strategy intents."""

    def __init__(
        self,
        config: LLMPolicyConfig | None = None,
        *,
        client: LLMStrategyDecisionClient | None = None,
        fallback_policy: RuleStrategyPolicy | None = None,
        observation_builder: ObservationBuilder | None = None,
    ) -> None:
        self.config = config or LLMPolicyConfig.from_env()
        self.config.validate()
        self.client = client if client is not None else self._default_client()
        self.fallback_policy = fallback_policy or RuleStrategyPolicy()
        self.observation_builder = observation_builder or _build_observation_dict
        self.last_decision_source = "llm"
        self.last_decision_reason = "uninitialized"

    def decide_strategy(self, bot: Any) -> StrategyAction:
        previous_action = _previous_action(bot)
        if self.client is None:
            return self._fallback(bot, "LLM strategy client is not configured")

        observation = dict(self.observation_builder(bot))
        try:
            decision = self.client.request_decision(
                observation=observation,
                previous_action=previous_action,
            )
        except Exception as exc:
            return self._fallback(bot, f"LLM strategy error: {exc}")

        self.last_decision_source = decision.source
        self.last_decision_reason = decision.reasoning
        write_strategy_decision_metadata(
            bot,
            source=self.last_decision_source,
            reason=self.last_decision_reason,
        )
        info = {
            "game_time": float(getattr(bot, "time", 0.0)),
            "action": decision.action.name,
            "reasoning": decision.reasoning,
            "confidence": float(decision.confidence),
            "source": decision.source,
            "policy_layer": "strategy",
        }
        bot.last_llm_strategy_decision = dict(info)
        self._write_log(info, observation)
        return decision.action

    def _default_client(self) -> LLMStrategyDecisionClient | None:
        if self.config.require_api_key and not resolve_api_key(self.config):
            return None
        return OpenAICompatibleStrategyDecisionClient(self.config)

    def _fallback(self, bot: Any, reason: str) -> StrategyAction:
        action = self.fallback_policy.decide_strategy(bot)
        self.last_decision_source = "rule-fallback"
        self.last_decision_reason = reason
        write_strategy_decision_metadata(
            bot,
            source=self.last_decision_source,
            reason=self.last_decision_reason,
        )
        bot.last_llm_strategy_decision = {
            "game_time": float(getattr(bot, "time", 0.0)),
            "action": action.name,
            "reasoning": reason,
            "confidence": 0.0,
            "source": "rule-fallback",
            "policy_layer": "strategy",
        }
        return action

    def _write_log(
        self,
        info: dict[str, Any],
        observation: dict[str, float],
    ) -> None:
        if self.config.log_path is None:
            return
        row = dict(info)
        row["observation"] = observation
        row["logged_at"] = time.time()
        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_llm_strategy_decision(
    text: str | dict[str, Any],
) -> LLMStrategyDecision:
    data = text if isinstance(text, dict) else _loads_json_object(text)
    try:
        action = StrategyAction[str(data["action"]).strip().upper()]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid LLM strategy action: {data.get('action')!r}") from exc
    reasoning = str(data.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("LLM strategy reasoning must not be empty")
    confidence = _clamp_float(data.get("confidence", 0.0), 0.0, 1.0)
    return LLMStrategyDecision(
        action=action,
        reasoning=reasoning,
        confidence=confidence,
    )


def _build_observation_dict(bot: Any) -> dict[str, float]:
    return build_strategy_observation(bot).to_dict()


def _previous_action(bot: Any) -> StrategyAction:
    try:
        return StrategyAction(
            getattr(bot, "last_strategy_action", StrategyAction.STAY_COURSE)
        )
    except (TypeError, ValueError):
        return StrategyAction.STAY_COURSE


def _loads_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM strategy response was not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM strategy response must be a JSON object")
    return data


def _clamp_float(value: Any, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = lower
    return max(lower, min(upper, number))
