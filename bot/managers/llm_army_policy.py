"""LLM-backed high-level army policy.

The LLM is deliberately constrained to the existing five-action army API. It
does not control economy, production, or raw unit micro; the rule bot keeps
those systems stable while this policy chooses explainable army intents.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from bot.managers.army_policy import ArmyAction
from bot.managers.rule_army_policy import (
    COMBAT_UNIT_TYPES,
    RuleArmyPolicy,
    base_under_threat,
)
from rl.actions import action_name
from rl.observations import build_observation


DEFAULT_LLM_PROVIDER = "openai-responses"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_API_KEY_ENV = "OPENAI_API_KEY"
SUPPORTED_PROVIDERS = frozenset({"openai-responses", "openai-chat"})

ACTION_DESCRIPTIONS: dict[str, str] = {
    "RALLY": "Move idle army units to the home rally point.",
    "ATTACK_MAIN": "Send idle army units toward the enemy start location.",
    "RETREAT_HOME": "Pull idle army units back toward the home rally point.",
    "DEFEND_BASE": "Attack known enemy units threatening the home base.",
    "HOLD": "Do not issue new army orders this step.",
}

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "reasoning", "confidence"],
    "properties": {
        "action": {
            "type": "string",
            "enum": list(ACTION_DESCRIPTIONS),
            "description": "Exactly one high-level army action.",
        },
        "reasoning": {
            "type": "string",
            "description": "One concise tactical explanation for the action.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "How confident the policy is in this action.",
        },
    },
}

SYSTEM_PROMPT = """You are the high-level army commander for a Protoss StarCraft II bot.
Economy, production, workers, and buildings are handled by deterministic code.
Choose exactly one available army action from the supplied list.
Prefer robust, boring choices over risky moves when the state is ambiguous.
Return only JSON matching the requested schema.
Keep reasoning to one short tactical explanation, not hidden chain-of-thought."""


class LLMDecisionError(RuntimeError):
    """Raised when an LLM decision cannot be obtained or parsed."""


@dataclass(frozen=True)
class LLMPolicyConfig:
    """Runtime configuration for the LLM army policy."""

    provider: str = DEFAULT_LLM_PROVIDER
    model: str = DEFAULT_LLM_MODEL
    base_url: str = DEFAULT_LLM_BASE_URL
    api_key_env: str = DEFAULT_LLM_API_KEY_ENV
    timeout_seconds: float = 2.5
    decision_interval: int = 64
    temperature: float = 0.2
    max_output_tokens: int = 180
    require_api_key: bool = True
    log_path: Path | None = None

    @classmethod
    def from_env(
        cls,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout_seconds: float | None = None,
        decision_interval: int | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        require_api_key: bool | None = None,
        log_path: Path | None = None,
    ) -> "LLMPolicyConfig":
        """Build config from CLI overrides and SC2_LLM_* environment values."""
        resolved_provider = (
            provider
            or os.environ.get("SC2_LLM_PROVIDER")
            or DEFAULT_LLM_PROVIDER
        )
        resolved_model = (
            model
            or os.environ.get("SC2_LLM_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or DEFAULT_LLM_MODEL
        )
        resolved_base_url = (
            base_url
            or os.environ.get("SC2_LLM_BASE_URL")
            or DEFAULT_LLM_BASE_URL
        )
        resolved_api_key_env = (
            api_key_env
            or os.environ.get("SC2_LLM_API_KEY_ENV")
            or DEFAULT_LLM_API_KEY_ENV
        )
        resolved_require_api_key = (
            _env_bool("SC2_LLM_REQUIRE_API_KEY", True)
            if require_api_key is None
            else bool(require_api_key)
        )
        return cls(
            provider=resolved_provider,
            model=resolved_model,
            base_url=resolved_base_url,
            api_key_env=resolved_api_key_env,
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else _env_float("SC2_LLM_TIMEOUT", 2.5)
            ),
            decision_interval=(
                decision_interval
                if decision_interval is not None
                else _env_int("SC2_LLM_DECISION_INTERVAL", 64)
            ),
            temperature=(
                temperature
                if temperature is not None
                else _env_float("SC2_LLM_TEMPERATURE", 0.2)
            ),
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else _env_int("SC2_LLM_MAX_OUTPUT_TOKENS", 180)
            ),
            require_api_key=resolved_require_api_key,
            log_path=log_path,
        )

    def validate(self) -> None:
        """Raise ValueError for unsupported configuration."""
        if self.provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ValueError(f"Unsupported LLM provider {self.provider!r}: {supported}")
        if self.decision_interval < 1:
            raise ValueError("decision_interval must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")


@dataclass(frozen=True)
class LLMDecision:
    """One normalized high-level decision from an LLM or fallback source."""

    action: ArmyAction
    reasoning: str
    confidence: float = 0.0
    source: str = "llm"


class LLMDecisionClient(Protocol):
    """Client interface used by LLMArmyPolicy and unit tests."""

    def request_decision(
        self,
        *,
        observation: dict[str, float],
        previous_action: ArmyAction,
        recent_decisions: list[dict[str, Any]],
    ) -> LLMDecision:
        """Return one normalized high-level army decision."""


class OpenAICompatibleDecisionClient:
    """Small stdlib HTTP client for OpenAI Responses or Chat-compatible APIs."""

    def __init__(self, config: LLMPolicyConfig) -> None:
        config.validate()
        self.config = config

    def request_decision(
        self,
        *,
        observation: dict[str, float],
        previous_action: ArmyAction,
        recent_decisions: list[dict[str, Any]],
    ) -> LLMDecision:
        """Ask the configured model for one high-level army decision."""
        payload = build_llm_payload(
            observation=observation,
            previous_action=previous_action,
            recent_decisions=recent_decisions,
        )
        if self.config.provider == "openai-chat":
            response = self._post_json("/chat/completions", self._chat_body(payload))
            text = extract_chat_text(response)
        else:
            response = self._post_json("/responses", self._responses_body(payload))
            text = extract_responses_text(response)
        decision = parse_llm_decision(text)
        return LLMDecision(
            action=decision.action,
            reasoning=decision.reasoning,
            confidence=decision.confidence,
            source="llm",
        )

    def _responses_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "instructions": SYSTEM_PROMPT,
            "input": json.dumps(payload, ensure_ascii=False),
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "sc2_army_decision",
                    "schema": DECISION_SCHEMA,
                    "strict": True,
                }
            },
        }

    def _chat_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "sc2_army_decision",
                    "schema": DECISION_SCHEMA,
                    "strict": True,
                },
            },
        }

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        api_key = resolve_api_key(self.config)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif self.config.require_api_key:
            raise LLMDecisionError(
                f"{self.config.api_key_env} is not set; cannot call LLM API"
            )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMDecisionError(
                f"LLM API HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMDecisionError(f"LLM API request failed: {exc}") from exc

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMDecisionError("LLM API returned non-JSON response") from exc


class LLMArmyPolicy:
    """Use an LLM for low-frequency high-level army choices."""

    def __init__(
        self,
        config: LLMPolicyConfig | None = None,
        *,
        client: LLMDecisionClient | None = None,
        fallback_policy: RuleArmyPolicy | None = None,
    ) -> None:
        self.config = config or LLMPolicyConfig.from_env()
        self.config.validate()
        self.executor = RuleArmyPolicy()
        self.fallback_policy = fallback_policy or RuleArmyPolicy()
        self.client = client if client is not None else self._default_client()
        self._cached_action = ArmyAction.RALLY
        self._last_decision_iteration = -self.config.decision_interval
        self._last_logged_iteration = -self.config.decision_interval
        self._last_logged_key: tuple[str, ArmyAction, str] | None = None
        self._last_console_key: tuple[str, ArmyAction, str] | None = None
        self._recent_decisions: list[dict[str, Any]] = []
        self._warned_no_client = False

    def manage_army(self, bot: Any) -> ArmyAction:
        """Choose an action, execute it through the rule executor, and return it."""
        army = bot.units.of_type(COMBAT_UNIT_TYPES)
        action = self.decide(bot, army)
        self._apply_action_state(bot, action)
        self.executor.execute(bot, action, army)
        return action

    def decide(self, bot: Any, army: Any | None = None) -> ArmyAction:
        """Return one high-level army action for the current bot state."""
        army = army if army is not None else bot.units.of_type(COMBAT_UNIT_TYPES)
        iteration = _bot_iteration(bot)

        guard_decision = self._local_guard_decision(bot, army)
        if guard_decision is not None:
            return self._remember_decision(bot, iteration, guard_decision)

        if self.client is None:
            if not self._warned_no_client:
                print(
                    "[LLMArmyPolicy] LLM API is not configured; "
                    "using rule fallback."
                )
                self._warned_no_client = True
            return self._fallback_decision(bot, army, iteration, "llm-unavailable")

        if iteration - self._last_decision_iteration < self.config.decision_interval:
            return self._cached_action

        observation = build_observation(bot).to_dict()
        previous_action = _previous_action(bot, self._cached_action)
        try:
            decision = self.client.request_decision(
                observation=observation,
                previous_action=previous_action,
                recent_decisions=self._recent_decisions[-5:],
            )
        except Exception as exc:
            return self._fallback_decision(
                bot,
                army,
                iteration,
                f"llm-error: {exc}",
            )

        return self._remember_decision(bot, iteration, decision, observation)

    def _default_client(self) -> LLMDecisionClient | None:
        if self.config.require_api_key and not resolve_api_key(self.config):
            return None
        return OpenAICompatibleDecisionClient(self.config)

    def _local_guard_decision(self, bot: Any, army: Any) -> LLMDecision | None:
        army_count = int(getattr(army, "amount", 0))
        if army_count <= 0:
            bot.is_attacking = False
            return LLMDecision(
                ArmyAction.HOLD,
                "No army units are available, so holding avoids useless orders.",
                1.0,
                "local-guard",
            )
        if base_under_threat(bot):
            return LLMDecision(
                ArmyAction.DEFEND_BASE,
                "Known enemy units are threatening home, so defense takes priority.",
                1.0,
                "local-guard",
            )
        if bool(getattr(bot, "is_attacking", False)) and _should_retreat(bot, army_count):
            return LLMDecision(
                ArmyAction.RETREAT_HOME,
                "The attack has lost too much army strength, so retreating preserves units.",
                1.0,
                "local-guard",
            )
        return None

    def _fallback_decision(
        self,
        bot: Any,
        army: Any,
        iteration: int,
        reason: str,
    ) -> ArmyAction:
        action = self.fallback_policy.decide(bot, int(getattr(army, "amount", 0)))
        decision = LLMDecision(
            action=action,
            reasoning=f"Rule fallback selected {action.name} ({reason}).",
            confidence=0.0,
            source="rule-fallback",
        )
        return self._remember_decision(bot, iteration, decision)

    def _remember_decision(
        self,
        bot: Any,
        iteration: int,
        decision: LLMDecision,
        observation: dict[str, float] | None = None,
    ) -> ArmyAction:
        if decision.source in {"llm", "rule-fallback"}:
            self._cached_action = decision.action
            self._last_decision_iteration = iteration
        info = {
            "iteration": iteration,
            "game_time": float(getattr(bot, "time", 0.0)),
            "action": decision.action.name,
            "reasoning": decision.reasoning,
            "confidence": float(decision.confidence),
            "source": decision.source,
        }
        bot.last_llm_decision = dict(info)
        log_key = (decision.source, decision.action, decision.reasoning)
        should_log = (
            decision.source == "llm"
            or log_key != self._last_logged_key
            or iteration - self._last_logged_iteration >= self.config.decision_interval
        )
        if should_log:
            self._last_logged_iteration = iteration
            self._last_logged_key = log_key
            self._recent_decisions.append(dict(info))
            self._recent_decisions = self._recent_decisions[-20:]
            self._write_log(info, observation)
            self._print_decision(info, decision)
        return decision.action

    def _write_log(
        self,
        info: dict[str, Any],
        observation: dict[str, float] | None,
    ) -> None:
        if self.config.log_path is None:
            return
        row = dict(info)
        if observation is not None:
            row["observation"] = observation
        row["logged_at"] = time.time()
        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _print_decision(self, info: dict[str, Any], decision: LLMDecision) -> None:
        reason = str(info["reasoning"])
        key = (str(info["source"]), decision.action, reason)
        if key == self._last_console_key:
            return
        self._last_console_key = key
        print(
            "[LLMArmyPolicy] "
            f"source={info['source']} action={decision.action.name} "
            f"confidence={float(info['confidence']):.2f} reason={reason}"
        )

    def _apply_action_state(self, bot: Any, action: ArmyAction) -> None:
        if action is ArmyAction.ATTACK_MAIN:
            bot.is_attacking = True
        elif action in {
            ArmyAction.RALLY,
            ArmyAction.RETREAT_HOME,
            ArmyAction.DEFEND_BASE,
        }:
            bot.is_attacking = False


def build_llm_payload(
    *,
    observation: dict[str, float],
    previous_action: ArmyAction,
    recent_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the compact state payload sent to the LLM."""
    return {
        "game": "StarCraft II",
        "race": "Protoss",
        "role": "high-level army policy",
        "observation_schema": "v3 numeric snapshot",
        "observation": observation,
        "previous_action": action_name(previous_action),
        "recent_decisions": recent_decisions[-5:],
        "available_actions": ACTION_DESCRIPTIONS,
        "constraints": [
            "Choose exactly one action from available_actions.",
            "Economy, production, buildings, and worker management are unavailable.",
            "Prefer DEFEND_BASE when base_under_threat is 1.",
            "Prefer RETREAT_HOME after large attack-phase losses.",
            "Prefer RALLY before the army is large enough to attack.",
        ],
    }


def parse_llm_decision(text: str | dict[str, Any]) -> LLMDecision:
    """Parse and validate an LLM JSON decision."""
    payload = text if isinstance(text, dict) else _loads_json_object(text)
    action = _parse_action(payload.get("action"))
    reasoning = str(payload.get("reasoning", "")).strip()
    if not reasoning:
        reasoning = f"Selected {action.name} from the current army state."
    confidence = _clamp_float(payload.get("confidence", 0.0), 0.0, 1.0)
    return LLMDecision(action=action, reasoning=reasoning, confidence=confidence)


def extract_responses_text(response: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI Responses API response."""
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if chunks:
        return "".join(chunks)
    raise LLMDecisionError("Responses API response did not include output text")


def extract_chat_text(response: dict[str, Any]) -> str:
    """Extract assistant text from a Chat Completions-compatible response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMDecisionError("Chat response did not include message content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        return "".join(chunks)
    raise LLMDecisionError("Chat response message content is not text")


def resolve_api_key(config: LLMPolicyConfig) -> str | None:
    """Return the first configured API key from environment variables."""
    names = [config.api_key_env, "SC2_LLM_API_KEY", "OPENAI_API_KEY"]
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        value = os.environ.get(name)
        if value:
            return value
    return None


def _loads_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise LLMDecisionError("LLM response did not contain a JSON object")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMDecisionError("LLM response JSON could not be parsed") from exc
    if not isinstance(payload, dict):
        raise LLMDecisionError("LLM response JSON must be an object")
    return payload


def _parse_action(value: Any) -> ArmyAction:
    if isinstance(value, int):
        try:
            return ArmyAction(value)
        except ValueError as exc:
            raise LLMDecisionError(f"Invalid numeric action: {value!r}") from exc
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized.isdigit():
            return _parse_action(int(normalized))
        try:
            return ArmyAction[normalized]
        except KeyError as exc:
            valid = ", ".join(action.name for action in ArmyAction)
            raise LLMDecisionError(
                f"Invalid action {value!r}; expected one of {valid}"
            ) from exc
    raise LLMDecisionError(f"Invalid action value: {value!r}")


def _clamp_float(value: Any, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lower
    return min(max(number, lower), upper)


def _bot_iteration(bot: Any) -> int:
    try:
        return int(getattr(bot, "_last_iteration", 0))
    except (TypeError, ValueError):
        return 0


def _previous_action(bot: Any, fallback: ArmyAction) -> ArmyAction:
    action = getattr(bot, "last_army_action", fallback)
    if isinstance(action, ArmyAction):
        return action
    try:
        return ArmyAction(action)
    except (TypeError, ValueError):
        return fallback


def _should_retreat(bot: Any, army_count: int) -> bool:
    threshold = int(getattr(bot, "ARMY_RETREAT_THRESHOLD", 5))
    if army_count <= threshold:
        return True
    memory = getattr(bot, "army_memory", None)
    peak = int(getattr(memory, "attack_army_peak", 0))
    lost = int(getattr(memory, "army_lost_from_peak", max(peak - army_count, 0)))
    ratio = float(
        getattr(memory, "army_lost_from_peak_ratio", (lost / peak) if peak else 0.0)
    )
    min_peak = int(getattr(bot, "RETREAT_MIN_PEAK_ARMY", 8))
    min_lost = int(getattr(bot, "RETREAT_MIN_LOST_FROM_PEAK", 3))
    min_ratio = float(getattr(bot, "RETREAT_PEAK_LOSS_RATIO", 0.25))
    return peak >= min_peak and lost >= min_lost and ratio >= min_ratio


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
