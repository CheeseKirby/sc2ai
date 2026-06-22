from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from sc2.ids.unit_typeid import UnitTypeId

from bot.managers.army_memory import ArmyMemory
from bot.managers.army_policy import ArmyAction
from bot.managers.llm_army_policy import (
    LLMArmyPolicy,
    LLMDecision,
    LLMPolicyConfig,
    extract_chat_text,
    extract_responses_text,
    parse_llm_decision,
)


@dataclass(frozen=True)
class FakePosition:
    x: float
    y: float
    name: str = "pos"

    def distance_to(self, other: FakePosition) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def towards(self, other: FakePosition, distance: int) -> FakePosition:
        return FakePosition(
            self.x + distance,
            self.y + distance,
            f"{self.name}->towards({other.name},{distance})",
        )


class FakeUnit:
    def __init__(self) -> None:
        self.commands: list[tuple[str, FakePosition]] = []

    def attack(self, target: FakePosition) -> None:
        self.commands.append(("attack", target))

    def move(self, target: FakePosition) -> None:
        self.commands.append(("move", target))


class FakeUnits:
    def __init__(
        self,
        amount: int,
        center: FakePosition | None = None,
        units: list[FakeUnit] | None = None,
    ) -> None:
        self.amount = amount
        self.center = center or FakePosition(0, 0)
        self._units = units or []

    @property
    def exists(self) -> bool:
        return self.amount > 0

    @property
    def idle(self) -> list[FakeUnit]:
        return self._units


class FakeUnitCollection:
    def __init__(self, units: list[FakeUnit]) -> None:
        self.units = units
        self.counts = {
            UnitTypeId.ZEALOT: len(units),
            UnitTypeId.STALKER: 0,
        }

    def of_type(self, unit_types: set[UnitTypeId] | frozenset[UnitTypeId]) -> FakeUnits:
        amount = sum(self.counts.get(unit_type, 0) for unit_type in unit_types)
        return FakeUnits(amount, FakePosition(6, 8), self.units)


class FakeStructures:
    def __call__(self, unit_type: UnitTypeId) -> FakeUnits:
        if unit_type is UnitTypeId.GATEWAY:
            return FakeUnits(4)
        if unit_type is UnitTypeId.CYBERNETICSCORE:
            return FakeUnits(1)
        return FakeUnits(0)


@dataclass
class FakeTownhall:
    position: FakePosition


class FakeTownhalls:
    amount = 1

    @property
    def exists(self) -> bool:
        return True

    @property
    def first(self) -> FakeTownhall:
        return FakeTownhall(FakePosition(0, 0, "home"))

    def __iter__(self):
        return iter((self.first,))


@dataclass
class FakeGameInfo:
    map_center: FakePosition


class FakeBot:
    time = 100.0
    minerals = 300
    vespene = 100
    supply_used = 40
    supply_cap = 50
    supply_left = 10
    enemy_start_locations = [FakePosition(10, 10, "enemy-main")]
    ARMY_ATTACK_THRESHOLD = 10
    ARMY_RETREAT_THRESHOLD = 3
    RETREAT_PEAK_LOSS_RATIO = 0.25
    RETREAT_MIN_PEAK_ARMY = 8
    RETREAT_MIN_LOST_FROM_PEAK = 3

    def __init__(self, army_count: int = 12) -> None:
        self.is_attacking = False
        self._last_iteration = 8
        self.last_army_action = ArmyAction.HOLD
        self.units_list = [FakeUnit() for _ in range(army_count)]
        self.units = FakeUnitCollection(self.units_list)
        self.workers = FakeUnits(22)
        self.townhalls = FakeTownhalls()
        self.structures = FakeStructures()
        self.enemy_units = FakeUnits(0)
        self.enemy_structures = FakeUnits(1)
        self.game_info = FakeGameInfo(FakePosition(5, 5, "center"))
        self.army_memory = ArmyMemory()
        self.army_memory.update(army_count, is_attacking=False)


class FakeDecisionClient:
    def __init__(self, decision: LLMDecision) -> None:
        self.decision = decision
        self.calls: list[dict[str, Any]] = []

    def request_decision(
        self,
        *,
        observation: dict[str, float],
        previous_action: ArmyAction,
        recent_decisions: list[dict[str, Any]],
    ) -> LLMDecision:
        self.calls.append(
            {
                "observation": observation,
                "previous_action": previous_action,
                "recent_decisions": recent_decisions,
            }
        )
        return self.decision


def _config() -> LLMPolicyConfig:
    return LLMPolicyConfig(
        require_api_key=False,
        decision_interval=16,
        timeout_seconds=0.1,
    )


@pytest.mark.unit
def test_llm_policy_uses_client_decision_and_exposes_reasoning() -> None:
    client = FakeDecisionClient(
        LLMDecision(
            ArmyAction.ATTACK_MAIN,
            "Army count is healthy enough to pressure the enemy main.",
            0.82,
        )
    )
    bot = FakeBot()

    action = LLMArmyPolicy(_config(), client=client).manage_army(bot)

    assert action is ArmyAction.ATTACK_MAIN
    assert bot.is_attacking is True
    assert len(client.calls) == 1
    assert client.calls[0]["observation"]["army_count"] == 12.0
    assert bot.last_llm_decision["source"] == "llm"
    assert "healthy enough" in bot.last_llm_decision["reasoning"]
    assert {command[0] for unit in bot.units_list for command in unit.commands} == {
        "attack"
    }


@pytest.mark.unit
def test_llm_policy_reuses_cached_decision_between_intervals() -> None:
    client = FakeDecisionClient(
        LLMDecision(ArmyAction.ATTACK_MAIN, "Attack timing is still valid.", 0.75)
    )
    policy = LLMArmyPolicy(_config(), client=client)
    bot = FakeBot()

    assert policy.manage_army(bot) is ArmyAction.ATTACK_MAIN
    bot._last_iteration = 9
    assert policy.manage_army(bot) is ArmyAction.ATTACK_MAIN

    assert len(client.calls) == 1


@pytest.mark.unit
def test_llm_policy_falls_back_to_rule_when_api_key_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SC2_LLM_API_KEY", raising=False)
    config = LLMPolicyConfig(api_key_env="MISSING_SC2_KEY", decision_interval=16)
    bot = FakeBot()

    action = LLMArmyPolicy(config).manage_army(bot)

    assert action is ArmyAction.ATTACK_MAIN
    assert bot.last_llm_decision["source"] == "rule-fallback"


@pytest.mark.unit
def test_parse_llm_decision_accepts_json_fences_and_clamps_confidence() -> None:
    decision = parse_llm_decision(
        '```json\n{"action":"retreat_home","reasoning":"Too much army was lost.","confidence":2}\n```'
    )

    assert decision.action is ArmyAction.RETREAT_HOME
    assert decision.reasoning == "Too much army was lost."
    assert decision.confidence == 1.0


@pytest.mark.unit
def test_extract_openai_response_text_helpers() -> None:
    response_text = '{"action":"RALLY","reasoning":"Build up first.","confidence":0.6}'

    assert (
        extract_responses_text(
            {"output": [{"content": [{"type": "output_text", "text": response_text}]}]}
        )
        == response_text
    )
    assert (
        extract_chat_text({"choices": [{"message": {"content": response_text}}]})
        == response_text
    )
