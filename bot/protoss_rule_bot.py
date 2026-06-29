"""
神族 (Protoss) 规则 Bot - 第一版

策略: 4 Gate Zealot + Stalker 一波流

整局只做最关键的几件事:
  1. 农民采矿 (sc2 库自动分配)
  2. 补农民到 22
  3. 不卡人口 -> 造水晶塔 (Pylon)
  4. 主基地附近开两个气矿 (Assimilator)
  5. 造模拟核心 (Cybernetics Core) 解锁追猎者
  6. 4 个兵营 (Gateway) 持续出兵
  7. 兵力 >= 15 集结向敌方主基地一波

这一版的设计目标:
  - 跑通管道 (能进游戏 / 能采矿 / 能造兵 / 能 A 过去)
  - 代码简单, 易读, 便于后续把决策替换为 RL
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from sc2.bot_ai import BotAI
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

from bot.managers.army_policy import ArmyAction, ArmyPolicy
from bot.managers.army_memory import ArmyMemory
from bot.managers.rule_army_policy import COMBAT_UNIT_TYPES, RuleArmyPolicy
from bot.managers.rule_strategy_policy import RuleStrategyPolicy
from bot.managers.strategy_executor import StrategyExecutor
from bot.managers.strategy_policy import StrategyPolicy
from bot.managers.surrender_policy import SurrenderPolicy, maybe_surrender
from rl.actions import action_name, action_to_int
from rl.observations import build_observation
from rl.strategy_actions import (
    StrategyAction,
    strategy_action_name,
    strategy_action_to_int,
)
from rl.strategy_observations import (
    build_strategy_observation,
    build_strategy_observation_details,
)
from rl.trajectory_recorder import StrategyTrajectoryStep, TrajectoryStep


class ProtossRuleBot(BotAI):
    """规则型神族 Bot - 一波狂热者+追猎者推进。"""

    NAME: str = "ProtossRuleBot"
    RACE: Race = Race.Protoss

    # 策略参数 - 集中放这里, 后续 RL 模块要替换的就是这些值
    TARGET_WORKERS: int = 22          # 每个基地理想农民数
    TARGET_GATEWAYS: int = 4          # 兵营目标数量
    ARMY_ATTACK_THRESHOLD: int = 15   # 累计这么多兵就发动进攻
    ARMY_RETREAT_THRESHOLD: int = 5   # 兵力低于这个数撤回
    RETREAT_PEAK_LOSS_RATIO: float = 0.25
    RETREAT_MIN_PEAK_ARMY: int = 8
    RETREAT_MIN_LOST_FROM_PEAK: int = 3
    STRATEGY_DECISION_INTERVAL: int = 64
    GG_MIN_GAME_TIME: float = 360.0
    GG_MAX_WORKERS_WITHOUT_BASE: int = 4
    GG_MAX_ARMY: int = 2
    GG_MAX_PRODUCTION_STRUCTURES: int = 1
    GG_MIN_ENEMY_UNITS_SEEN: int = 8

    def __init__(
        self,
        *,
        trajectory_recorder: Any | None = None,
        strategy_trajectory_recorder: Any | None = None,
        episode_metadata: dict[str, str] | None = None,
        record_decision_interval: int = 8,
    ) -> None:
        super().__init__()
        self.is_attacking: bool = False
        self.army_policy: ArmyPolicy = RuleArmyPolicy()
        self.strategy_policy: StrategyPolicy = RuleStrategyPolicy()
        self.strategy_executor = StrategyExecutor()
        self.trajectory_recorder = trajectory_recorder
        self.strategy_trajectory_recorder = strategy_trajectory_recorder
        self.episode_metadata = episode_metadata or {}
        self.episode_id = self.episode_metadata.get("episode_id", uuid4().hex)
        self.record_decision_interval = max(1, record_decision_interval)
        self.last_army_action: ArmyAction = ArmyAction.HOLD
        self.last_strategy_action: StrategyAction = StrategyAction.STAY_COURSE
        self.last_strategy_decision_source: str | None = None
        self.last_strategy_decision_reason: str | None = None
        self.last_strategy_execution_result = None
        self.army_memory = ArmyMemory()
        self.surrender_policy = SurrenderPolicy(
            min_game_time=self.GG_MIN_GAME_TIME,
            max_workers_without_base=self.GG_MAX_WORKERS_WITHOUT_BASE,
            max_army=self.GG_MAX_ARMY,
            max_production_structures=self.GG_MAX_PRODUCTION_STRUCTURES,
            min_enemy_units_seen=self.GG_MIN_ENEMY_UNITS_SEEN,
        )
        self._gg_surrendered = False
        self._last_iteration: int = 0

    async def on_start(self) -> None:
        """游戏开始时被调用一次。"""
        # 仅打印, 不做奇怪的事
        print(f"[{self.NAME}] Game started. Race=Protoss")

    async def on_step(self, iteration: int) -> None:
        """每个游戏步被调用一次。这里是所有决策的入口。"""
        self._last_iteration = iteration
        if await maybe_surrender(self):
            return

        # 1. 经济: 分配空闲农民
        await self.distribute_workers()

        # 2. 训练农民
        await self._train_workers()

        # 3. 补水晶塔避免卡人口
        await self._build_pylons()

        # 4. 造气矿
        await self._build_assimilators()

        # 5. 造模拟核心
        await self._build_cybernetics_core()

        # 6. 造兵营
        await self._build_gateways()

        # 7. 出兵
        await self._train_army()

        # 8. 低频宏观策略插槽，默认 no-op，供后续 strategy policy 接管
        await self._manage_strategy(iteration)

        # 9. 进攻/撤退判断 (这里就是后续 RL 接管的位置)
        await self._manage_army(iteration)

    # ----------------------------------------------------------------
    # 各个子模块
    # ----------------------------------------------------------------

    async def _train_workers(self) -> None:
        """每个星灵枢纽 (Nexus) 持续训练农民, 直到达到上限。"""
        for nexus in self.townhalls.ready.idle:
            if self.workers.amount >= self.TARGET_WORKERS * self.townhalls.amount:
                return
            if self.can_afford(UnitTypeId.PROBE) and self.supply_left > 0:
                nexus.train(UnitTypeId.PROBE)

    async def _build_pylons(self) -> None:
        """人口快卡死时造水晶塔。"""
        if self.supply_left < 5 and self.already_pending(UnitTypeId.PYLON) == 0:
            if not self.townhalls.ready.exists:
                return
            nexus = self.townhalls.ready.first
            if self.can_afford(UnitTypeId.PYLON):
                # 在主基地附近造
                await self.build(
                    UnitTypeId.PYLON,
                    near=nexus.position.towards(self.game_info.map_center, 8),
                )

    async def _build_assimilators(self) -> None:
        """每个星灵枢纽周围造 2 个气矿。"""
        for nexus in self.townhalls.ready:
            vespenes = self.vespene_geyser.closer_than(15, nexus)
            for vespene in vespenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                # 如果这个气矿没建过气矿建筑
                if not self.gas_buildings.closer_than(1.0, vespene).exists:
                    worker.build_gas(vespene)

    async def _build_cybernetics_core(self) -> None:
        """造模拟核心 (解锁追猎者)。需要至少 1 个兵营。"""
        if (
            self.structures(UnitTypeId.GATEWAY).ready.exists
            and not self.structures(UnitTypeId.CYBERNETICSCORE).exists
            and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0
        ):
            if self.can_afford(UnitTypeId.CYBERNETICSCORE):
                pylon = self.structures(UnitTypeId.PYLON).ready
                if pylon.exists:
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon.random)

    async def _build_gateways(self) -> None:
        """造兵营, 上限 TARGET_GATEWAYS。"""
        total_gateways = (
            self.structures(UnitTypeId.GATEWAY).amount
            + self.already_pending(UnitTypeId.GATEWAY)
        )
        if total_gateways >= self.TARGET_GATEWAYS:
            return
        if not self.structures(UnitTypeId.PYLON).ready.exists:
            return
        if self.can_afford(UnitTypeId.GATEWAY):
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            await self.build(UnitTypeId.GATEWAY, near=pylon)

    async def _train_army(self) -> None:
        """每个兵营有空就出兵。优先追猎者, 缺气矿就出狂热者。"""
        has_cyber = self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists

        for gateway in self.structures(UnitTypeId.GATEWAY).ready.idle:
            # 优先追猎者(更强但要气矿)
            if (
                has_cyber
                and self.can_afford(UnitTypeId.STALKER)
                and self.supply_left >= 2
            ):
                gateway.train(UnitTypeId.STALKER)
            elif self.can_afford(UnitTypeId.ZEALOT) and self.supply_left >= 2:
                gateway.train(UnitTypeId.ZEALOT)

    async def _manage_army(self, iteration: int) -> None:
        """Delegate high-level army control to the active policy."""
        was_attacking = self.is_attacking
        self._update_army_memory()
        action = self.army_policy.manage_army(self)
        if action is ArmyAction.ATTACK_MAIN and not was_attacking:
            self.army_memory.start_attack()
        self.last_army_action = action
        self._record_trajectory_step(iteration, action)

    async def _manage_strategy(self, iteration: int) -> None:
        """Run the low-frequency macro strategy layer when enabled."""
        interval = max(1, int(getattr(self, "STRATEGY_DECISION_INTERVAL", 64)))
        if iteration % interval != 0:
            return
        action = self.strategy_policy.decide_strategy(self)
        self.last_strategy_execution_result = await self.strategy_executor.execute(
            self,
            action,
        )
        self._record_strategy_trajectory_step(iteration, action)
        self.last_strategy_action = action

    def _update_army_memory(self) -> None:
        """Refresh army trend state before policy selection."""
        army = self.units.of_type(COMBAT_UNIT_TYPES)
        self.army_memory.update(
            int(getattr(army, "amount", 0)),
            is_attacking=self.is_attacking,
        )

    def _record_trajectory_step(
        self,
        iteration: int,
        action: ArmyAction,
        *,
        reward: float = 0.0,
        done: bool = False,
        result: str | None = None,
    ) -> None:
        """Record one high-level decision when trajectory capture is enabled."""
        if self.trajectory_recorder is None:
            return
        if not done and iteration % self.record_decision_interval != 0:
            return

        try:
            observation = build_observation(self).to_dict()
            step = TrajectoryStep(
                episode_id=self.episode_id,
                step=iteration,
                map_name=self.episode_metadata.get("map_name", ""),
                difficulty=self.episode_metadata.get("difficulty", ""),
                opponent_race=self.episode_metadata.get("opponent_race", ""),
                observation=observation,
                action=action_to_int(action),
                action_name=action_name(action),
                reward=reward,
                done=done,
                result=result,
                opponent_ai_build=self.episode_metadata.get(
                    "opponent_ai_build", "RandomBuild"
                ),
            )
            self.trajectory_recorder.record(step)
        except Exception as exc:
            print(f"[{self.NAME}] Trajectory recording failed: {exc!r}")

    def _record_strategy_trajectory_step(
        self,
        iteration: int,
        action: StrategyAction,
        *,
        reward: float = 0.0,
        done: bool = False,
        result: str | None = None,
    ) -> None:
        """Record one low-frequency strategy decision when enabled."""
        if self.strategy_trajectory_recorder is None:
            return

        try:
            army_action = getattr(self, "last_army_action", None)
            tactic_state = getattr(self, "last_tactic_state", None)
            action_before_filter = getattr(
                self,
                "last_strategy_action_before_tactic_filter",
                None,
            )
            action_after_filter = getattr(
                self,
                "last_strategy_action_after_tactic_filter",
                None,
            )
            execution_result = getattr(self, "last_strategy_execution_result", None)
            step = StrategyTrajectoryStep(
                episode_id=self.episode_id,
                step=iteration,
                map_name=self.episode_metadata.get("map_name", ""),
                difficulty=self.episode_metadata.get("difficulty", ""),
                opponent_race=self.episode_metadata.get("opponent_race", ""),
                strategy_observation=build_strategy_observation(self).to_dict(),
                strategy_action=strategy_action_to_int(action),
                strategy_action_name=strategy_action_name(action),
                strategy_observation_details=(
                    build_strategy_observation_details(self).to_dict()
                ),
                strategy_policy_source=getattr(
                    self,
                    "last_strategy_decision_source",
                    None,
                ),
                strategy_policy_reason=getattr(
                    self,
                    "last_strategy_decision_reason",
                    None,
                ),
                army_observation=build_observation(self).to_dict(),
                army_action=(
                    action_to_int(army_action)
                    if isinstance(army_action, ArmyAction)
                    else None
                ),
                army_action_name=(
                    action_name(army_action)
                    if isinstance(army_action, ArmyAction)
                    else None
                ),
                reward=reward,
                done=done,
                result=result,
                opponent_ai_build=self.episode_metadata.get(
                    "opponent_ai_build", "RandomBuild"
                ),
                tactic_id=_enum_name(getattr(tactic_state, "current_tactic", None)),
                tactic_phase=_enum_name(getattr(tactic_state, "phase", None)),
                tactic_source=getattr(self, "last_tactic_source", None),
                tactic_started_game_time=getattr(
                    tactic_state,
                    "started_game_time",
                    None,
                ),
                tactic_switch_reason=getattr(
                    tactic_state,
                    "last_switch_reason",
                    None,
                ),
                tactic_previous_id=_enum_name(
                    getattr(tactic_state, "previous_tactic", None)
                ),
                strategy_action_before_tactic_filter=(
                    _strategy_action_int_or_none(action_before_filter)
                ),
                strategy_action_before_tactic_filter_name=(
                    _strategy_action_name_or_none(action_before_filter)
                ),
                strategy_action_after_tactic_filter=(
                    _strategy_action_int_or_none(action_after_filter)
                ),
                strategy_action_after_tactic_filter_name=(
                    _strategy_action_name_or_none(action_after_filter)
                ),
                strategy_execution_attempted=(
                    getattr(execution_result, "attempted", None)
                ),
                strategy_execution_effect=getattr(execution_result, "effect", None),
                strategy_execution_blocker=getattr(execution_result, "blocker", None),
                strategy_execution_unit_type=getattr(
                    execution_result,
                    "unit_type",
                    None,
                ),
                strategy_execution_target=getattr(execution_result, "target", None),
            )
            self.strategy_trajectory_recorder.record(step)
        except Exception as exc:
            print(f"[{self.NAME}] Strategy trajectory recording failed: {exc!r}")

    async def on_end(self, game_result) -> None:
        """游戏结束。"""
        print(f"[{self.NAME}] Game ended. Result = {game_result}")
        result_name = getattr(game_result, "name", str(game_result))
        self._record_trajectory_step(
            self._last_iteration + 1,
            self.last_army_action,
            reward=_terminal_reward(result_name),
            done=True,
            result=result_name,
        )
        self._record_strategy_trajectory_step(
            self._last_iteration + 1,
            self.last_strategy_action,
            reward=_terminal_reward(result_name),
            done=True,
            result=result_name,
        )
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.close()
        if self.strategy_trajectory_recorder is not None:
            self.strategy_trajectory_recorder.close()


def _terminal_reward(result_name: str) -> float:
    if "Victory" in result_name:
        return 100.0
    if "Defeat" in result_name:
        return -100.0
    return 0.0


def _enum_name(value: Any | None) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    return str(value)


def _strategy_action_int_or_none(action: Any | None) -> int | None:
    if action is None:
        return None
    return strategy_action_to_int(StrategyAction(action))


def _strategy_action_name_or_none(action: Any | None) -> str | None:
    if action is None:
        return None
    return strategy_action_name(StrategyAction(action))
