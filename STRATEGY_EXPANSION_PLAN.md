# Strategy Expansion Plan

面向后续 Codex / coding agent 的策略拓展开发计划。

## 当前边界

当前项目已经具备：

- rule-based Protoss bot
- `ArmyPolicy` 抽象
- 5-action army action space
- schema v3 observation
- trajectory recording / diagnostics
- imitation training
- checkpoint-backed online inference
- experimental LLM army policy

策略拓展开发边界：

- 不要实现 PPO。
- 不要改 PPO / RL 主线架构。
- 不要改变当前 `ArmyAction` v1 语义。
- 不要破坏 rule baseline、coverage-teacher、RL checkpoint policy 或 LLM policy。
- 不要修改当前 schema v3 的默认运行路径。

## 目标

当前 5 个军队高层动作适合验证训练闭环，但策略上限较低。后续策略拓展的目标是让 bot 逐步具备：

- 扩张
- 科技选择
- 兵种组合调整
- 升级
- 静态防守
- 更清晰的宏观战略意图

核心思路：

```text
先让 rule executor 会执行更多战略意图
再让 teacher 生成可解释标签
再训练 imitation strategy policy
最后才考虑 PPO
```

不要让模型直接控制：

- 建筑坐标
- 单位选择
- 每帧移动
- 技能释放细节
- 目标点微操

模型应输出高层意图，执行细节继续由规则代码处理。

## 总体架构

建议从单层 army policy 扩展为两层：

```text
ProtossRuleBot
  -> macro/economy/build/production rule executor
  -> StrategyPolicy    低频宏观策略
  -> ArmyPolicy        高频军队姿态
```

推荐频率：

```text
StrategyPolicy: 每 64 / 128 iteration 决策一次
ArmyPolicy:     每 8 / 16 iteration 决策一次
```

两层职责：

```text
StrategyPolicy:
  决定开矿、转科技、补建筑、升级、防守建设、爆兵优先级。

ArmyPolicy:
  决定 RALLY / ATTACK_MAIN / RETREAT_HOME / DEFEND_BASE / HOLD。
```

不要把宏观动作和军队动作一开始混成一个巨大 action space。

## 阶段 0：保护现有兼容性

目标：在扩展策略能力前，保护现有训练、评测和 checkpoint 兼容路径。

要求：

- 不改 `ArmyAction` 当前 5 动作。
- 不改 schema v3 默认字段和含义。
- 不改当前 checkpoint 加载兼容逻辑。
- 新能力必须使用新枚举、新接口、新 schema 版本或显式启用参数。

验收：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

现有 rule / coverage-teacher / imitation v3 eval 命令仍可运行。

## 阶段 1：新增 StrategyPolicy 接口

新增低频宏观策略层，不替代 `ArmyPolicy`。

建议新增文件：

```text
bot/managers/strategy_policy.py
bot/managers/rule_strategy_policy.py
bot/managers/strategy_executor.py
rl/strategy_actions.py
```

建议第一版 `StrategyAction`：

```text
0 STAY_COURSE
1 EXPAND
2 ADD_GATEWAYS
3 TECH_ROBO
4 FORGE_UPGRADES
5 BUILD_STATIC_DEFENSE
6 PRODUCE_ARMY
7 BOOST_WORKERS
```

暂不加入：

- Stargate
- Twilight
- High Templar
- Disruptor
- Carrier
- Warp Prism micro

原因：第一版优先做稳定中期能力，不碰复杂微操和远科技。

验收：

- `StrategyAction` 有稳定 int/name 映射。
- `StrategyPolicy` 独立于 `ArmyPolicy`。
- 默认 bot 不启用 learned strategy policy。
- 单元测试覆盖 action mapping 和默认 no-op / rule 行为。

## 阶段 2：扩 Rule Executor 能力

目标：bot 先真的会执行宏观意图。

建议新增或拆分执行函数：

```text
_expand()
_add_gateways_by_base_count()
_build_robo()
_train_robo_units()
_build_forge()
_research_ground_upgrades()
_build_static_defense()
_boost_worker_production()
```

第一批支持：

```text
Nexus 二矿
Robotics Facility
Immortal
Observer
Forge
Ground Weapons / Ground Armor
Shield Battery
Photon Cannon
```

执行原则：

```text
StrategyAction = 意图
Executor = 处理资源、前置、建筑位置、等待、fallback
```

示例：

```text
TECH_ROBO:
  如果没有 Cybernetics Core，先走 Cyber 前置。
  如果有 Cyber 且无 Robo，造 Robotics Facility。
  如果 Robo ready，训练 Immortal / Observer。
  资源不足则等待，不 crash。
```

验收：

- `EXPAND` 能尝试开二矿。
- `TECH_ROBO` 能按前置条件逐步执行。
- `FORGE_UPGRADES` 能造 Forge 并升级。
- `BUILD_STATIC_DEFENSE` 不会在无 Pylon / 资源不足时崩溃。
- 默认 rule baseline 行为不变。

## 阶段 3：定义 Strategy Observation Schema

不要覆盖当前 schema v3 默认运行路径。新增 schema v4 或 strategy-specific observation。

建议字段：

```text
own_bases
ready_gateways
ready_robo
ready_forge
ready_static_defense

immortals
observers
sentries

ground_weapon_level
ground_armor_level

enemy_air_units_known
enemy_armored_units_known
enemy_cloaked_units_seen

worker_saturation_ratio
gateway_idle_count
robo_idle_count

base_under_air_threat
base_under_ground_threat
```

原则：

```text
模型要决策什么，就必须看见对应状态。
```

例如：

```text
TECH_ROBO 需要看到：
  Cyber / Robo 状态
  Immortal / Observer 数量
  敌方 armored / cloaked 信号
  gas 是否足够

EXPAND 需要看到：
  当前基地数
  资源
  威胁
  worker saturation
```

验收：

- schema version 明确，例如 `strategy_v1` 或 `v4`。
- 不影响 v3 observation 默认输出。
- diagnostics 能识别新 schema。
- 测试覆盖字段顺序、缺字段校验、向量维度。

## 阶段 4：扩展 Trajectory 记录

目标：同时记录宏观策略和军队动作。

建议记录结构：

```json
{
  "observation": {},
  "strategy_action": 3,
  "strategy_action_name": "TECH_ROBO",
  "army_action": 1,
  "army_action_name": "ATTACK_MAIN",
  "reward": 0.0,
  "done": false
}
```

注意：

- 不要把 `strategy_action` 和 `army_action` 混成一个标签。
- 两类动作频率不同，后续应能分别训练和诊断。
- 旧 army-only trajectory loader 不应被破坏。

验收：

- 旧 trajectory 仍可诊断。
- 新 trajectory 可统计 strategy action coverage。
- terminal rows 正常。
- JSONL 格式保持稳定。

## 阶段 5：实现 StrategyCoverageTeacher

目标：先用规则 teacher 生成可解释的宏观策略数据。

建议新增：

```text
bot/managers/coverage_strategy_policy.py
```

规则草案：

```text
资源多 + 只有 1 基地 + 无严重威胁:
  EXPAND

资源多 + gateway 少:
  ADD_GATEWAYS

有 Cyber + 无 Robo + 敌方 armored / 中期压力:
  TECH_ROBO

Forge 缺失 + 游戏进入中期:
  FORGE_UPGRADES

基地受威胁或敌军接近:
  BUILD_STATIC_DEFENSE

农民不足或饱和度低:
  BOOST_WORKERS

其他:
  STAY_COURSE 或 PRODUCE_ARMY
```

验收：

- strategy action coverage 不低于 70%。
- `EXPAND`、`TECH_ROBO`、`FORGE_UPGRADES`、`BUILD_STATIC_DEFENSE` 都有样本。
- 没有单一动作严重塌缩。
- 标签能通过 observation feature stats 解释。
- coverage teacher 只用于采数据，不当强 baseline。

## 阶段 6：训练 Strategy Imitation Candidate

建议候选名：

```text
imitation_v4_strategy_candidate
```

流程：

```text
采 strategy trajectories
诊断 strategy action coverage
训练 strategy imitation
guarded eval
对比 rule_strategy / coverage_strategy / imitation_strategy
```

不要直接 PPO。

验收门槛：

- schema 全为 strategy schema / v4。
- `rows_defaulted_observation_fields=0`。
- 关键 strategy action 都有训练和验证样本。
- 不只看总 accuracy。
- per-action accuracy 包含：
  - `EXPAND`
  - `TECH_ROBO`
  - `FORGE_UPGRADES`
  - `BUILD_STATIC_DEFENSE`
- online eval 不 crash、不 timeout、不 action collapse。

## 阶段 7：组合 StrategyPolicy + ArmyPolicy

目标：宏观策略低频，军队动作高频。

建议执行顺序：

```text
1. 更新经济 / 生产 / 建筑规则
2. StrategyPolicy 输出宏观意图
3. StrategyExecutor 更新 build / production priorities
4. ArmyPolicy 输出军队姿态
5. Rule executor 执行实际单位命令
```

紧急情况优先级：

```text
DEFEND_BASE / RETREAT_HOME 不应被宏观策略阻塞。
基地威胁和军队撤退仍应有高优先级。
```

验收：

- 宏观策略不会阻塞军队防守。
- 开矿、科技、升级不会导致经济卡死。
- 轨迹能同时记录 strategy 和 army 决策。
- online eval 可稳定完成。

## 阶段 8：LLM 作为解释 / 顾问层

LLM 适合做：

- 战略解释
- 战局总结
- 低频 advisory
- 赛后 trajectory 分析
- 辅助改 teacher 规则

第一版不建议让 LLM 在线覆盖实际策略动作。

验收：

- LLM 超时不影响 bot。
- LLM 输出只作为 metadata 或 advisory。
- 实际动作仍来自可验证 policy / rule executor。

## 阶段 9：再考虑 PPO

只有满足以下条件后再进入 PPO：

- v3 army imitation baseline 稳定。
- strategy imitation baseline 稳定。
- rule / coverage / imitation 三方 eval 有清楚对比。
- 没有严重 action collapse。
- reward breakdown 已设计清楚。
- reset / step 环境边界清楚。

PPO 第一版仍只做高层动作，不做单位级微操。

## 不建议近期做的事

- 不要直接把动作空间扩成几十上百个。
- 不要让模型选择建筑坐标。
- 不要让模型直接控制单位技能和微操。
- 不要破坏 v3 checkpoint 兼容性。
- 不要为了增加样本刷脏标签，例如撤退后 sticky 生成 observation 不一致的 `RETREAT_HOME`。

## 推荐近期路线

建议按这个顺序推进：

```text
1. 新增 StrategyPolicy 接口
2. 新增 StrategyAction v1
3. 扩 Rule Executor：Expand + Robo + Forge + Static Defense
4. 定义 strategy schema / v4
5. 增加 strategy trajectory 记录和 diagnostics
6. 实现 StrategyCoverageTeacher
7. 采集 strategy 数据
8. 训练 imitation_v4_strategy_candidate
9. guarded eval
10. 再讨论 PPO
```

## 给后续 Codex 的提醒

进入该计划前，先确认 `CODEX.md` 和 `STATE.md` 中当前项目状态，然后从阶段 1 开始小步推进：

```text
每个阶段补 tests 和 diagnostics。
保持默认 rule baseline 可运行。
保持旧 trajectory / checkpoint 路径兼容。
```
