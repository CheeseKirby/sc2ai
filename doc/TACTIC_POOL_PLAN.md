# AIBuild 与战术池计划

面向后续 Codex / coding agent 的开发计划。官方内置 AI 的
`AIBuild` 维度已经接入评测和采集，让数据能按对手战术类型拆分；下一步是设计
项目自己的 `TacticSpec` 战术池，把宏观策略动作组织成更连贯的战术计划。

> 当前状态：本文件是 AIBuild / TacticSpec 背景计划。当前执行计划以
> `doc\STRATEGY_OUTCOME_PLAN.md` 为准。

## 文档定位

- `doc\CODEX.md`：当前事实、约束、下一步简表。
- `doc\README.md`：用户视角的项目说明和常用命令。
- `doc\STATE.md`：实验账本和详细历史记录。
- `doc\STRATEGY_OUTCOME_PLAN.md`：当前下一步执行方案，先做 strategy action 的落地结果诊断。
- `doc\TACTIC_POOL_PLAN.md`：AIBuild / TacticSpec 背景计划。
- `doc\archive\STRATEGY_EXPANSION_PLAN.md`：低频 strategy layer 的历史/背景路线。

## 当前判断

当前 strategy layer 已能输出和执行 8 个宏观动作：

```text
STAY_COURSE
EXPAND
ADD_GATEWAYS
TECH_ROBO
FORGE_UPGRADES
BUILD_STATIC_DEFENSE
PRODUCE_ARMY
BOOST_WORKERS
```

但这些动作仍偏“单步宏观意图”，缺少一个稳定的战术上下文。Hard Terran
focused compare 显示，rule/no-op 在该场景更强，而 strategy policy 可能因为
扩张、科技、静态防守或补产能动作干扰了 rule baseline 的自然节奏。

因此下一步不应直接 PPO，也不应继续盲目重训 strategy imitation。更合适的路线已经收敛为：

```text
1. 用已接入的 AIBuild 和现有 Power A/B 数据定位失败类型。
2. StrategyOutcomeDiagnostics 已实现，首批 Power A/B outcome 诊断已生成。
3. 下一步把 TacticSpec filter 改成 guardrail-first，而不是大面积 rewrite。
4. 只有离线 outcome 和 guarded A/B 都稳定后，才考虑 tactic-aware 数据采集和训练。
```

## 非目标

- 不在缺少 env / reward / baseline 对照时直接实现或启动 PPO。
- 不无证据改 PPO / RL 主线架构。
- 不破坏默认 rule baseline。
- 不把 `AIBuild` 当成我方策略动作。
- 不把官方 AI build 当成最优 teacher。
- 不让模型直接控制建筑坐标、单位选择、技能释放或逐帧微操。
- 不把 tactic layer 默认启用；新战术池必须保持显式 opt-in 或规则 no-op 兼容。

训练和数据采集不是非目标本身。满足 guarded comparison、outcome diagnostics、
fresh dirs、可回滚和默认 baseline 不变这些门槛后，可以采小规模 tactic-aware
数据，并优先训练 action-outcome / veto 模型，再考虑 tactic-aware imitation。

## 官方 AIBuild 参考

本地 `burnysc2` 暴露的官方 build 枚举：

```text
RandomBuild
Rush
Timing
Power
Macro
Air
```

这些 build 应作为“对手战术类型”和“数据分桶维度”，不是直接照抄成我方策略。

建议映射：

| 官方 AIBuild | 数据用途 | 我方战术池启发 |
|---|---|---|
| `Rush` | 压力、防守、反 rush 数据 | `ANTI_RUSH_DEFENSE`, `GATEWAY_PRESSURE` |
| `Timing` | 中期 timing、防守反打数据 | `ROBO_TIMING`, `GATEWAY_PRESSURE` |
| `Power` | 科技/升级爆发窗口数据 | `TECH_POWER`, `ROBO_TIMING` |
| `Macro` | 经济、扩张、压制窗口数据 | `SAFE_MACRO`, `TECH_POWER` |
| `Air` | 反空、侦测、科技转向数据 | `ANTI_AIR_RESPONSE`, `ROBO_TIMING` |
| `RandomBuild` | 默认兼容和随机对照 | `SAFE_MACRO`, scout-first flexible plan |

## 阶段 1：接入 AIBuild 评测维度（已完成）

### 目标

让 `scripts/evaluate.py` 可以按官方 `AIBuild` 批量跑评测/采集，并把 build 信息写入：

- `eval.jsonl`
- `summary.json`
- army trajectory row
- strategy trajectory row
- experiment config
- 文件名或至少 metadata，避免不同战术对手的数据混在一起无法诊断

当前实现状态：

```text
run.py --ai-build RandomBuild|Rush|Timing|Power|Macro|Air，默认 RandomBuild
scripts/evaluate.py --ai-builds RandomBuild Rush Timing Power Macro Air，默认 RandomBuild
EvalRecord / eval.jsonl / summary.json / experiment config 记录 opponent_ai_build
army trajectory / strategy trajectory 记录 opponent_ai_build metadata
new evaluate trajectory filenames include opponent_ai_build
```

### 建议用户参数

`run.py` 新增：

```text
--ai-build RandomBuild|Rush|Timing|Power|Macro|Air
```

`scripts/evaluate.py` 新增：

```text
--ai-builds RandomBuild Rush Timing Power Macro Air
```

默认值保持兼容：

```text
run.py: --ai-build RandomBuild
evaluate.py: --ai-builds RandomBuild
```

这样旧命令不需要改，也不会改变默认 rule baseline 行为。

### 需要修改的文件

| 文件 | 动作 | 要点 |
|---|---|---|
| `run.py` | UPDATE | import `AIBuild`，新增 `AIBUILD_MAP` 和 `--ai-build`，`Computer(..., ai_build=...)` |
| `scripts/evaluate.py` | UPDATE | `EvalRecord` 增加 `opponent_ai_build`，新增循环维度和命令转发 |
| `scripts/summarize_eval.py` | UPDATE | summary 分组增加 `opponent_ai_build`，表格显示 build |
| `rl/trajectory_recorder.py` | UPDATE | `TrajectoryStep` / `StrategyTrajectoryStep` 增加 `opponent_ai_build` |
| `bot/protoss_rule_bot.py` | UPDATE | `episode_metadata` 写入 trajectory row |
| `tests/test_evaluate.py` | UPDATE | 覆盖 `--ai-build` 转发、config、record 字段 |
| `tests/test_summarize_eval.py` | UPDATE | 覆盖按 build 分组 |
| `tests/test_strategy_trajectory_recording.py` | UPDATE | 覆盖 strategy trajectory 记录 build metadata |

### 实现细节

`run.py` 建议新增：

```python
from sc2.data import AIBuild, Difficulty, Race

AIBUILD_MAP: dict[str, AIBuild] = {
    "RandomBuild": AIBuild.RandomBuild,
    "Rush": AIBuild.Rush,
    "Timing": AIBuild.Timing,
    "Power": AIBuild.Power,
    "Macro": AIBuild.Macro,
    "Air": AIBuild.Air,
}
```

创建对手时：

```python
Computer(
    RACE_MAP[args.opponent],
    DIFFICULTY_MAP[args.difficulty],
    ai_build=AIBUILD_MAP[args.ai_build],
)
```

`episode_metadata` 增加：

```python
"opponent_ai_build": args.ai_build
```

`scripts/evaluate.py` 中 `_trajectory_path()` 建议包含 build 名，便于文件层面诊断：

```text
20260623_150000_AcropolisLE_Hard_Terran_Rush_001.jsonl
```

如果为了兼容不改文件名，也必须保证 row metadata 里有 `opponent_ai_build`。

### 数据兼容

旧 trajectory 没有 `opponent_ai_build`。诊断和 dataset loader 应继续可读：

```text
missing opponent_ai_build -> "Unknown" 或 "RandomBuild"
```

建议诊断中显示：

```text
opponent_ai_build_counts
```

但不要把缺失 build metadata 视为 observation schema defaulting。它是 episode metadata，不是 ML observation 字段。

### 第一阶段验收（2026-06-23 已通过）

本阶段已用单元测试和 1 局 guarded smoke 验证参数转发和 metadata。

建议测试命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_evaluate.py tests\test_summarize_eval.py tests\test_strategy_trajectory_recording.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

本轮实际结果：

```text
focused tests: 26 passed
full pytest: 124 passed
check_env: all OK
guard pid: 21392
smoke run: runs\20260623_144443_eval_aibuild_smoke_v1
eval: runs\20260623_144443_eval_aibuild_smoke_v1\artifacts\eval.jsonl
summary: runs\20260623_144443_eval_aibuild_smoke_v1\artifacts\summary.json
army trajectory: data\trajectories\aibuild_smoke_army_v1
strategy trajectory: data\trajectories\aibuild_smoke_strategy_v1
result: 1 game, return_code=0, Result.Tie
metadata check: eval/summary/army trajectory/strategy trajectory all opponent_ai_build=Rush
```

通过后再跑 guarded smoke：

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"

.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Medium --opponents Terran --ai-builds Rush --games-per-combo 1 --run-root runs --run-name eval_aibuild_smoke_v1 --policy-name rule_aibuild_smoke --army-policy rule --strategy-policy rule --trajectory-dir data\trajectories\aibuild_smoke_army_v1 --strategy-trajectory-dir data\trajectories\aibuild_smoke_strategy_v1 --record-decision-interval 16 --game-time-limit 600
```

验收门槛：

```text
return_code=0
eval.jsonl 有 opponent_ai_build=Rush
summary.json 按 opponent_ai_build 分组
army trajectory 有 opponent_ai_build
strategy trajectory 有 opponent_ai_build
默认不传 --ai-build/--ai-builds 的旧命令仍可运行
```

## 阶段 2：按 AIBuild 采集对手战术矩阵

当前状态：已完成第一组 Hard Terran 小矩阵，用于启动 TacticSpec 设计。

```text
rule/no-op:
  run: runs\20260623_145519_eval_rule_aibuild_hard_terran_v1
  Rush=Victory, Timing=Victory, Power=Defeat, Macro=Defeat, Air=Tie

coverage-teacher:
  run: runs\20260623_150115_eval_coverage_teacher_aibuild_hard_terran_v1
  Rush=Victory, Timing=Victory, Power=Defeat, Macro=Victory, Air=Tie
```

初步结论：

```text
Macro 是第一处清晰差异：coverage-teacher 胜，rule/no-op 败。
Power 是共享失败：coverage-teacher 在该局出现 38 次 TECH_ROBO 和后续 static-defense 响应，提示 tactic filter 需要限制重复 tech/static 行为。
Air 两者都是 Tie，后续可作为 anti-air tactic 的 smoke 场景。
```

### 目标

在不混旧数据目录的前提下，采集按官方 build 分桶的数据，用来回答：

- rule baseline 对不同官方 build 的胜负和动作分布如何？
- coverage-teacher 在不同 build 下是否产生不同 strategy action？
- Hard Terran 退化是否只集中在 `Rush` / `Timing` / `Power` 等特定 build？
- strategy action 是否在某些 build 下过度 `BUILD_STATIC_DEFENSE` 或过早 `TECH_ROBO`？

### 建议第一批数据

先不要全量大采集。建议小矩阵：

```text
map: AcropolisLE
difficulties: Hard
opponents: Protoss Terran Zerg
ai_builds: Rush Timing Power Macro Air
games_per_combo: 1
```

rule baseline：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Protoss Terran Zerg --ai-builds Rush Timing Power Macro Air --games-per-combo 1 --run-root runs --run-name eval_rule_aibuild_matrix_v1 --policy-name rule_aibuild_matrix_v1 --army-policy rule --strategy-policy rule --trajectory-dir data\trajectories\rule_aibuild_matrix_army_v1 --strategy-trajectory-dir data\trajectories\rule_aibuild_matrix_strategy_v1 --record-decision-interval 16 --game-time-limit 900
```

coverage strategy teacher：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Protoss Terran Zerg --ai-builds Rush Timing Power Macro Air --games-per-combo 1 --run-root runs --run-name eval_strategy_coverage_teacher_aibuild_matrix_v1 --policy-name strategy_coverage_teacher_aibuild_matrix_v1 --army-policy rule --strategy-policy coverage-teacher --trajectory-dir data\trajectories\strategy_coverage_teacher_aibuild_matrix_army_v1 --strategy-trajectory-dir data\trajectories\strategy_coverage_teacher_aibuild_matrix_strategy_v1 --record-decision-interval 16 --game-time-limit 900
```

每次启动 SC2 前必须先运行 hidden-window guard。

### 诊断要求

现有诊断基础上建议增加或人工检查：

```text
result_counts by opponent_ai_build
strategy action distribution by opponent_ai_build
threat_action_counts by opponent_ai_build
TECH_ROBO signal timing by opponent_ai_build
pending_repeat_counts by opponent_ai_build
Hard Terran per-build timelines
```

如果暂时不改诊断工具，至少用 `eval.jsonl` 和 strategy trajectory row metadata 做人工/脚本汇总。

### 数据质量门槛

进入 TacticSpec 设计前，至少需要有这些结论：

```text
schema 全 strategy_v2
rows_defaulted_observation_fields=0
return_code 全 0 或失败原因明确
每个 opponent_ai_build 至少有 eval result
coverage-teacher strategy action coverage 未塌缩
Hard Terran 的失败/退化能按 build 类型定位
```

不要只看总胜率。需要看每个 `opponent_race + opponent_ai_build` 的动作和时机。

## 阶段 3：设计 TacticSpec 战术池（雏形已完成）

### 目标

把当前离散 strategy action 组织进更高层的战术计划：

```text
observation + opponent context -> tactic_id
tactic_id + observation -> strategy_action
strategy_action -> StrategyExecutor
```

当前代码：

```text
rl\tactics.py
bot\managers\tactic_selector.py
tests\test_tactics.py
```

当前边界：

```text
TacticSpec / TacticState / RuleTacticSelector 已有。
filter_strategy_action() 已能根据 tactic 约束替换不允许或 pending 重复动作。
已新增显式 opt-in --strategy-tactic-mode rule。
当前只支持 --strategy-policy coverage-teacher --strategy-tactic-mode rule。
默认 --strategy-policy rule 仍保持 no-op。
```

战术池不是硬编码 build order，而是带约束的意图系统。它应该允许切换、撤销和恢复。

### 第一版 TacticId

建议新增：

```text
SAFE_MACRO
ANTI_RUSH_DEFENSE
GATEWAY_PRESSURE
ROBO_TIMING
TECH_POWER
ANTI_AIR_RESPONSE
RECOVERY
```

含义：

| TacticId | 核心意图 | 主要触发 |
|---|---|---|
| `SAFE_MACRO` | 稳健经济、二矿、正常产能 | 默认、对方 Macro/Random、低威胁 |
| `ANTI_RUSH_DEFENSE` | 防 rush，保基地和工人 | 对方 Rush、早期威胁、敌军近家 |
| `GATEWAY_PRESSURE` | 补 gateway，前中期施压 | 对方贪经济、我方兵力达标 |
| `ROBO_TIMING` | Robo 科技，Observer/Immortal，守后反打 | armored/cloaked 信号、中期 Timing/Power |
| `TECH_POWER` | 升级/科技爆发，压制前攒强度 | Forge/upgrade 中期、经济稳定 |
| `ANTI_AIR_RESPONSE` | 反空/侦测/科技转向 | 对方 Air、空军信号 |
| `RECOVERY` | 劣势恢复，少贪经济，优先补工人/兵 | 我方基地/军队/工人受重创 |

### TacticSpec 建议字段

建议新增文件：

```text
rl\tactics.py
bot\managers\tactic_selector.py
bot\managers\tactic_state.py
```

核心结构草案：

```python
@dataclass(frozen=True)
class TacticSpec:
    tactic_id: TacticId
    name: str
    allowed_strategy_actions: tuple[StrategyAction, ...]
    preferred_strategy_actions: tuple[StrategyAction, ...]
    avoid_strategy_actions: tuple[StrategyAction, ...]
    opponent_ai_build_hints: tuple[str, ...]
    min_game_time: float
    max_game_time: float | None
    mineral_reserve: int
    vespene_reserve: int
    max_pending_gateways: int
    max_pending_robo: int
    max_pending_static_defense: int
    attack_army_threshold_bias: int
    expand_allowed_under_threat: bool
    transition_triggers: tuple[str, ...]
    abort_triggers: tuple[str, ...]
```

不要一开始把字段设计成最终完美形态。第一版只需要支持：

- 当前战术 id
- 当前战术阶段
- 允许/偏好/禁止的 strategy action
- 基础资源预留
- pending 建筑防重复
- 切换原因记录

### TacticState 建议字段

```python
@dataclass
class TacticState:
    current_tactic: TacticId
    phase: str
    started_game_time: float
    last_switch_game_time: float
    last_switch_reason: str
    previous_tactic: TacticId | None = None
```

第一版 phase 可以很简单：

```text
OPENING
STABILIZE
POWER_SPIKE
ATTACK_WINDOW
RECOVERY
```

### TacticSelector 第一版

先用规则，不急着训练：

```text
if no base / very low workers / army almost gone:
  RECOVERY
elif opponent_ai_build == Air or enemy_air_units_known > 0:
  ANTI_AIR_RESPONSE
elif early base_under_threat or opponent_ai_build == Rush:
  ANTI_RUSH_DEFENSE
elif enemy_cloaked_units_seen or enemy_armored_units_known:
  ROBO_TIMING
elif opponent_ai_build in Timing/Power and game_time >= midgame:
  ROBO_TIMING or TECH_POWER
elif opponent_ai_build == Macro and threat low:
  GATEWAY_PRESSURE or SAFE_MACRO
else:
  SAFE_MACRO
```

切换要有冷却，避免每次 strategy decision 都抖动：

```text
minimum_tactic_duration: 90 game seconds
emergency tactics can override cooldown: RECOVERY, ANTI_RUSH_DEFENSE
```

### Tactic 与 StrategyAction 的关系

第一版不要新增巨大动作空间。保留现有 `StrategyAction`，让 tactic 约束它：

```text
SAFE_MACRO:
  preferred: BOOST_WORKERS, EXPAND, ADD_GATEWAYS, PRODUCE_ARMY
  avoid: excessive BUILD_STATIC_DEFENSE

ANTI_RUSH_DEFENSE:
  preferred: PRODUCE_ARMY, BUILD_STATIC_DEFENSE, ADD_GATEWAYS
  avoid: EXPAND before stable

GATEWAY_PRESSURE:
  preferred: ADD_GATEWAYS, PRODUCE_ARMY
  avoid: early TECH_ROBO unless armored/cloaked signal

ROBO_TIMING:
  preferred: TECH_ROBO, PRODUCE_ARMY, ADD_GATEWAYS
  avoid: greedy EXPAND under pressure

TECH_POWER:
  preferred: FORGE_UPGRADES, TECH_ROBO, PRODUCE_ARMY
  avoid: static-defense spam

ANTI_AIR_RESPONSE:
  preferred: TECH_ROBO, PRODUCE_ARMY, BUILD_STATIC_DEFENSE
  avoid: blind EXPAND

RECOVERY:
  preferred: BOOST_WORKERS, PRODUCE_ARMY, BUILD_STATIC_DEFENSE
  avoid: EXPAND, FORGE_UPGRADES, non-urgent TECH_ROBO
```

## 阶段 4：Tactic metadata 进入 trajectory

### 目标

让后续 imitation 不只学“当前输出什么 strategy action”，还能学“当前在执行什么战术”。

strategy trajectory 建议新增字段：

```text
opponent_ai_build
tactic_id
tactic_phase
tactic_source
tactic_started_game_time
tactic_switch_reason
strategy_action_before_tactic_filter
strategy_action_after_tactic_filter
```

第一版可以只记录：

```text
tactic_id
tactic_phase
tactic_switch_reason
```

不要把 `tactic_id` 塞进 `strategy_observation` 数值 schema，除非明确要训练 action conditioned on tactic。先作为 metadata 和 label 保存更稳。

## 阶段 5：Tactic-aware StrategyPolicy

### 目标

让 coverage-teacher 或 rule selector 先选择 tactic，再在 tactic 约束下选择 strategy action。

建议结构：

```text
CoverageStrategyPolicy
  -> build strategy observation
  -> TacticSelector selects TacticId
  -> base teacher proposes StrategyAction
  -> TacticSpec filters/adjusts action
  -> StrategyExecutor executes action
```

这一步要非常保守。当前 Hard Terran 退化提示我们，`StrategyExecutor` 的副作用可能拖累 rule baseline。TacticSpec 应优先做保护：

```text
resource reserve before expand/tech/static defense
pending structure repeat caps
minimum army before expand under pressure
static defense cap under threat
TECH_ROBO cooldown after starting Robo
ADD_GATEWAYS cooldown while pending_gateways > 0
```

### 小步实现建议

第一版只在 `coverage-teacher` 的 opt-in mode 下启用 tactic-aware 过滤：

```text
--strategy-policy coverage-teacher
--strategy-tactic-mode rule
```

默认：

```text
--strategy-policy rule
```

仍然必须等价 no-op，不改变 rule baseline。

当前 smoke：

```text
run: runs\20260623_152644_eval_tactic_coverage_aibuild_hard_terran_v1
strategy trajectory: data\trajectories\tactic_coverage_aibuild_hard_terran_strategy_v1
Macro: Victory
Power: Victory
return_code=0 for both
tactic diagnostics: runs\20260623_152644_eval_tactic_coverage_aibuild_hard_terran_v1\artifacts\tactic_diagnostics.json
rows_with_tactic_metadata=88/88
rows_with_filter_metadata=88/88
filter_changes=29 terminal-inclusive
training_filter_changes=28
Power TECH_ROBO -> PRODUCE_ARMY changes: 19 terminal-inclusive, 18 training rows
```

后续 guarded comparison 更新：

```text
Broad tactic filtering hurt non-Power builds:
  runs\20260623_154821_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v1
  Rush=Defeat, Timing=Defeat, Power=Victory, Macro=Defeat, Air=Defeat

Safer spec filtering recovered Timing/Air partly but not enough:
  runs\20260623_160120_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v2
  Rush=Defeat, Timing=Victory, Power=Defeat, Macro=Tie, Air=Tie

Power-targeted filter avoids changing non-Power actions:
  runs\20260623_160924_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v3
  Rush=Victory, Timing=Defeat, Power=Defeat, Macro=Victory, Air=Defeat
  filter_changes=12, all on Power rows

Power-only 3-game recheck:
  no-filter coverage-teacher:
    runs\20260623_161606_eval_coverage_teacher_aibuild_hard_terran_power_recheck_v1
    1W / 1T / 1L
  Power-targeted tactic filter:
    runs\20260623_162021_eval_tactic_power_targeted_hard_terran_power_recheck_v1
    0W / 2T / 1L
```

当前结论：

```text
不要推广当前 tactic-aware filter。
不要采当前 tactic-aware coverage-teacher 数据做训练。
保留 Power-targeted opt-in 作为实验诊断路径，因为它不会改非 Power build
动作，但下一步应该离线细化 TECH_POWER / Power failure modes。
```

当前诊断工具：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_tactics.py <strategy-trajectory-dir> --show-files --json-output <run-artifacts-dir>\tactic_diagnostics.json
```

Power failure 专用离线诊断工具：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_power_tactics.py <strategy-trajectory-dir> [<strategy-trajectory-dir> ...] --show-files --json-output <run-artifacts-dir>\power_tactic_diagnostics.json --text-output <run-artifacts-dir>\power_tactic_diagnostics.txt
```

最新 Power recheck 诊断产物：

```text
runs\20260623_power_tactic_diagnostics_v1\artifacts\power_tactic_diagnostics.txt
runs\20260623_power_tactic_diagnostics_v1\artifacts\power_tactic_diagnostics.json
```

诊断输入：

```text
data\trajectories\coverage_teacher_aibuild_hard_terran_power_recheck_strategy_v1
data\trajectories\tactic_power_targeted_hard_terran_power_recheck_strategy_v1
```

关键离线结论：

```text
files=6
training_rows=431
results=1 Victory / 3 Tie / 2 Defeat
all training rows opponent_ai_build=Power

Power-targeted filter changes:
  TECH_POWER TECH_ROBO -> PRODUCE_ARMY: 15
  TECH_POWER ADD_GATEWAYS -> FORGE_UPGRADES: 11
  TECH_POWER BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 8

counterfactual delta:
  TECH_ROBO -14
  ADD_GATEWAYS -16
  BUILD_STATIC_DEFENSE -12
  PRODUCE_ARMY +26
  FORGE_UPGRADES +11
```

Interpretation:

```text
The current Power filter suppresses static-defense spam, but also suppresses too
many timely TECH_ROBO and ADD_GATEWAYS labels. No-filter coverage-teacher starts
TECH_ROBO around 240.0-251.4s in the Power recheck, while one tactic-filter Tie
delays actual TECH_ROBO to 628.6s. The filter still underperforms no-filter in
the 3-game recheck, so it must remain diagnostic-only.
```

Revised `TECH_POWER` offline spec adjustment:

```text
Implemented after the Power diagnostics, without launching SC2.

Focused tests:
  .\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
  19 passed

Full tests:
  .\.venv\Scripts\python.exe -m pytest -q
  150 passed

Spec changes:
  TECH_POWER fallback order is now:
    TECH_ROBO -> PRODUCE_ARMY -> ADD_GATEWAYS -> FORGE_UPGRADES

  First Robo is protected:
    low-mineral TECH_ROBO with no ready/pending Robo returns STAY_COURSE to
    save resources instead of PRODUCE_ARMY.

  ADD_GATEWAYS -> FORGE_UPGRADES is reduced:
    capped ADD_GATEWAYS falls to TECH_ROBO before the first Robo, then to
    PRODUCE_ARMY once Robo is started/ready.

  Static-defense repeat cap remains:
    rejected BUILD_STATIC_DEFENSE under threat falls to PRODUCE_ARMY.

  Observer/Immortal gap is tracked:
    once a Robo is ready, repeated TECH_ROBO falls to PRODUCE_ARMY so the
    production path can fill Observer/Immortal instead of adding another Robo.
```

Guarded revised Power A/B result:

```text
Scenario:
  AcropolisLE / Hard / Terran / Power
  3 games per side
  guard pid=21392

No-filter coverage-teacher:
  run: runs\20260623_170757_20260623_eval_power_ab_no_filter_revised_v1
  strategy trajectory: data\trajectories\power_ab_no_filter_revised_strategy_v1
  result: 1 Victory / 2 Defeat / 0 Tie

Revised Power tactic filter:
  run: runs\20260623_171134_20260623_eval_power_ab_revised_tactic_v1
  strategy trajectory: data\trajectories\power_ab_revised_tactic_strategy_v1
  result: 0 Victory / 2 Defeat / 1 Tie

Diagnostics:
  runs\20260623_power_ab_revised_diagnostics_v1\artifacts\power_ab_diagnostics.txt
  runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_tactic_diagnostics.json
```

A/B interpretation:

```text
Robo was partially improved:
  no-filter first TECH_ROBO: 251.4 / 251.4 / 251.4s
  revised first TECH_ROBO: 274.3 / 297.1 / 274.3s
  revised ready_robo: 331.4 / 354.3 / 331.4s
  The prior 628.6s first-Robo delay is gone, but no-filter is still earlier.

Gateway rhythm was not fixed:
  no-filter ADD_GATEWAYS count=6, min=91.4s, avg=110.5s
  revised ADD_GATEWAYS count=5, min=502.9s, avg=685.7s
  revised filter still changed SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS 7 times.

Observer/Immortal payoff remains unreliable:
  revised Tie got Observer at 560.0s and Immortal at 605.7s
  revised Defeat files still had no Observer / no Immortal

Conclusion:
  revised filter is not promoted.
  do not collect tactic-aware training data from this filter.
```

## 阶段 6：训练路线

不要马上训练。推荐顺序：

```text
1. AIBuild metadata 接入并测试。
2. AIBuild 小矩阵采集 rule / coverage-teacher 数据。
3. 增加按 opponent_ai_build 的诊断。
4. 设计 TacticSpec 和 rule TacticSelector。
5. StrategyOutcomeDiagnostics 已完成，确认 strategy action 是否在 +30/+60/+90/+120s 落地。
6. 用 outcome diagnostics 改成 guardrail-first tactic filter，并优先修复 ready_robo 后
   Robo idle 但没有 Observer/Immortal 的生产触发缺口。
7. guarded smoke / Power-only A/B 确认 tactic-aware executor 不拖累 no-filter。
8. 仅在 guarded comparison 稳定优于 no-filter 后，才采 tactic-aware coverage teacher 数据。
9. 用 scripts/diagnose_tactics.py 和 outcome diagnostics 同时诊断 tactic distribution、filter changes、action payoff。
10. 再训练 tactic-aware strategy imitation。
```

后续 imitation 可以拆两层：

```text
model A: observation + context -> tactic_id
model B: observation + tactic_id -> strategy_action
```

但第一版不建议同时训练两个模型。先让规则 TacticSelector 产生稳定数据。

## 推荐验证矩阵

### 单元测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_evaluate.py tests\test_summarize_eval.py tests\test_strategy_trajectory_recording.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_coverage_strategy_policy.py tests\test_strategy_policy.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

### 环境检查

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

### SC2 启动前 guard

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

### AIBuild smoke

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Medium --opponents Terran --ai-builds Rush --games-per-combo 1 --run-root runs --run-name eval_aibuild_smoke_v1 --policy-name rule_aibuild_smoke --army-policy rule --strategy-policy rule --trajectory-dir data\trajectories\aibuild_smoke_army_v1 --strategy-trajectory-dir data\trajectories\aibuild_smoke_strategy_v1 --record-decision-interval 16 --game-time-limit 600
```

### AIBuild strategy data smoke

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Rush Timing Macro Air --games-per-combo 1 --run-root runs --run-name eval_strategy_coverage_teacher_aibuild_terran_smoke_v1 --policy-name strategy_coverage_teacher_aibuild_terran_smoke_v1 --army-policy rule --strategy-policy coverage-teacher --trajectory-dir data\trajectories\strategy_coverage_teacher_aibuild_terran_army_smoke_v1 --strategy-trajectory-dir data\trajectories\strategy_coverage_teacher_aibuild_terran_strategy_smoke_v1 --record-decision-interval 16 --game-time-limit 900
```

## 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| `AIBuild` 只影响内置 AI，不能完整代表真实战术 | 数据标签会粗糙 | 把它作为 scenario label，不作为真值 teacher |
| build 维度扩大后采集成本上升 | 评测耗时变长 | 先 Hard 小矩阵 smoke，再扩大 |
| trajectory metadata 改动破坏旧数据读取 | 诊断/训练失败 | 缺失 build 默认 `RandomBuild`，不影响 observation schema |
| tactic filter 过强，压制了 rule baseline | 在线胜率下降 | tactic mode 显式 opt-in，先 smoke 对比 rule |
| 战术切换过频繁 | 动作抖动，资源节奏乱 | minimum tactic duration + emergency-only override |
| StrategyExecutor 继续抢资源 | Hard Terran 退化 | TacticSpec 加 resource reserve 和 pending/cooldown 保护 |

## 下一步开发顺序

建议后续 agent 按这个顺序做：

```text
1. Revised TECH_POWER 已完成 focused offline tests 和 guarded Power-only A/B。
2. 不要推广 revised tactic filter；它没有赢过 no-filter coverage-teacher。
3. StrategyOutcomeDiagnostics 已实现，结果在
   `runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.txt`
   和 `.json`。
4. 已离线修第一层 SAFE_MACRO guardrail：240s 前 1 个 pending Gateway 仍保留
   ADD_GATEWAYS，2 个 pending Gateway 才 fallback。
5. 已补 ready_robo 产兵单测和实现：PRODUCE_ARMY 会在 ready Robo 上补
   Observer / Immortal，并继续委托 Gateway 出兵。
6. guarded Power-only A/B 已完成：no-filter coverage-teacher 1W/2L，tactic
   guardrail 1W/2L。Outcome 显示 source-level early Gateway 明显改善，三局 tactic
   guardrail 都补到 Observer，但仍没有 Immortal payoff；不要推广。
7. 最新离线 follow-up 已解释第三局 late ADD_GATEWAYS：102.9s 的早期
   ADD_GATEWAYS 被压制时已有 2 个 pending Gateway，后续 ready Gateways 到过 4；
   571.4s 是后期产能被打掉后的重建标签。
8. StrategyOutcomeDiagnostics 已扩展 Robo payoff blocker；三局 tactic guardrail
   都补到 Observer，Immortal 未落地均为 resource_or_supply_blocked。
9. 已做一个最小 opt-in runtime 修复：first-Immortal bias，仅在 tactic filter
   显式启用时，让 ready Robo + Observer 后的第一只 Immortal 优先于继续花矿。
10. first-Immortal bias fresh-dir guarded Power-only A/B 已完成：tactic-rule
   产出 1 局 Immortal、Observer 未退化，但 0W/3L 没赢过 no-filter 0W/3L；
   source-level first ADD_GATEWAYS 明显退到 422.9s，static-defense suppression
   仍让 threat persisted。
11. 不要推广 first-Immortal bias tactic filter，也不要用这轮数据训练 tactic-aware
   imitation / outcome / veto 模型。
12. 已离线修 Gateway precedence 和 static-defense retention：underbuilt Gateway
   时 first-Immortal bias 不再抢 `ADD_GATEWAYS`；`TECH_POWER` 在 active threat
   且 static-defense slot 可用时保留/留矿 `BUILD_STATIC_DEFENSE`。
13. Gateway/static-defense retention fresh-dir guarded Power-only A/B 已完成：
   no-filter coverage-teacher 2W/1L，tactic-rule 0W/3L。不要推广，不要采
   tactic-aware 数据。
14. 诊断显示 tactic-rule 仍过度压 `TECH_ROBO`；第三局在可负担第一座 Robo
   时选了 `STAY_COURSE`，之后没有 ready Robo。
15. 已离线补 initial-Robo precedence：`TECH_POWER` 在 no ready/pending Robo、
   Core ready、无威胁且资源足够时，可将 `STAY_COURSE` 改成 `TECH_ROBO`。
16. Initial-Robo fresh-dir guarded Power-only A/B 已完成：no-filter 1W/1T/1L，
   tactic-rule 0W/1T/2L。tactic 三局均有 ready Robo，两局产出 Immortal，但
   `ADD_GATEWAYS` 仍明显晚于 no-filter；不要推广，不要采 tactic-aware 数据。
17. 已离线补 midgame Gateway cap：`TECH_POWER` 在 Robo 已 ready/pending、无威胁、
   Gateway 低于 `own_bases * 4` 时允许最多 2 个 pending Gateway；no-Robo first
   tech fallback 仍保持。
18. Midgame Gateway cap fresh-dir guarded Power-only A/B 已完成：no-filter
   coverage-teacher 1W/2L，tactic-rule 1W/2L；first `ADD_GATEWAYS` 改善到
   171.4s，但仍晚于 no-filter 91.4s，且仍有 no-ready-Robo / missed-Immortal
   文件。不要推广，不要采 tactic-aware 数据。
19. 已离线补 first-Robo banking guard：`TECH_POWER` 在 no ready/pending Robo、
   Core ready、无威胁时，会把 `FORGE_UPGRADES` / `PRODUCE_ARMY` / `BOOST_WORKERS`
   等可能打散第一座 Robo 的动作改为 `TECH_ROBO` 或留矿；安全 `EXPAND` 保留。
20. 最新离线验证：focused tactic tests 39 passed，full pytest 178 passed，
   `scripts\check_env.py` OK。
21. First-Robo banking guard fresh-dir guarded Power-only A/B 已完成：no-filter
   coverage-teacher 1W/1T/1L，tactic-rule 2W/1T/0L；Robo/Observer/threat
   指标改善，但 first `ADD_GATEWAYS` 退到 411.4s，仍不要采 tactic-aware 数据。
22. 已离线补 Gateway preservation follow-up：`SAFE_MACRO` 在 120s 前、无 ready
   Gateway、已有 2 pending Gateway、minerals>=250、vespene<100、无威胁时，
   允许最多 3 个 pending Gateway；Robo gas ready 后或 active base threat 下
   仍保持 2 pending cap。最新边界验证为 focused tactic tests 44 passed，
   full pytest 183 passed，`scripts\check_env.py` OK。
23. Gateway preservation fresh-dir guarded Power-only A/B 已完成：no-filter
   coverage-teacher 1W/2L，tactic-rule 2W/1T/0L；first `ADD_GATEWAYS` 回到
   91.4s，三局 tactic 都有 ready Robo，2/3 Observer，1/3 Immortal。仍不要采
   tactic-aware 数据，因为一局 Observer resource/supply blocked，且 Tie threat_rows=17。
24. Gateway preservation 6-valid confirmatory Power-only A/B 已完成：no-filter
   valid 2W/1T/3L（另有 1 个 NO_RESULT 后 fresh-dir top-up），tactic-rule
   2W/1T/3L。Gateway timing 保住，Robo payoff 改善到 ready 5/6、Observer
   5/6、Immortal 2/6；但 tactic base_threat_rows=44 vs no-filter 39，
   BUILD_STATIC_DEFENSE=13 vs 39，filter_change_rows=75。不要推广，不要采
   tactic-aware 数据，不训练。
25. 已离线补 static-defense retention follow-up：active threat 下把 static
   retention 扩到 ANTI_RUSH_DEFENSE / ROBO_TIMING / ANTI_AIR_RESPONSE /
   RECOVERY；无 ready/pending static 且矿不足时留矿，已有 static 且矿不足时
   可 fallback 到产兵。
26. 已扩展 `scripts\diagnose_power_tactics.py`：`TECH_ROBO -> STAY_COURSE`
   会输出 `robo_banking_filter_contexts`。旧 confirm6 和新 confirm6 的
   26 条 `TECH_POWER TECH_ROBO -> STAY_COURSE` 都是
   `first_robo_mineral_short`，不是可负担 first Robo 被误压。
27. Static-retention confirm6 fresh-dir guarded Power-only A/B 已完成：
   no-filter coverage-teacher 3W/3L，tactic-rule 3W/1T/2L；first
   `ADD_GATEWAYS` 保持 91.4s，Robo payoff 提升到 ready 5/6、Observer 4/6、
   Immortal 2/6。但 tactic `base_threat_rows=52` vs no-filter 46，
   `BUILD_STATIC_DEFENSE=6` vs 43，`filter_change_rows=82`。不要推广，
   不采 tactic-aware 数据，不训练。
28. 最新验证：focused tactic/power tests 52 passed，full pytest 188 passed，
   `scripts\check_env.py` OK。A/B 使用 guard pid 26628，且只通过
   `scripts\evaluate.py` 启动 SC2。
29. 已扩展 static-defense filter context 诊断：`scripts\diagnose_power_tactics.py`
   会统计 active threat 下 `BUILD_STATIC_DEFENSE -> <other>` 的
   `static_defense_filter_contexts`。旧 static-retention confirm6 tactic 轨迹中
   `no_static_affordable=0`，主要可行动问题是 `pending_static_waiting`：已有
   pending static、没有 ready static 时仍 fallback 到 `PRODUCE_ARMY` / `TECH_ROBO`。
30. 已离线补 pending-static wait follow-up：active threat 下，如果 proposed action
   是 `BUILD_STATIC_DEFENSE` 且 pending static 已到 cap、ready static 仍为 0，
   tactic filter 改为 `STAY_COURSE` 等待落地；如果已经有 ready static，则保留
   原 fallback 行为。离线 replay 只改变 5 行，均为
   `base_under_threat=1 / pending_static_defense=1 / ready_static_defense=0`。
31. 最新验证：focused tactic/power tests 56 passed，full pytest 192 passed，
   `scripts\check_env.py` OK。这个 follow-up 没有启动 SC2、没有训练/PPO、没有采
   tactic-aware 数据，默认仍是 `--strategy-policy rule / --strategy-tactic-mode off`。
32. Pending-static wait confirmatory Power-only A/B 已完成：no-filter
   coverage-teacher 1W/2T/3L，tactic-rule 1W/1T/4L。tactic ready
   Robo/Observer/Immortal 改善到 5/6、5/6、1/6，但 first `ADD_GATEWAYS`
   平均退到 121.9s（no-filter 97.1s），`ADD_GATEWAYS` count 15 vs 28，
   `base_threat_rows=72` vs 33，`BUILD_STATIC_DEFENSE=15` vs 33。
33. Pending-static wait 没有通过 guarded comparison：不要推广 tactic filter，
   不采 tactic-aware 数据，不训练/PPO。下一步先离线拆 active-threat outcome，
   特别是 `ready_static_low_minerals -> PRODUCE_ARMY`、`pending_static_with_ready`
   和 `no_static_mineral_short -> STAY_COURSE` 是否真的能在 +30/+60/+90/+120s
   清 threat。
34. 已新增 active-threat outcome diagnostics：
   `rl\active_threat_outcome_diagnostics.py`、
   `scripts\diagnose_active_threat_outcomes.py`、
   `tests\test_active_threat_outcome_diagnostics.py`。focused active/power/outcome
   diagnostics tests 12 passed，full pytest 194 passed，`scripts\check_env.py` OK。
   产物在
   `runs\20260624_active_threat_outcome_pending_static_wait_v1\artifacts\`。
35. 路线修正：冻结当前 tactic-rule runtime，停止“手写小 patch + 小样本 A/B +
   再补副作用”的循环。新诊断显示坏桶是 tactic-specific：
   `ANTI_AIR_RESPONSE ready_static_low_minerals -> PRODUCE_ARMY` 在 +60s
   仍 11/11 threat persisted，+120s 仍 5/11 persisted；但 `RECOVERY`
   同 context 到 +120s 为 8/8 cleared，不能做全局 ready-static fallback。
36. 下一步只考虑 ANTI_AIR active-threat 的窄修复；修前先看 exact timeline，
   判断应该 `STAY_COURSE` 留矿、保留 `BUILD_STATIC_DEFENSE`，还是调整 anti-air
   产兵/防守 bias。默认 rule/off 不变。
37. 已离线实施 ANTI_AIR ready-static 低矿窄修复：显式 opt-in tactic filter 下，
   active base threat + `base_under_air_threat > 0` + 已有 ready static、无 pending
   static、minerals < 100 时，`BUILD_STATIC_DEFENSE` 不再 fallback 到
   `PRODUCE_ARMY`，而是 `STAY_COURSE` 留矿/稳住。ground-only threat 仍产兵，
   `RECOVERY` 不变。
38. 离线 replay 只改变旧 pending-static wait tactic A/B 中的 11 行，全部是
   `Power / ANTI_AIR_RESPONSE / BUILD_STATIC_DEFENSE` 从 recorded `PRODUCE_ARMY`
   变为 current `STAY_COURSE`；对应三个文件为 002/003/004，完全命中坏桶。
39. 最新验证：focused tactic tests 53 passed，active/power/outcome diagnostics
   tests 12 passed，full pytest 196 passed，`scripts\check_env.py` OK。未启动 SC2，
   未训练/PPO，未采 tactic-aware 数据，默认仍是
   `--strategy-policy rule / --strategy-tactic-mode off`。
40. Anti-air ready-static fresh-dir guarded Power-only A/B 已完成：no-filter
   coverage-teacher 2W/2T/2L，tactic-rule 0W/0T/6L。guard pid 26628，SC2
   只通过 `scripts\evaluate.py` 启动。
41. 这轮 tactic first `ADD_GATEWAYS` 仍为 91.4s，Robo/Observer 覆盖改善到
   ready 6/6、Observer 6/6，但 Immortal 仍 0/6，且 match result 全败。
   Tactic `BUILD_STATIC_DEFENSE=19` vs no-filter 26，`PRODUCE_ARMY=47` vs
   25，`TECH_ROBO=6` vs 34，threat actions 中 `STAY_COURSE=35`、`PRODUCE_ARMY=27`。
42. 新 anti-air STAY_COURSE 小桶本身只有 2 行，+120s threat_cleared=2/2；
   旧 anti-air PRODUCE_ARMY 小桶 2 行 +120s 仍 2/2 persisted。说明局部诊断
   方向成立，但 broader tactic-rule 仍失败，不能推广、采数据或训练。
43. 下一轮对局前必须先运行 hidden-window guard，并且只能用 scripts\evaluate.py /
   scripts\safe_launch.py，fresh dirs。
44. 只有稳定优于 no-filter coverage-teacher，且 Gateway timing 不明显退化后，
   再扩大 AIBuild 矩阵。
45. 数据稳定后再考虑 tactic-aware imitation / outcome / veto training。
46. 已新增 active-threat suppression diagnostics：
   `rl\active_threat_suppression_diagnostics.py`、
   `scripts\diagnose_active_threat_suppression.py`、
   `tests\test_active_threat_suppression_diagnostics.py`。它专门离线诊断
   `RECOVERY` / `TECH_POWER` 下 `BUILD_STATIC_DEFENSE`、`TECH_ROBO` 被改写为
   `PRODUCE_ARMY` / `STAY_COURSE` 后的 +30/+60/+90/+120s outcome、per-file
   timeline 和 replay-only candidate impact。
47. 最新 suppression 产物在
   `runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\`。
   no-filter target_suppression_rows=0；tactic-rule target_suppression_rows=50。
   Replay-only pass-through 会恢复 `BUILD_STATIC_DEFENSE +42`、`TECH_ROBO +8`，
   同时减少 `PRODUCE_ARMY -29`、`STAY_COURSE -21`，但只有 6/50 candidate rows
   立即可执行。
48. 最大静态防守 suppression 桶是 `RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY`
   / `ready_static_low_minerals` / `ground_threat`：18 行在 +30s 15/18
   threat persisted，+60s 10/18 persisted，+120s 仍 2/18 persisted，且
   static / army / Robo payoff 为负。但它不是全局可一刀切桶，因为多数行最终
   清掉 threat。
49. `pending_static_waiting -> STAY_COURSE` 会让 static 落地，但
   `TECH_POWER` 到 +120s 仍 3/5 threat persisted，`RECOVERY` 到 +120s 仍 2/4
   persisted；单纯等待 pending static 不是足够的 active-threat 方案。
   `TECH_ROBO` suppression 的 8 行都是 no-threat `first_robo_mineral_short`，
   不是资源已足的 active-threat Robo 被误压。
50. 最新验证：suppression focused tests 2 passed，active-threat / power /
   strategy diagnostics focused tests 14 passed，full pytest 198 passed，
   `scripts\check_env.py` OK。本轮未启动 SC2、未运行 guard、未训练/PPO、未采
   tactic-aware 数据。下一步继续冻结 tactic-rule runtime，直到出现更窄、更清晰、
   replay diff 更小的 outcome-backed 修复候选。
```

## 当前推荐结论

可以参考官方 AI 的 `Rush` / `Timing` / `Power` / `Macro` / `Air` 来组织训练场景和战术池骨架，但不要把它当成我方最终打法。AIBuild 评测/采集维度和 metadata 已接入；当前 `TacticSpec` 雏形也已接入，但 revised filter 没有赢过 no-filter。下一步要先让数据回答“动作有没有落地”，再把我方策略升级为更稳的 guardrail-first 战术池。

PPO 仍不建议进入。当前更重要的是让 strategy layer 的宏观动作拥有明确战术上下文和可量化 outcome，并证明它不会拖累默认 rule baseline。
