# Strategy Outcome 开发计划

## 2026-06-25 路线修正

当前计划已从“继续调 tactic filter”修正为“先补执行可观测性，再做候选审计”。

短期决策：

```text
冻结 tactic-rule runtime。
不推广 --strategy-tactic-mode rule。
不采 tactic-aware training data。
不训练 tactic-aware imitation。
不启动 PPO。
```

当前实现主线：

```text
StrategyAction -> StrategyExecutor execution result -> strategy trajectory metadata
-> execution effect/blocker diagnostics -> candidate audit -> replay-only candidate
```

新增总览文档：

```text
doc\DEVELOPMENT_PLAN.md
```

当前阶段优先开发 execution observability。新 strategy trajectory 应记录：

```text
strategy_execution_attempted
strategy_execution_effect
strategy_execution_blocker
strategy_execution_unit_type
strategy_execution_target
```

这样 outcome diagnostics 才能区分：

```text
动作没有执行成功
动作执行了但 payoff 延迟
动作执行了但资源 / 供应 / 产能阻塞
动作被 tactic filter 改写后才进入 executor
```

## 目标

定义下一阶段 strategy / tactic 层开发计划。当前阶段优先建立离线
strategy action 结果诊断能力，在继续修改 tactic filter、采集数据、训练模型或考虑
PPO 之前，先确认每个宏观策略动作是否在后续游戏状态中产生了预期结果。

## 范围

本阶段包含：

- 新增离线诊断，用固定 lookahead 窗口衡量 strategy action 是否落地。
- 使用现有 Power build strategy trajectory，对比 no-filter coverage-teacher 与
  revised Power tactic filter。
- 在诊断证据明确后，将 tactic filter 调整为 guardrail-first 风格。
- 保持默认 rule/no-op runtime 行为不变。
- 保持 strategy / tactic 功能显式 opt-in。

本阶段不包含：

- PPO 实现。
- PPO / RL 主线改动。
- 新 tactic-aware 模型训练。
- 使用当前或 revised tactic filter 采集 tactic-aware 训练数据。
- 将 tactic metadata 加入 observation schema。
- 替换或削弱默认 rule baseline。

上面的“不包含”是当前 outcome-diagnostics 阶段的范围控制，不是项目永久禁令。
当 guarded A/B 和 outcome diagnostics 证明 tactic/strategy 改动至少不拖累默认
baseline 后，可以按 fresh dirs、小批量、可回滚的方式采 tactic-aware 数据或训练
action-outcome / veto / imitation 模型。PPO 可以作为更后面的路线，但需要先补齐
SC2 environment、reward、baseline 对照和 safe-launch 边界。

## 默认运行边界

默认运行参数必须保持：

```text
--strategy-policy rule
--strategy-tactic-mode off
```

strategy / tactic 行为必须显式 opt-in。默认 rule baseline 必须继续保持可运行，
且不受 strategy / tactic 副作用影响。

## 安全要求

初始诊断实现不需要启动 SC2。

任何后续可能启动 SC2 的命令之前，必须先运行 hidden-window guard：

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

启动对局只能使用：

```text
scripts\evaluate.py
scripts\safe_launch.py
```

不要直接裸跑可见 `run.py`。任何新采集或新对照实验都必须使用 fresh run 和
trajectory 目录。

## 当前基础状态

已完成能力：

```text
AIBuild 评测 / 采集维度。
opponent_ai_build 写入 eval、summary、army trajectory、strategy trajectory。
strategy_v2 observation schema。
StrategyAction 宏观动作空间。
TacticSpec / TacticState / RuleTacticSelector 雏形。
显式 opt-in 的 tactic-aware coverage-teacher runtime。
scripts\diagnose_tactics.py。
scripts\diagnose_power_tactics.py。
```

当前测试状态：

```text
.\.venv\Scripts\python.exe -m pytest -q
201 passed
```

## 当前证据

最新 guarded Power-only A/B：

```text
场景：
  AcropolisLE / Hard / Terran / Power
  每组 3 局

No-filter coverage-teacher：
  run: runs\20260623_170757_20260623_eval_power_ab_no_filter_revised_v1
  strategy trajectory: data\trajectories\power_ab_no_filter_revised_strategy_v1
  result: 1 Victory / 2 Defeat / 0 Tie

Revised Power tactic filter：
  run: runs\20260623_171134_20260623_eval_power_ab_revised_tactic_v1
  strategy trajectory: data\trajectories\power_ab_revised_tactic_strategy_v1
  result: 0 Victory / 2 Defeat / 1 Tie
```

诊断结论：

```text
Robo timing：
  revised filter 修复了之前 628.6s 才 first TECH_ROBO 的严重延迟。
  但 revised first TECH_ROBO 仍晚于 no-filter。

Gateway rhythm：
  no-filter 的 ADD_GATEWAYS 出现较早。
  revised 的实际 ADD_GATEWAYS 明显偏晚。
  SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS 是重点嫌疑。

Observer / Immortal payoff：
  revised 只在 Tie 文件中补到了 Observer / Immortal。
  两个 revised Defeat 文件仍没有 Observer / Immortal。

结果：
  revised Power tactic filter 没有优于 no-filter coverage-teacher。
  revised filter 只能保留为诊断实验，不能推广。
```

## 开发方向

下一阶段评价目标从“标签像不像 teacher”转为“动作是否产生结果”。

旧循环：

```text
coverage-teacher -> trajectory -> imitation -> online eval -> 调 teacher
```

新循环：

```text
strategy action -> 结果诊断 -> guardrail / veto 设计 -> focused tests -> 小规模 guarded A/B
```

当前 tactic-aware 数据不能用于训练。后续如果引入学习模块，应优先考虑 action
critic / veto 模型，而不是直接训练新的 strategy controller。

## 阶段 1：StrategyOutcomeDiagnostics

新增文件：

```text
rl\strategy_outcome_diagnostics.py
scripts\diagnose_strategy_outcomes.py
tests\test_strategy_outcome_diagnostics.py
```

当前状态：已完成。

```text
focused pytest:
  .\.venv\Scripts\python.exe -m pytest tests\test_strategy_outcome_diagnostics.py -q
  4 passed

full pytest:
  .\.venv\Scripts\python.exe -m pytest -q
  159 passed

check_env:
  .\.venv\Scripts\python.exe scripts\check_env.py
  OK
```

首批输入 trajectory：

```text
data\trajectories\power_ab_no_filter_revised_strategy_v1
data\trajectories\power_ab_revised_tactic_strategy_v1
```

lookahead 窗口：

```text
+30s
+60s
+90s
+120s
```

诊断必须逐文件处理 strategy trajectory，并输出聚合统计。

必需输出：

```text
human-readable report
JSON report
逐文件结果摘要
按 action 聚合的 outcome 摘要
按 action + lookahead window 聚合的 outcome 摘要
存在 tactic metadata 时，输出 filter-change outcome 摘要
```

实现约束：

```text
仅离线分析。
不启动 SC2。
不改 observation schema。
不改 runtime 行为。
不改 PPO / RL 主线。
```

## 必需结果指标

### ADD_GATEWAYS

衡量指标：

```text
ready_gateway_delta
pending_gateway_seen
first_pending_gateway_after_action
first_ready_gateway_delta_time
minerals_after_action
idle_gateway_after_action
```

需要回答：

```text
ADD_GATEWAYS 是否真的带来额外产能？
SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS 是否压制了早期 Gateway timing？
```

### TECH_ROBO

衡量指标：

```text
pending_robo_seen
ready_robo_seen
first_pending_robo_after_action
first_ready_robo_after_action
observer_delta
immortal_delta
robo_idle_after_ready
```

需要回答：

```text
TECH_ROBO 是否及时开始和完成 Robotics Facility？
ready_robo 后是否生产 Observer 或 Immortal？
Robo payoff 缺失来自 action、资源、idle production，还是 executor 行为？
```

### PRODUCE_ARMY

衡量指标：

```text
army_count_delta
zealot_delta
stalker_delta
sentry_delta
immortal_delta
minerals_bank_after
vespene_bank_after
idle_gateway_after
idle_robo_after
```

需要回答：

```text
PRODUCE_ARMY 是否提升 army_count？
static defense 被 filter 成 PRODUCE_ARMY 后，替代动作是否真的产兵？
idle production 或资源囤积是否阻止了预期结果？
```

### BUILD_STATIC_DEFENSE

衡量指标：

```text
static_defense_delta
base_under_threat_after
threat_persisted
threat_cleared
army_count_delta_under_threat
```

需要回答：

```text
static defense 是否降低或清除了 base threat？
压制重复 static defense 后，威胁处理是改善还是变差？
如果 threat 持续，是否应该保留一次 static defense？
```

### FORGE_UPGRADES

衡量指标：

```text
forge_pending_seen
forge_ready_seen
upgrade_pending_seen
upgrade_level_delta
army_count_at_upgrade
ready_gateway_count_at_upgrade
ready_robo_at_upgrade
```

需要回答：

```text
FORGE_UPGRADES 是否带来升级结果？
Forge / upgrade 是否过早，占用了 army、Gateway 或 Robo timing 的资源？
```

### EXPAND

衡量指标：

```text
base_count_delta
pending_nexus_seen
worker_count_delta
worker_saturation_after
army_count_delta_after_expand
base_under_threat_after
```

需要回答：

```text
EXPAND 是否产生新基地或 pending Nexus？
扩张 timing 是否削弱防守或生产？
```

### BOOST_WORKERS

衡量指标：

```text
worker_count_delta
worker_saturation_after
army_count_delta_after_workers
gateway_count_delta_after_workers
minerals_bank_after
```

需要回答：

```text
BOOST_WORKERS 是否改善经济且不压制早期 Gateway rhythm？
补农民是否和造兵、科技 timing 竞争资源？
```

## 阶段 2：复盘现有 A/B

新增诊断后运行：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_strategy_outcomes.py data\trajectories\power_ab_no_filter_revised_strategy_v1 data\trajectories\power_ab_revised_tactic_strategy_v1 --show-files --json-output runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.json --text-output runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.txt
```

报告必须回答：

```text
1. SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS 是否压制了 240s 前的 Gateway timing？
2. TECH_POWER ADD_GATEWAYS -> PRODUCE_ARMY 是否提升了 army_count？
3. revised Defeat 文件中，ready_robo 后为什么没有 Observer / Immortal？
4. static-defense suppression 是否让 base threat 更快清除，还是持续更久？
5. no-filter 胜局和 revised 败 / 平局最大的 outcome 差异是什么？
```

首批诊断结果：

```text
run/artifacts:
  runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.json

inputs:
  data\trajectories\power_ab_no_filter_revised_strategy_v1
  data\trajectories\power_ab_revised_tactic_strategy_v1

files=6
rows=393
training_rows=387
results=1 Victory / 1 Tie / 4 Defeat
```

答案摘要：

```text
1. SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS:
   count=7, all before 240s. It suppresses early strategy-level ADD_GATEWAYS
   labels; the default rule build loop still constructs some Gateways, so this
   is not a total production block, but it does explain why revised actual
   ADD_GATEWAYS is very late.

2. TECH_POWER ADD_GATEWAYS -> PRODUCE_ARMY:
   count=2. It does not improve immediate +30/+60 army count, but both samples
   show Observer/Immortal payoff by +120s. Treat as promising-but-too-small,
   not a promotion signal.

3. Revised Defeat Robo payoff:
   ready_robo appears at 331.4s / 354.3s in the two revised Defeat files, but
   no Observer or Immortal appears. After ready_robo, robo_idle_count stays 1
   repeatedly; later actions are mostly STAY_COURSE / PRODUCE_ARMY /
   FORGE_UPGRADES, so StrategyExecutor does not get a TECH_ROBO trigger to train
   Robo units.

4. Static-defense suppression:
   TECH_POWER BUILD_STATIC_DEFENSE -> PRODUCE_ARMY cleared threat in 3/3
   samples by +30s. RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY was mixed
   early: at +60s, 7/8 cleared and 1/8 persisted; by +120s, 8/8 cleared.

5. Biggest no-filter vs revised difference:
   no-filter first ADD_GATEWAYS=91.4s and first TECH_ROBO=251.4s; revised first
   actual ADD_GATEWAYS=502.9s and first TECH_ROBO=274.3s. Revised has 42 filter
   changes and still misses Observer/Immortal in both Defeat files.
```

## 阶段 3：Guardrail-First Tactic Filter

阶段 1 诊断和 focused tests 完成前，不实施本阶段 runtime 改动。

设计原则：

```text
默认 pass-through。
只拦截或改写 repeated、pending-capped、resource-invalid 或明确有害动作。
避免大面积 action rewrite。
```

候选护栏：

```text
Early Gateway：
  已离线实现第一层 guardrail：SAFE_MACRO 在 240s 前保留 ADD_GATEWAYS，
  除非 minerals < 100 或 pending_gateways >= 2。

First Robo：
  Cybernetics Core ready 且 no ready/pending Robo 时，保留一次及时 TECH_ROBO。
  如果 minerals 不足但 vespene 充足，优先 STAY_COURSE 存资源，而不是转移到其他消耗动作。

Robo payoff：
  PRODUCE_ARMY 已补 ready-Robo 产兵单测和实现：若 ready Robo idle，
  先补 Observer，已有 Observer 后补 Immortal，同时继续委托 Gateway 出兵。
  后续仍需用 outcome diagnostics 复查 +30/+60/+90/+120s payoff。
  不把 tactic metadata 加入 observation schema。

Static defense：
  保留 repeat cap。
  是否调整 threat fallback 必须由 outcome diagnostics 支持。

Forge / upgrades：
  不把 FORGE_UPGRADES 当作 capped ADD_GATEWAYS 的默认 fallback。
  upgrade timing 应受 army、Gateway 或 Robo readiness 约束。
```

## 阶段 4：可选 Production Bias Hook

只有当 outcome diagnostics 证明 ready_robo 经常因为生产优先级缺失而没有产出
Observer / Immortal 时，才考虑本阶段。

候选 opt-in 参数：

```text
--strategy-production-bias tactic
```

候选行为：

```text
if tactic in TECH_POWER / ROBO_TIMING and ready_robo > 0:
  需要侦测 / 侦察时，优先补一个 Observer
  Power / Timing / armored pressure 相关时，优先补一个 Immortal
```

约束：

```text
默认 off。
不影响 --strategy-policy rule / --strategy-tactic-mode off。
不改 observation schema。
任何 guarded game 前必须先补 focused tests。
```

## 阶段 5：Guarded Power-Only A/B

只有满足以下条件后才能运行：

```text
StrategyOutcomeDiagnostics 已实现。
focused tests 通过。
guardrail 改动有离线证据支持。
已准备 fresh dirs。
```

对照场景：

```text
AcropolisLE / Hard / Terran / Power
no-filter coverage-teacher vs guardrail revised tactic filter
```

当前 guarded Power-only A/B 复查：

```text
No-filter coverage-teacher:
  run: runs\20260624_103759_20260624_power_ab_guardrail_no_filter_v1
  strategy trajectory: data\trajectories\power_ab_guardrail_no_filter_strategy_v1
  result: 1 Victory / 2 Defeat

Tactic guardrail:
  run: runs\20260624_104104_20260624_power_ab_guardrail_tactic_v1
  strategy trajectory: data\trajectories\power_ab_guardrail_tactic_strategy_v1
  result: 1 Victory / 2 Defeat

Outcome diagnostics:
  runs\20260624_strategy_outcome_power_ab_guardrail_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_guardrail_v1\artifacts\strategy_outcomes.json

Conclusion:
  do not promote tactic guardrail yet.
```

最新离线 follow-up：

```text
Tactic filter timeline:
  runs\20260624_guardrail_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_guardrail_tactic_timeline_v1\artifacts\tactic_timeline.json

Enhanced Robo payoff outcome:
  runs\20260624_guardrail_tactic_robo_outcomes_v1\artifacts\strategy_outcomes.txt
  runs\20260624_guardrail_tactic_robo_outcomes_v1\artifacts\strategy_outcomes.json

Findings:
  第三局 102.9s 的 ADD_GATEWAYS -> BOOST_WORKERS 发生在 pending_gateways=2。
  之后 ready Gateways 到过 4；571.4s 的 ADD_GATEWAYS 是后期产能被打掉后
  的重建标签，不应单独解释为 early Gateway 持续被 filter 压制。
  三局 tactic guardrail 的 Observer 均落地；Immortal 未落地的 blocker 均为
  resource_or_supply_blocked。

Runtime follow-up:
  仅在显式 opt-in tactic filter 下加入 first-Immortal bias：
  ready_robo > 0、observers > 0、immortals == 0、base_under_threat == 0 时，
  若资源/供给够则偏向 PRODUCE_ARMY；若矿不足但可能被其它动作消耗，则
  STAY_COURSE 银行到第一只 Immortal。
```

First-Immortal bias guarded A/B：

```text
No-filter coverage-teacher:
  run: runs\20260624_111950_20260624_power_ab_immortal_bias_no_filter_v1
  strategy trajectory: data\trajectories\power_ab_immortal_bias_no_filter_strategy_v1
  result: 0 Victory / 3 Defeat
  avg_duration_seconds: 45.49

Tactic-rule first-Immortal bias:
  run: runs\20260624_112225_20260624_power_ab_immortal_bias_tactic_v1
  strategy trajectory: data\trajectories\power_ab_immortal_bias_tactic_strategy_v1
  result: 0 Victory / 3 Defeat
  avg_duration_seconds: 59.72

Diagnostics:
  runs\20260624_power_ab_immortal_bias_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_immortal_bias_tactic_timeline_v1\artifacts\tactic_timeline.json
  runs\20260624_strategy_outcome_power_ab_immortal_bias_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_immortal_bias_v1\artifacts\strategy_outcomes.json

Findings:
  Tactic-rule produced Immortal in 1/3 files; no-filter produced Immortal in 0/3.
  Observer did not regress in tactic-rule; all 3 files produced Observer.
  Gateway timing regressed badly: source-level first ADD_GATEWAYS 91.4s -> 422.9s.
  Static-defense suppression remains unsafe: TECH_POWER BUILD_STATIC_DEFENSE ->
  PRODUCE_ARMY had +60s threat_persisted=7/13; ANTI_AIR_RESPONSE had
  +60s threat_persisted=5/6.

Conclusion:
  Do not promote first-Immortal bias tactic filter.
  Do not collect tactic-aware training data from this run.
  Gateway precedence and static-defense retention must be fixed before another
  guarded comparison.
```

## 2026-06-24 Offline Guardrail Retention Fix

Status:

```text
No SC2 launch.
No training / PPO.
No tactic-aware data collection.
Default rule/off path unchanged.
```

Implemented:

```text
Gateway precedence:
  first-Immortal bias no longer rewrites ADD_GATEWAYS when
  ready_gateways + pending_gateways < own_bases * 4 and pending_gateways is below
  the tactic cap.

Static-defense retention:
  TECH_POWER now preserves BUILD_STATIC_DEFENSE under active base threat when a
  static-defense slot is available.
  minerals >= 100 -> BUILD_STATIC_DEFENSE
  minerals < 100  -> STAY_COURSE to bank
  pending_static_defense capped -> previous PRODUCE_ARMY fallback remains
```

Validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
31 passed

.\.venv\Scripts\python.exe -m pytest -q
170 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK
```

Next evidence step:

```text
Run hidden-window guard.
Use fresh dirs.
Use scripts\evaluate.py only.
Repeat guarded Power-only A/B against no-filter coverage-teacher.
Generate tactic timeline and enhanced strategy outcome diagnostics.
Promote nothing unless Gateway timing, Observer/Immortal payoff, static-defense
retention, and match results are at least not worse than no-filter.
```

## 2026-06-24 Guardrail Retention A/B Result

Fresh-dir guarded Power-only A/B was run after the Gateway/static-defense
retention fix.

```text
guard pid: 26628

no-filter coverage-teacher:
  run: runs\20260624_114236_20260624_power_ab_guardrail_retention_no_filter_v1
  trajectory: data\trajectories\power_ab_guardrail_retention_no_filter_strategy_v1
  result: 2 Victory / 1 Defeat

tactic-rule:
  run: runs\20260624_114514_20260624_power_ab_guardrail_retention_tactic_v1
  trajectory: data\trajectories\power_ab_guardrail_retention_tactic_strategy_v1
  result: 0 Victory / 3 Defeat

diagnostics:
  runs\20260624_power_ab_guardrail_retention_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_guardrail_retention_tactic_timeline_v1\artifacts\tactic_timeline.json
  runs\20260624_strategy_outcome_power_ab_guardrail_retention_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_guardrail_retention_v1\artifacts\strategy_outcomes.json
```

Findings:

```text
Tactic-rule still loses to no-filter and cannot be promoted.
Tactic-rule reduced TECH_ROBO from 36 no-filter rows to 3 selected rows.
Largest filter changes were still RECOVERY TECH_ROBO -> PRODUCE_ARMY,
RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY, and TECH_POWER TECH_ROBO ->
STAY_COURSE.

Robo payoff:
  no-filter produced ready Robo in both Victory files.
  tactic-rule produced ready Robo in two Defeat files and missed ready Robo
  entirely in the third Defeat file.

Root observed failure:
  In the third tactic file, TECH_POWER had resources and no threat at
  251.4s/274.3s but selected STAY_COURSE while no Robo existed. Later TECH_ROBO
  rows arrived when minerals were low or were filtered away, so no ready Robo
  ever landed.
```

Conclusion:

```text
Do not promote guardrail-retention tactic filter.
Do not collect tactic-aware training data from this A/B.
```

## 2026-06-24 Offline Initial-Robo Precedence Fix

Implemented offline after the negative A/B:

```text
TECH_POWER can rewrite STAY_COURSE -> TECH_ROBO only when:
  has_cybernetics_core > 0
  ready_robo == 0
  pending_robo == 0
  base_under_threat == 0
  TECH_ROBO passes tactic timing/resource/pending checks
```

Validation:

```text
focused tactic tests: 34 passed
full pytest: 173 passed
check_env: OK
```

Next evidence step:

```text
Initial-Robo precedence A/B has now been run.
Generate tactic timeline and strategy outcome diagnostics.
Do not train or collect tactic-aware data unless that comparison is positive.
```

## 2026-06-24 Initial-Robo A/B Result

Fresh-dir guarded Power-only A/B was run after the initial-Robo precedence fix.

```text
guard pid: 26628

no-filter coverage-teacher:
  run: runs\20260624_115458_20260624_power_ab_initial_robo_no_filter_v1
  trajectory: data\trajectories\power_ab_initial_robo_no_filter_strategy_v1
  result: 1 Victory / 1 Tie / 1 Defeat

tactic-rule:
  run: runs\20260624_115849_20260624_power_ab_initial_robo_tactic_v1
  trajectory: data\trajectories\power_ab_initial_robo_tactic_strategy_v1
  result: 0 Victory / 1 Tie / 2 Defeat

diagnostics:
  runs\20260624_power_ab_initial_robo_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_initial_robo_tactic_timeline_v1\artifacts\tactic_timeline.json
  runs\20260624_strategy_outcome_power_ab_initial_robo_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_initial_robo_v1\artifacts\strategy_outcomes.json
```

Findings:

```text
Initial-Robo precedence fixed the no-ready-Robo failure:
  tactic file 001: ready_robo=297.1 observer=422.9 immortal=491.4
  tactic file 002: ready_robo=297.1 observer=445.7 immortal=none
  tactic file 003: ready_robo=354.3 observer=422.9 immortal=502.9

It still did not beat no-filter:
  no-filter result: 1W/1T/1L
  tactic-rule result: 0W/1T/2L

Remaining regression:
  no-filter source first ADD_GATEWAYS=91.4s, count=18
  tactic source first ADD_GATEWAYS=377.1s, count=6
  TECH_POWER ADD_GATEWAYS -> PRODUCE_ARMY occurred 5 times.
```

Conclusion:

```text
Do not promote initial-Robo tactic filter.
Do not collect tactic-aware training data from this A/B.
```

## 2026-06-24 Offline Midgame Gateway Cap Fix

Implemented offline after the negative initial-Robo A/B:

```text
TECH_POWER allows up to 2 pending Gateways only when:
  base_under_threat == 0
  ready_robo > 0 or pending_robo > 0
  ready_gateways + pending_gateways < own_bases * 4

No-Robo first-tech behavior remains unchanged: a capped ADD_GATEWAYS with no
ready/pending Robo still falls back toward TECH_ROBO.
```

Validation:

```text
focused tactic tests: 36 passed
full pytest: 175 passed
check_env: OK
```

Next evidence step:

```text
The midgame Gateway cap A/B has now been run.
Do not train or collect tactic-aware data from that mixed comparison.
Use the first-Robo banking result and Gateway regression findings below.
```

## 2026-06-24 Midgame Gateway Cap A/B Result

Fresh-dir guarded Power-only A/B has now been run.

```text
guard pid: 26628

no-filter coverage-teacher:
  run: runs\20260624_134126_20260624_power_ab_midgame_gateway_no_filter_v1
  trajectory: data\trajectories\power_ab_midgame_gateway_no_filter_strategy_v1
  result: 1 Victory / 2 Defeat

tactic-rule:
  run: runs\20260624_134449_20260624_power_ab_midgame_gateway_tactic_v1
  trajectory: data\trajectories\power_ab_midgame_gateway_tactic_strategy_v1
  result: 1 Victory / 2 Defeat

diagnostics:
  runs\20260624_power_ab_midgame_gateway_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_midgame_gateway_tactic_timeline_v1\artifacts\tactic_timeline.json
  runs\20260624_strategy_outcome_power_ab_midgame_gateway_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_midgame_gateway_v1\artifacts\strategy_outcomes.json
```

Findings:

```text
Gateway timing improved but is still behind no-filter:
  no-filter first ADD_GATEWAYS=91.4s, count=7
  tactic-rule first ADD_GATEWAYS=171.4s, count=5
  previous initial-Robo tactic first ADD_GATEWAYS=377.1s

Robo payoff remains mixed:
  tactic file 001: no ready Robo
  tactic file 002: ready_robo=297.1 observer=468.6 immortal=514.3
  tactic file 003: ready_robo=297.1 observer=320.0 immortal=none,
    blocker=not_produced_after_affordable_action

Filter pressure is still high:
  filter_change_rows=71
  TECH_POWER TECH_ROBO -> STAY_COURSE: 28
  TECH_POWER BUILD_STATIC_DEFENSE -> STAY_COURSE: 13
```

Conclusion:

```text
Do not promote midgame Gateway cap tactic filter.
Do not collect tactic-aware training data from this A/B.
```

## 2026-06-24 Offline First-Robo Banking Guard

Implemented offline after the mixed midgame Gateway A/B:

```text
TECH_POWER now protects first-Robo resources across more original actions:
  STAY_COURSE
  ADD_GATEWAYS
  TECH_ROBO
  FORGE_UPGRADES
  BUILD_STATIC_DEFENSE
  PRODUCE_ARMY
  BOOST_WORKERS

When no ready/pending Robo exists, Core is ready, and there is no base threat:
  resources >= 150 minerals / 100 gas -> select TECH_ROBO
  gas ready but minerals short -> select STAY_COURSE to bank

EXPAND is not intercepted, preserving safe teacher expands.
The change remains explicit opt-in through tactic-aware strategy mode.
```

Validation:

```text
focused tactic tests: 39 passed
full pytest: 178 passed
check_env: OK
```

Next evidence step:

```text
The first-Robo banking A/B has now been run.
Do not train or collect tactic-aware data from this single positive sample.
Use a confirmatory fresh-dir A/B or a Gateway preservation follow-up next.
```

## 2026-06-24 First-Robo Banking A/B Result

Fresh-dir guarded Power-only A/B was run after the first-Robo banking guard.

```text
guard pid: 26628

no-filter coverage-teacher:
  run: runs\20260624_135850_20260624_power_ab_first_robo_bank_no_filter_v1
  trajectory: data\trajectories\power_ab_first_robo_bank_no_filter_strategy_v1
  result: 1 Victory / 1 Tie / 1 Defeat

tactic-rule:
  run: runs\20260624_140148_20260624_power_ab_first_robo_bank_tactic_v1
  trajectory: data\trajectories\power_ab_first_robo_bank_tactic_strategy_v1
  result: 2 Victory / 1 Tie / 0 Defeat

diagnostics:
  runs\20260624_power_ab_first_robo_bank_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_first_robo_bank_tactic_timeline_v1\artifacts\tactic_timeline.json
  runs\20260624_power_ab_first_robo_bank_power_tactics_v1\artifacts\power_tactic_diagnostics.txt
  runs\20260624_power_ab_first_robo_bank_power_tactics_v1\artifacts\power_tactic_diagnostics.json
  runs\20260624_strategy_outcome_power_ab_first_robo_bank_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_first_robo_bank_v1\artifacts\strategy_outcomes.json
```

Findings:

```text
Positive:
  tactic-rule beat no-filter in this 3-game sample: 2W/1T/0L vs 1W/1T/1L.
  tactic-rule had ready Robo in all files.
  tactic-rule produced Observer in all files.
  tactic-rule produced Immortal in 1 file.
  tactic base_threat_rows were 0 / 1 / 0 versus no-filter 11 / 0 / 16.
  filter_change_rows dropped to 23 from the prior midgame Gateway sample's 71.

Negative / still risky:
  no-filter first ADD_GATEWAYS=91.4s, count=13
  tactic-rule first ADD_GATEWAYS=411.4s, count=6
  SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS still occurred 6 times before 240s.
  Two tactic Victory files had no Immortal, both resource_or_supply_blocked.
```

Conclusion:

```text
Evidence supports continuing this branch, but not training from it yet.
The next change should protect Gateway rhythm without undoing first-Robo /
Observer / threat gains, or run a confirmatory fresh-dir A/B before any data
collection.
```

## 2026-06-24 Gateway Preservation Follow-Up

Implemented offline after first-Robo banking A/B:

```text
SAFE_MACRO pre-Robo-gas Gateway preservation:
  game_time < 120s
  ready_gateways == 0
  pending_gateways >= 2
  minerals >= 250
  vespene < 100
  base_under_threat == 0

Effect:
  allow up to 3 pending Gateways.

Safety:
  once vespene >= 100, keep the 2 pending Gateway cap so first-Robo banking is
  not interrupted once Robo gas is ready.
  under active base threat, keep the 2 pending Gateway cap as well.
```

Validation:

```text
focused tactic tests: 44 passed
full pytest: 183 passed
check_env: OK
```

Fresh-dir guarded Power-only A/B:

```text
guard pid: 26628

no-filter coverage-teacher:
  run: runs\20260624_141715_20260624_power_ab_gateway_preserve_no_filter_v1
  trajectory: data\trajectories\power_ab_gateway_preserve_no_filter_strategy_v1
  result: 1 Victory / 2 Defeat

tactic-rule:
  run: runs\20260624_141951_20260624_power_ab_gateway_preserve_tactic_v1
  trajectory: data\trajectories\power_ab_gateway_preserve_tactic_strategy_v1
  result: 2 Victory / 1 Tie / 0 Defeat

diagnostics:
  runs\20260624_power_ab_gateway_preserve_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_gateway_preserve_tactic_timeline_v1\artifacts\tactic_timeline.json
  runs\20260624_power_ab_gateway_preserve_power_tactics_v1\artifacts\power_tactic_diagnostics.txt
  runs\20260624_power_ab_gateway_preserve_power_tactics_v1\artifacts\power_tactic_diagnostics.json
  runs\20260624_strategy_outcome_power_ab_gateway_preserve_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_gateway_preserve_v1\artifacts\strategy_outcomes.json
```

Findings:

```text
Positive:
  tactic-rule again beat no-filter in this 3-game sample: 2W/1T/0L vs 1W/2L.
  first ADD_GATEWAYS returned to 91.4s, matching no-filter.
  ADD_GATEWAYS count improved to 11 versus first-Robo banking A/B's 6.
  tactic-rule had ready Robo in all files.
  tactic-rule produced Observer in 2/3 files and Immortal in 1/3 files.

Negative / still risky:
  one tactic Victory had ready Robo but no Observer, resource_or_supply_blocked.
  tactic Tie had base_threat_rows=17.
  filter_change_rows=26, still not sparse enough for training data.
```

Conclusion:

```text
Gateway preservation repaired the biggest selected-action regression without
breaking this sample's result. Still do not collect tactic-aware training data
or train from it. Next evidence step should either confirm with another
fresh-dir A/B or target post-ready-Robo Observer/resource blocking.
```

## 2026-06-24 Gateway Preservation Confirmatory A/B

Fresh-dir guarded Power-only confirmation:

```text
no-filter coverage-teacher:
  run: runs\20260624_143758_20260624_power_ab_gateway_confirm6_no_filter_v1
  trajectory: data\trajectories\power_ab_gateway_confirm6_no_filter_strategy_v1
  result: 2W / 3L / 1 NO_RESULT

no-filter top-up:
  run: runs\20260624_145141_20260624_power_ab_gateway_confirm6_no_filter_topup1_v1
  trajectory: data\trajectories\power_ab_gateway_confirm6_no_filter_topup1_strategy_v1
  result: 1T

tactic-rule:
  run: runs\20260624_145322_20260624_power_ab_gateway_confirm6_tactic_v1
  trajectory: data\trajectories\power_ab_gateway_confirm6_tactic_strategy_v1
  result: 2W / 1T / 3L
```

Diagnostics:

```text
runs\20260624_power_ab_gateway_confirm6_tactic_timeline_v1\artifacts\tactic_timeline.txt
runs\20260624_power_ab_gateway_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics.txt
runs\20260624_strategy_outcome_power_ab_gateway_confirm6_v1\artifacts\strategy_outcomes.txt
```

Aggregate result, excluding the no-filter empty NO_RESULT file:

```text
no-filter valid: 2W / 1T / 3L
tactic-rule:     2W / 1T / 3L

Gateway:
  no-filter first ADD_GATEWAYS=91.4s, count=25
  tactic-rule first ADD_GATEWAYS=91.4s, count=23

Robo payoff:
  no-filter ready_robo=4/6, Observer=4/6, Immortal=0/6
  tactic-rule ready_robo=5/6, Observer=5/6, Immortal=2/6

Threat/static defense:
  no-filter base_threat_rows=39, BUILD_STATIC_DEFENSE=39
  tactic-rule base_threat_rows=44, BUILD_STATIC_DEFENSE=13

Filter changes:
  tactic training_filter_change_rows=75
  TECH_POWER TECH_ROBO -> STAY_COURSE: 26
  RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 13
  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 9
```

Conclusion:

```text
Do not promote this tactic filter.
Do not collect tactic-aware training data yet.
Do not train action-outcome / veto / imitation models from this data.

The branch preserved Gateway timing and improved Robo payoff, but it did not
beat no-filter and worsened static-defense/threat handling. Next offline work
should target static-defense retention and excessive TECH_ROBO -> STAY_COURSE
before another A/B.
```

## 2026-06-24 Static-Defense Retention Follow-Up

Implemented offline after confirmatory A/B:

```text
Extend active-threat static-defense retention beyond TECH_POWER to tactics that
allow static defense:
  ANTI_RUSH_DEFENSE
  ROBO_TIMING
  ANTI_AIR_RESPONSE
  RECOVERY

Rules:
  pending static slot full -> no retention override
  minerals >= 100 -> keep BUILD_STATIC_DEFENSE
  minerals < 100 and no ready static -> STAY_COURSE to bank
  minerals < 100 and ready static exists -> allow fallback, usually PRODUCE_ARMY
```

Offline replay on confirm6 tactic trajectories:

```text
ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY:
  3 rows would now become STAY_COURSE because minerals were too low and no
  static defense existed.

RECOVERY low-mineral/no-static BUILD_STATIC_DEFENSE rows:
  now bank with STAY_COURSE instead of attempting PRODUCE_ARMY or unaffordable
  static defense.

TECH_POWER TECH_ROBO -> STAY_COURSE:
  unchanged in replay; sampled rows were mostly minerals < 150, so diagnose as
  affordability/banking noise before changing Robo behavior again.
```

Validation:

```text
focused tactic tests: 48 passed
full pytest: 187 passed
check_env: OK
SC2 launched: no
training/PPO: no
```

Next evidence gate:

```text
Run a fresh-dir guarded Power-only A/B only after this offline fix is reviewed.
Promotion remains blocked unless threat rows/static-defense improve without
Gateway timing or Robo payoff regression.
```

## 2026-06-24 Robo Banking Context + Static-Retention Confirm6 A/B

Implemented a diagnostics-only extension in `rl\power_tactic_diagnostics.py`:

```text
robo_banking_filter_contexts
```

It classifies `TECH_ROBO -> STAY_COURSE` filter changes as affordability /
Robo-state buckets. This is intentionally not observation schema metadata and
does not affect runtime behavior.

Focused validation:

```text
tests\test_power_tactic_diagnostics.py
  4 passed

tests\test_tactics.py tests\test_tactic_strategy_policy.py tests\test_power_tactic_diagnostics.py
  52 passed

full pytest:
  188 passed

check_env:
  OK
```

Offline replay on old confirm6 tactic data:

```text
runs\20260624_power_tactic_robo_banking_context_v1\artifacts\power_tactic_diagnostics.txt

TECH_POWER TECH_ROBO -> STAY_COURSE: 26
robo_banking_filter_context:
  Power, TECH_POWER, first_robo_mineral_short: 26
```

Fresh-dir guarded Power-only confirmatory A/B:

```text
guard pid: 26628
SC2 launch path: scripts\evaluate.py only
training/PPO: no

no-filter coverage-teacher:
  run: runs\20260624_152347_20260624_power_ab_static_retention_confirm6_no_filter_v1
  strategy trajectory: data\trajectories\power_ab_static_retention_confirm6_no_filter_strategy_v1
  result: 3W / 3L

tactic-rule:
  run: runs\20260624_152825_20260624_power_ab_static_retention_confirm6_tactic_v1
  strategy trajectory: data\trajectories\power_ab_static_retention_confirm6_tactic_strategy_v1
  result: 3W / 1T / 2L
```

Diagnostic artifacts:

```text
runs\20260624_power_ab_static_retention_confirm6_tactic_timeline_v1\artifacts\tactic_timeline.txt
runs\20260624_power_ab_static_retention_confirm6_tactic_timeline_v1\artifacts\tactic_timeline.json

runs\20260624_power_ab_static_retention_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics.txt
runs\20260624_power_ab_static_retention_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics.json
runs\20260624_power_ab_static_retention_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics_no_filter.txt
runs\20260624_power_ab_static_retention_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics_no_filter.json

runs\20260624_strategy_outcome_power_ab_static_retention_confirm6_v1\artifacts\strategy_outcomes.txt
runs\20260624_strategy_outcome_power_ab_static_retention_confirm6_v1\artifacts\strategy_outcomes.json
```

Source-level outcome summary:

```text
Gateway:
  no-filter first ADD_GATEWAYS=91.4s, count=25
  tactic-rule first ADD_GATEWAYS=91.4s, count=15

Robo:
  no-filter ready/Observer/Immortal = 5/6, 3/6, 0/6
  tactic-rule ready/Observer/Immortal = 5/6, 4/6, 2/6
  tactic Observer delay is worse: 82.9s avg after ready Robo vs no-filter 34.3s

Threat/static:
  no-filter base_threat_rows=46, BUILD_STATIC_DEFENSE=43
  tactic-rule base_threat_rows=52, BUILD_STATIC_DEFENSE=6

Filter changes:
  tactic training_filter_change_rows=82
  TECH_POWER TECH_ROBO -> STAY_COURSE: 26
  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 14
  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> STAY_COURSE: 7
```

Decision:

```text
Do not promote the tactic filter.
Do not collect tactic-aware data from this run.
Do not train action-outcome / veto / imitation models yet.

The tactic-rule run improves raw results and Immortal payoff but still worsens
active-threat handling: fewer static defense actions, more threat rows, and a
non-sparse filter-change profile. The new Robo banking classification shows
the largest TECH_ROBO -> STAY_COURSE bucket is mineral-short banking, so the
next runtime work should not loosen first-Robo affordability. Diagnose and
narrow static-defense suppression under active threat.
```

必需产物：

```text
eval.jsonl
summary.json
army trajectory
strategy trajectory
tactic diagnostics
power tactic diagnostics
strategy outcome diagnostics
```

推广门槛：

```text
revised 结果至少不弱于 no-filter。
Gateway timing 不能显著晚于 no-filter。
first TECH_ROBO 不能显著晚于 no-filter。
ready_robo 后 Observer / Immortal payoff 明显改善。
static-defense suppression 不会让 base threat 持续更久。
filter changes 稀疏且可解释。
```

## 2026-06-24 Pending Static Wait Follow-Up

Added `static_defense_filter_contexts` to Power tactic diagnostics. This is
diagnostics-only and does not add tactic metadata to observation schema.

Artifact:

```text
runs\20260624_power_ab_static_retention_confirm6_static_context_v2\artifacts\power_tactic_diagnostics.txt
runs\20260624_power_ab_static_retention_confirm6_static_context_v2\artifacts\power_tactic_diagnostics.json
```

Key finding:

```text
no_static_affordable: 0

pending_static_waiting rows:
  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 2
  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> TECH_ROBO: 2
  RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 1
```

Minimal opt-in runtime fix:

```text
If base_under_threat > 0 and proposed action is BUILD_STATIC_DEFENSE:
  pending_static_defense >= cap and ready_static_defense == 0:
    STAY_COURSE

  pending_static_defense >= cap and ready_static_defense > 0:
    allow previous fallback
```

Offline replay against the old confirm6 tactic trajectory:

```text
old_after != new_after rows: 5
all five rows had:
  base_under_threat=1
  pending_static_defense=1
  ready_static_defense=0
```

Validation:

```text
focused tactic/power tests: 56 passed
full pytest: 192 passed
check_env: OK
SC2 launched: no
training/PPO: no
```

Decision:

```text
This is still only an opt-in, offline-supported runtime fix. Do not promote,
train, or collect tactic-aware data yet. Next guarded A/B should focus on
whether pending-static waiting reduces TECH_ROBO/PRODUCE_ARMY under threat
without regressing Gateway timing or Robo payoff.
```

## Pending Static Wait Confirmatory A/B

Fresh-dir guarded Power-only A/B:

```text
no-filter coverage-teacher:
  run: runs\20260624_155710_20260624_power_ab_pending_static_wait_confirm6_no_filter_v1
  strategy trajectory: data\trajectories\power_ab_pending_static_wait_confirm6_no_filter_strategy_v1
  result: 1W / 2T / 3L

tactic-rule:
  run: runs\20260624_160251_20260624_power_ab_pending_static_wait_confirm6_tactic_v1
  strategy trajectory: data\trajectories\power_ab_pending_static_wait_confirm6_tactic_strategy_v1
  result: 1W / 1T / 4L
```

Diagnostics:

```text
runs\20260624_power_ab_pending_static_wait_confirm6_tactic_timeline_v1\artifacts\tactic_timeline.txt
runs\20260624_power_ab_pending_static_wait_confirm6_tactic_timeline_v1\artifacts\tactic_timeline.json

runs\20260624_power_ab_pending_static_wait_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics.txt
runs\20260624_power_ab_pending_static_wait_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics.json
runs\20260624_power_ab_pending_static_wait_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics_no_filter.txt
runs\20260624_power_ab_pending_static_wait_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics_no_filter.json

runs\20260624_strategy_outcome_power_ab_pending_static_wait_confirm6_v1\artifacts\strategy_outcomes.txt
runs\20260624_strategy_outcome_power_ab_pending_static_wait_confirm6_v1\artifacts\strategy_outcomes.json
```

Summary:

```text
Gateway:
  no-filter first ADD_GATEWAYS: first=91.4s avg=97.1s count=28
  tactic-rule first ADD_GATEWAYS: first=91.4s avg=121.9s count=15

Robo:
  no-filter ready/Observer/Immortal: 3/6, 3/6, 0/6
  tactic-rule ready/Observer/Immortal: 5/6, 5/6, 1/6
  Observer delay after ready Robo worsened slightly: 73.1s vs 64.8s.

Threat/static:
  no-filter base_threat_rows: 33
  tactic-rule base_threat_rows: 72
  no-filter BUILD_STATIC_DEFENSE: 33
  tactic-rule BUILD_STATIC_DEFENSE: 15
  tactic-rule threat actions: STAY_COURSE=32, PRODUCE_ARMY=24,
    BUILD_STATIC_DEFENSE=15, BOOST_WORKERS=1

Filter:
  training_filter_change_rows: 85
  TECH_POWER TECH_ROBO -> STAY_COURSE: 20, all first_robo_mineral_short
  BUILD_STATIC_DEFENSE suppression remains high:
    ready_static_low_minerals -> PRODUCE_ARMY: 19
    pending_static_with_ready -> PRODUCE_ARMY: 5
    pending_static_waiting -> STAY_COURSE: 4
```

Decision:

```text
Pending-static wait did not validate online. It improved Robo ready/Observer
coverage but regressed match result, Gateway volume, and active-threat defense.

Do not promote the tactic filter.
Do not collect tactic-aware data from this run.
Do not train action-outcome / veto / imitation models yet.
Do not run PPO.
Keep default rule/off runtime unchanged.
```

Next offline diagnostic target:

```text
Add or run an active-threat outcome slice for BUILD_STATIC_DEFENSE filter
changes. The key question is not just why static was suppressed, but whether
the selected fallback clears threat by +30/+60/+90/+120s.

Focus rows:
  ready_static_low_minerals -> PRODUCE_ARMY
  pending_static_with_ready -> PRODUCE_ARMY
  no_static_mineral_short -> STAY_COURSE

Only consider a new opt-in runtime change if that slice shows a clear static
defense payoff gap and does not threaten Gateway/Robo timing.
```

## Active-Threat Outcome Slice

Implemented diagnostics:

```text
rl\active_threat_outcome_diagnostics.py
scripts\diagnose_active_threat_outcomes.py
tests\test_active_threat_outcome_diagnostics.py
```

Focused validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_active_threat_outcome_diagnostics.py -q
2 passed

.\.venv\Scripts\python.exe -m pytest tests\test_active_threat_outcome_diagnostics.py tests\test_power_tactic_diagnostics.py tests\test_strategy_outcome_diagnostics.py -q
12 passed

.\.venv\Scripts\python.exe -m pytest -q
194 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK
```

Artifact:

```text
runs\20260624_active_threat_outcome_pending_static_wait_v1\artifacts\active_threat_outcomes.txt
runs\20260624_active_threat_outcome_pending_static_wait_v1\artifacts\active_threat_outcomes.json
```

Key result on `power_ab_pending_static_wait_confirm6_tactic_strategy_v1`:

```text
active_threat_filter_rows: 41

ANTI_AIR_RESPONSE ready_static_low_minerals -> PRODUCE_ARMY:
  count=11
  +60s: threat_persisted=11/11, army_count_delta=-7.0
  +120s: threat_persisted=5/11, threat_cleared=6/11

RECOVERY ready_static_low_minerals -> PRODUCE_ARMY:
  count=8
  +60s: threat_cleared=6/8
  +120s: threat_cleared=8/8

ANTI_AIR_RESPONSE pending_static_waiting -> STAY_COURSE:
  count=4
  static_defense_increased=4/4 by +30s
  threat_persisted=4/4 through +120s

TECH_POWER no_static_mineral_short -> STAY_COURSE:
  count=6
  +60s: threat_cleared=6/6

RECOVERY no_static_mineral_short -> STAY_COURSE:
  count=4
  +60s: threat_cleared=4/4
```

Development route correction:

```text
Freeze current tactic-rule runtime. Stop adding broad guardrails based only on
aggregate A/B counters.

The bad bucket is tactic-specific:
  ANTI_AIR_RESPONSE + ready_static_low_minerals -> PRODUCE_ARMY

The same context under RECOVERY is acceptable by +120s, so a global
ready-static rule would likely overcorrect. The next candidate fix must be
narrow, opt-in, and limited to anti-air active-threat handling after inspecting
the exact timelines.
```

## Anti-Air Ready-Static Banking Follow-Up

Implemented after the active-threat outcome slice, offline only:

```text
No SC2 launch.
No training / PPO.
No tactic-aware data collection.
Default rule/off path unchanged.
```

Runtime change:

```text
Inside explicit opt-in tactic filtering only:

If proposed action is BUILD_STATIC_DEFENSE under active base threat, return
STAY_COURSE only when:

  tactic_id == ANTI_AIR_RESPONSE
  base_under_air_threat > 0
  ready_static_defense > 0
  pending_static_defense <= 0
  minerals < 100

Ground-only threat with ready static still falls back to PRODUCE_ARMY.
RECOVERY ready_static_low_minerals behavior is unchanged.
```

Offline replay against the previous pending-static wait tactic A/B trajectory:

```text
input:
  data\trajectories\power_ab_pending_static_wait_confirm6_tactic_strategy_v1

changed_rows=11

All changed rows:
  Power / ANTI_AIR_RESPONSE / BUILD_STATIC_DEFENSE
  recorded PRODUCE_ARMY -> current STAY_COURSE

Files:
  20260624_160322_AcropolisLE_Hard_Terran_Power_002.jsonl: 2
  20260624_160425_AcropolisLE_Hard_Terran_Power_003.jsonl: 2
  20260624_160510_AcropolisLE_Hard_Terran_Power_004.jsonl: 7
```

Validation:

```text
focused tactic tests:
  53 passed

active-threat / power / strategy diagnostics focused tests:
  12 passed

full pytest:
  196 passed

check_env:
  OK
```

Decision:

```text
This repair is narrow enough to continue the branch, but it is still only an
opt-in tactic-rule change. It does not justify promotion, tactic-aware data
collection, or training by itself.

Next step should be a guarded fresh-dir Power-only A/B only if we need online
confirmation. Required diagnostics afterward:
  tactic timeline
  power tactic diagnostics
  strategy outcome diagnostics
  active-threat outcome slice if threat rows remain worse
```

## Anti-Air Ready-Static Guarded A/B Result

The guarded fresh-dir Power-only A/B did not validate the current tactic-rule
branch.

```text
guard pid: 26628
SC2 launch path: scripts\evaluate.py only
training/PPO/data collection: no
default rule/off path: unchanged
```

Results:

```text
no-filter coverage-teacher:
  run: runs\20260624_164350_20260624_power_ab_anti_air_ready_static_no_filter_v1
  strategy trajectory: data\trajectories\power_ab_anti_air_ready_static_no_filter_strategy_v1
  result: 2W / 2T / 2L

tactic-rule:
  run: runs\20260624_164917_20260624_power_ab_anti_air_ready_static_tactic_v1
  strategy trajectory: data\trajectories\power_ab_anti_air_ready_static_tactic_strategy_v1
  result: 0W / 0T / 6L
```

Diagnostics:

```text
runs\20260624_power_ab_anti_air_ready_static_tactic_timeline_v1\artifacts\tactic_timeline.txt
runs\20260624_power_ab_anti_air_ready_static_power_tactics_v1\artifacts\power_tactic_diagnostics.txt
runs\20260624_power_ab_anti_air_ready_static_power_tactics_v1\artifacts\power_tactic_diagnostics_no_filter.txt
runs\20260624_strategy_outcome_power_ab_anti_air_ready_static_v1\artifacts\strategy_outcomes.txt
runs\20260624_active_threat_outcome_anti_air_ready_static_v1\artifacts\active_threat_outcomes.txt
```

Summary:

```text
Gateway:
  both sides first ADD_GATEWAYS=91.4s
  no-filter ADD_GATEWAYS=33
  tactic-rule ADD_GATEWAYS=19

Robo:
  no-filter ready/Observer/Immortal = 4/6, 4/6, 0/6
  tactic-rule ready/Observer/Immortal = 6/6, 6/6, 0/6

Threat/static:
  no-filter threat actions:
    BUILD_STATIC_DEFENSE=26, STAY_COURSE=7, PRODUCE_ARMY=1
  tactic-rule threat actions:
    STAY_COURSE=35, PRODUCE_ARMY=27, BUILD_STATIC_DEFENSE=19

Filter:
  training_filter_change_rows=66
  BUILD_STATIC_DEFENSE=-46
  PRODUCE_ARMY=+26
  STAY_COURSE=+26
  TECH_ROBO=-3
```

Anti-air slice:

```text
The specific new anti-air STAY_COURSE bucket was small and locally better:

  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> STAY_COURSE
  ready_static_low_minerals
  count=2
  +120s threat_cleared=2/2

Remaining old fallback rows:

  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY
  ready_static_low_minerals
  count=2
  +120s threat_persisted=2/2

This supports the narrow anti-air diagnosis, but not the broader tactic-rule
runtime.
```

Decision:

```text
Do not promote.
Do not collect tactic-aware data.
Do not train.
Do not patch immediately from this online result.

The next offline target is no longer ANTI_AIR_RESPONSE. It is the broader
active-threat suppression surface:

  RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY ready_static_low_minerals
  RECOVERY / TECH_POWER pending_static_waiting -> STAY_COURSE
  RECOVERY TECH_ROBO -> PRODUCE_ARMY after ready Robo

Any next runtime change must first show offline outcome evidence and a replay
diff with a small, explainable changed-row set.
```

## Active-Threat Suppression Slice

Implemented after the anti-air ready-static A/B showed tactic-rule 0W/6L.

Files:

```text
rl\active_threat_suppression_diagnostics.py
scripts\diagnose_active_threat_suppression.py
tests\test_active_threat_suppression_diagnostics.py
```

Purpose:

```text
Offline-only diagnostics for RECOVERY / TECH_POWER rows where tactic filtering
rewrites:

  BUILD_STATIC_DEFENSE -> PRODUCE_ARMY / STAY_COURSE
  TECH_ROBO -> PRODUCE_ARMY / STAY_COURSE

The report groups by source, opponent build, tactic, original action, selected
action, context, and threat state. It emits +30/+60/+90/+120s outcomes,
per-file timelines, and replay-only pass-through candidate impact.
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_active_threat_suppression.py data\trajectories\power_ab_anti_air_ready_static_no_filter_strategy_v1 data\trajectories\power_ab_anti_air_ready_static_tactic_strategy_v1 --show-files --show-timeline --json-output runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\active_threat_suppression.json --text-output runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\active_threat_suppression.txt
```

Artifacts:

```text
runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\active_threat_suppression.txt
runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\active_threat_suppression.json
```

Key results:

```text
no-filter:
  target_suppression_rows=0
  result=2W / 2T / 2L

tactic-rule:
  target_suppression_rows=50
  filter_change_rows=66
  result=0W / 0T / 6L

replay-only pass-through candidate:
  BUILD_STATIC_DEFENSE +42
  TECH_ROBO +8
  PRODUCE_ARMY -29
  STAY_COURSE -21
  only 6/50 candidate rows immediately executable
```

Main outcomes:

```text
RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY
ready_static_low_minerals / ground_threat
count=18
  +30s threat_persisted=15/18
  +60s threat_persisted=10/18
  +90s threat_persisted=5/18
  +120s threat_persisted=2/18
  static / army / ready_robo / observer deltas are negative by +120s.

TECH_POWER BUILD_STATIC_DEFENSE -> STAY_COURSE
pending_static_waiting / ground_threat
count=5
  static_defense_increased=5/5 by +30s
  threat_persisted=3/5 through +120s.

RECOVERY BUILD_STATIC_DEFENSE -> STAY_COURSE
pending_static_waiting / ground_threat
count=4
  threat_persisted=4/4 through +90s
  threat_persisted=2/4 at +120s.

RECOVERY TECH_ROBO -> PRODUCE_ARMY
first_robo_mineral_short / no_threat
count=4
  candidate_executable=0/4
  no pending / ready Robo payoff within +120s.

TECH_POWER TECH_ROBO -> STAY_COURSE
first_robo_mineral_short / no_threat
count=4
  candidate_executable=0/4
  pending_robo_seen=4/4 by +60s, ready_robo_seen=4/4 by +120s,
  but no Observer / Immortal payoff.
```

Interpretation:

```text
This does not support an immediate runtime patch.

The largest static-defense suppression bucket is bad early but not a clean
global retain-static rule, because many rows eventually clear threat. Waiting
for pending static also does not solve the active-threat problem by itself.

The TECH_ROBO rows are not active-threat rows; they are no-threat,
mineral-short first-Robo contexts. TECH_POWER banking eventually builds Robo,
while RECOVERY TECH_ROBO -> PRODUCE_ARMY remains suspicious but too late and
resource/supply constrained to patch blindly.
```

Validation:

```text
focused suppression tests: 2 passed
active-threat / power / strategy diagnostics focused tests: 14 passed
full pytest: 201 passed
check_env: OK
SC2 launched: no
training/PPO/data collection: no
```

Next gate:

```text
Keep tactic-rule runtime frozen. Only consider another opt-in runtime change if
the candidate replay diff is narrower than this 50-row surface and shows a
clear +30/+60/+90/+120s outcome gap without threatening Gateway/Robo timing.
```

## 2026-06-25 Strategy Candidate Audit Gate

Implemented offline-only candidate promotion gate:

```text
rl\strategy_candidate_audit.py
scripts\audit_strategy_candidate.py
tests\test_strategy_candidate_audit.py
```

Purpose:

```text
Compare a frozen baseline strategy trajectory directory with a candidate
strategy trajectory directory before considering promotion, data collection,
training, or another runtime patch.
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\audit_strategy_candidate.py <baseline-strategy-dir> <candidate-strategy-dir> --json-output <run-artifacts-dir>\promotion_gate.json --text-output <run-artifacts-dir>\strategy_candidate_audit.txt
```

The first gate uses `StrategyOutcomeDiagnostics` plus the new executor metadata.
It blocks promotion when:

```text
candidate result score is worse than baseline
base_threat_rows increases
ADD_GATEWAYS count regresses
TECH_ROBO count regresses
BUILD_STATIC_DEFENSE count regresses
strategy_execution_blocker counts increase
```

It warns, but does not yet block, when:

```text
candidate_has_filter_changes
```

Result scoring accepts both modern SC2 result names and older bare names:

```text
Result.Victory / Victory
Result.Defeat / Defeat
Result.Tie / Tie
```

Current scope:

```text
No SC2 launch.
No training / PPO.
No tactic-aware data collection.
Default --strategy-policy rule / --strategy-tactic-mode off unchanged.
This does not replace guarded evaluation; it is a pre-promotion offline gate.
```

Next development target:

```text
P3 replay-only candidate:
  recorded trajectory -> candidate action -> changed rows -> executable rows
  -> +30/+60/+90/+120s outcome slice
```

Validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_candidate_audit.py -q
4 passed

.\.venv\Scripts\python.exe -m pytest tests\test_strategy_replay_candidate.py -q
4 passed

.\.venv\Scripts\python.exe -m pytest -q
209 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK

Get-Process SC2,SC2_x64 -ErrorAction SilentlyContinue
no process output
```

## 2026-06-25 Strategy Replay Candidate Diagnostics

Implemented first P3 replay-only candidate diagnostic:

```text
rl\strategy_replay_candidate.py
scripts\diagnose_strategy_replay_candidate.py
tests\test_strategy_replay_candidate.py
```

Purpose:

```text
Before changing runtime, replay a candidate action against recorded trajectory
rows and summarize the changed-row surface:

  machine-readable gate_decision
  recorded selected action
  candidate action
  changed rows
  immediate candidate executable rows
  action delta
  +30/+60/+90/+120s recorded outcome slice
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_strategy_replay_candidate.py <strategy-trajectory-dir> --show-files --show-timeline --json-output <run-artifacts-dir>\strategy_replay_candidate.json --text-output <run-artifacts-dir>\strategy_replay_candidate.txt
```

Current candidate source:

```text
before_filter:
  candidate_action = strategy_action_before_tactic_filter
  recorded_action = strategy_action_after_tactic_filter / selected action
```

This first mode is designed for the frozen tactic-rule branch: it answers what
would happen if changed rows were passed through to the original strategy action
instead of the tactic-selected action. It is still offline evidence only; it
does not simulate the alternate future, but it tells us whether the candidate
surface is narrow, executable, and associated with a clear recorded outcome gap.

Runtime gate:

```text
Do not patch runtime unless replay diagnostics show:
  changed rows are narrow
  candidate rows are mostly executable
  affected contexts are specific
  recorded outcome gap is clear at +30/+60/+90/+120s
  default rule/off path remains unchanged
```

Current anti-air ready-static gate result:

```text
inputs:
  baseline audit:
    data\trajectories\power_ab_anti_air_ready_static_no_filter_strategy_v1
  candidate audit / replay:
    data\trajectories\power_ab_anti_air_ready_static_tactic_strategy_v1

artifacts:
  runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\promotion_gate.json
  runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\strategy_candidate_audit.txt
  runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\strategy_replay_candidate.json
  runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\strategy_replay_candidate.txt

promotion audit:
  promotable=false
  baseline_result_score=0.333
  candidate_result_score=0.000
  base_threat_rows_delta=48
  filter_change_rows=66

replay gate:
  gate_decision=hold_runtime_patch
  runtime_patch_candidate=false
  changed_rows=66
  candidate_executable=15/66
  largest_group=18 rows, 0/18 executable

largest group:
  RECOVERY PRODUCE_ARMY -> BUILD_STATIC_DEFENSE
  ready_static_low_minerals / ground_threat

decision:
  no runtime patch
  no promotion
  no tactic-aware data collection
  no training / PPO
```

## 训练路线

不要使用未经 outcome / guarded A/B 支持的 tactic-aware 数据，直接训练
`observation -> strategy_action` 模型。

后续推荐顺序：

```text
1. Action outcome / action critic：
   observation + proposed_action -> landed / delayed / harmful / blocked

2. Veto / filter model：
   observation + proposed_action -> pass / veto / safer fallback

3. Tactic-aware strategy imitation：
   在 outcome diagnostics 和 guarded comparisons 稳定后，可以小规模考虑。

4. PPO：
   在 rule baseline、strategy baseline、reward design、SC2 environment 边界稳定后考虑。
```

## 验证清单

交接前运行：

```text
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

阶段 1 实现验收：

```text
strategy outcome diagnostics focused tests 通过。
full pytest 通过。
不启动 SC2。
不运行与本阶段无关的 PPO / training 命令。
不为 tactic metadata 新增 observation schema 字段。
默认 rule/no-op 行为保持不变。
```
