# CODEX.md

面向 coding agent 的简洁交接文档。

## 先读顺序

按下面顺序阅读：

1. `doc\CODEX.md`：当前事实、安全约束、下一步工作。
2. `doc\README.md`：项目概览和常用命令。
3. `doc\STATE.md`：实验账本和最新状态。
4. `doc\DEVELOPMENT_PLAN.md`：当前开发主线，冻结 tactic-rule，execution observability、candidate audit、replay diagnostics 已落地。
5. `doc\STRATEGY_OUTCOME_PLAN.md`：strategy action outcome diagnostics、candidate audit 和 replay-only 计划。
6. `doc\TACTIC_POOL_PLAN.md`：AIBuild / TacticSpec 背景计划。

历史背景按需阅读：

- `doc\archive\STRATEGY_EXPANSION_PLAN.md`：从 army-only 扩展到低频 strategy layer 的历史路线。

## 当前事实

- 项目根目录：`D:\opus\data\raw\alpaca-gpt4\sc2\sc2-ai-bot`
- 当前目录是 git 仓库；不要执行破坏性 git 操作。
- 当前 bot 是 Protoss rule bot，加高层 army policy 基础设施，以及显式 opt-in 的低频 strategy slot。
- 当前学习策略边界：army-level 决策和显式 opt-in 的低频 strategy imitation。默认 runtime 仍保持 rule/no-op，除非显式选择策略。
- 当前 army action space：5 个离散动作。
- 当前 strategy action space：8 个宏观意图，默认 `STAY_COURSE`。
- 当前 army observation schema：v3，26 个数值特征。
- 当前 strategy observation schema：`strategy_v2`，40 个数值特征。
- 当前测试：`.\.venv\Scripts\python.exe -m pytest -q` -> `209 passed`。
- 最新推荐 strategy 数据：`strategy_coverage_teacher_v2_pending_tuned_v1`，strategy-v2 覆盖健康，低频动作没有低计数问题。
- PPO 未实现。
- `coverage-teacher` 用于数据采集，不是强 baseline strategy。
- 旧 checkpoint 是归档或诊断产物；当前 runtime 会故意拒绝不兼容的 schema / action metadata。
- 官方 AI `AIBuild` 已接入 eval / data-collection metadata；本地 `TacticSpec` 雏形已存在，且只通过显式 opt-in 启用。
- revised Power tactic filter 只修复了部分 Robo timing，没有赢过 no-filter coverage-teacher；第一层 guardrail A/B 仍是 1W/2L vs 1W/2L，不能推广。
- 离线 `StrategyOutcomeDiagnostics` 已实现，可衡量每个 strategy action 在 +30/+60/+90/+120s 后是否产生预期结果。
- 最新离线复查已定位 `power_ab_guardrail_tactic_strategy_v1` 第三局 late `ADD_GATEWAYS`：102.9s 的 `ADD_GATEWAYS -> BOOST_WORKERS` 发生在 `pending_gateways=2` 时，后续 ready Gateways 已到 4；571.4s 是后期产能被打掉后的重建标签，不是持续 early-Gateway 压制。
- `StrategyOutcomeDiagnostics` 已扩展 ready-Robo payoff 分类：三局 tactic guardrail 都补到 Observer，Immortal blocker 均为 `resource_or_supply_blocked`，不是 `robo_not_idle` 或 `action_not_triggered`。
- 最新最小 runtime 修复：仅在显式 opt-in tactic filter 下加入 guardrail-first 保护。ready Robo + 已有 Observer + 无 Immortal + 无基地威胁时仍可偏向第一只 Immortal；underbuilt Gateway 优先保留 `ADD_GATEWAYS`；active base threat 下的 `TECH_POWER` static defense 优先保留或留矿；`ANTI_AIR_RESPONSE` / `ANTI_RUSH_DEFENSE` / `ROBO_TIMING` / `RECOVERY` 在 active threat 且无 ready/pending static defense 时会保留或留矿 `BUILD_STATIC_DEFENSE`；`TECH_POWER` 在无 ready/pending Robo、Core ready、无威胁且资源足够时会从 `STAY_COURSE` 抢回第一座 Robo；`TECH_POWER` 在 Robo 已 ready/pending 且 Gateway 低于 `own_bases * 4` 时允许最多 2 个 pending Gateway。默认 `rule/off` 不变。
- first-Immortal bias guarded Power-only A/B 已完成：no-filter coverage-teacher 0W/3L，tactic-rule 0W/3L；tactic 第 1 局产出 Immortal 并存活更久，但 source-level first `ADD_GATEWAYS` 从 91.4s 退到 422.9s，static-defense suppression 仍让威胁持续。
- 最新离线 guardrail 修复已完成：`TECH_POWER` 在 active base threat 且 static-defense slot 可用时保留/留矿 `BUILD_STATIC_DEFENSE`；first-Immortal bias 在 Gateway 仍低于 `own_bases * 4` 时不抢 `ADD_GATEWAYS`。
- Gateway/static-defense retention guarded Power-only A/B 已完成：no-filter 2W/1L，tactic-rule 0W/3L；diagnostics 显示 tactic filter 仍过度压 `TECH_ROBO`，第 3 局没有 ready Robo。
- Initial-Robo precedence guarded Power-only A/B 已完成：no-filter 1W/1T/1L，tactic-rule 0W/1T/2L；tactic 三局都有 ready Robo，两局产出 Immortal，但 `ADD_GATEWAYS` 仍显著滞后（source first 377.1s vs no-filter 91.4s）。
- Midgame Gateway cap guarded Power-only A/B 已完成：no-filter coverage-teacher 1W/2L，tactic-rule 1W/2L；tactic 的 first `ADD_GATEWAYS` 从 initial-Robo A/B 的 377.1s 改善到 171.4s，但仍晚于 no-filter 91.4s，且仍有 71 个 filter changes、1 局没有 ready Robo、1 局 ready Robo 后没落地 Immortal。不要推广，不要采 tactic-aware 数据。
- 最新离线修复：first-Robo banking guard 扩展为只在显式 opt-in tactic filter 下生效。无 ready/pending Robo、Core ready、无威胁时，`TECH_POWER` 会把 `STAY_COURSE` / `FORGE_UPGRADES` / `ADD_GATEWAYS` / `PRODUCE_ARMY` / `BOOST_WORKERS` 等可能打散第一座 Robo 节奏的动作改为 `TECH_ROBO` 或留矿 `STAY_COURSE`；安全 `EXPAND` 仍按原测试约束保留。focused tactic tests 39 passed，full pytest 178 passed，`scripts\check_env.py` OK。
- First-Robo banking guard guarded Power-only A/B 已完成：no-filter coverage-teacher 1W/1T/1L，tactic-rule 2W/1T/0L；tactic 三局都有 ready Robo + Observer，1 局 Immortal 落地，base_threat_rows 明显低于 no-filter。但 first `ADD_GATEWAYS` 退到 411.4s（no-filter 91.4s），`ADD_GATEWAYS` count 6 vs 13，仍不适合采 tactic-aware 训练数据。
- Gateway preservation follow-up 已完成：`SAFE_MACRO` 在 120s 前、无 ready Gateway、已有 2 pending Gateway、minerals>=250、vespene<100、无威胁时，允许第 3 个 pending Gateway；Robo gas ready 后或 active base threat 下仍保持 2 pending cap。focused tactic tests 44 passed，full pytest 183 passed，`check_env.py` OK。
- Gateway preservation guarded Power-only A/B 已完成：no-filter coverage-teacher 1W/2L，tactic-rule 2W/1T/0L；tactic first `ADD_GATEWAYS` 回到 91.4s，count 11 vs no-filter 13，同时三局均有 ready Robo，2/3 有 Observer，1/3 有 Immortal。仍有一局 Victory 缺 Observer（resource_or_supply_blocked）和一个 Tie threat_rows=17，所以继续不要采 tactic-aware 训练数据。
- Gateway preservation 6-valid confirmatory A/B 已完成：no-filter 有效样本 2W/1T/3L（另有 1 个 `NO_RESULT code=1` top-up 前异常尝试），tactic-rule 2W/1T/3L；first `ADD_GATEWAYS` 都是 91.4s，tactic ready Robo/Observer/Immortal payoff 更好（ready 5/6、Observer 5/6、Immortal 2/6 vs no-filter ready 4/6、Observer 4/6、Immortal 0/6），但 tactic base_threat_rows 44 vs no-filter 39、BUILD_STATIC_DEFENSE 13 vs 39、filter_change_rows 75，不能推广或训练。
- Confirm6 后的离线 static-defense retention follow-up 已完成：扩展 active-threat static retention 到非 `TECH_POWER` 的防守/恢复 tactics；confirm6 replay 显示 `ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY` 中 3 行会改为留矿，`RECOVERY` 低矿无 static 行也会留矿。
- `PowerTacticDiagnostics` 已扩展 `robo_banking_filter_contexts`，可把 `TECH_ROBO -> STAY_COURSE` 区分为 `first_robo_mineral_short`、`first_robo_affordable`、`pending_robo_cap`、`ready_robo_already_exists` 等。旧 confirm6 和新 confirm6 中 26 条 `TECH_POWER TECH_ROBO -> STAY_COURSE` 都是 `first_robo_mineral_short`，不是矿气已够却被误压。
- Static-defense retention fresh-dir confirm6 A/B 已完成：no-filter coverage-teacher 3W/3L，tactic-rule 3W/1T/2L；tactic first `ADD_GATEWAYS` 仍为 91.4s，Robo payoff 改善到 ready 5/6、Observer 4/6、Immortal 2/6（no-filter ready 5/6、Observer 3/6、Immortal 0/6）。但 tactic base_threat_rows=52 vs no-filter 46，`BUILD_STATIC_DEFENSE`=6 vs 43，filter_change_rows=82，且仍有一局无 ready Robo、一局 ready Robo 后 Observer 未触发。不要推广，不采 tactic-aware 数据，不训练。
- `PowerTacticDiagnostics` 已继续扩展 `static_defense_filter_contexts`，可区分 active-threat `BUILD_STATIC_DEFENSE` 改写时的 `no_static_mineral_short`、`ready_static_low_minerals`、`pending_static_waiting`、`pending_static_with_ready` 等。新诊断显示没有 `no_static_affordable`；5 行 `pending_static_waiting` 仍被旧 filter 改到 `PRODUCE_ARMY/TECH_ROBO`。
- 最新最小 runtime 修复：只在显式 opt-in tactic filter 下，active threat 且 `pending_static_defense >= cap`、`ready_static_defense == 0` 时返回 `STAY_COURSE` 等待 static 落地；已有 ready static 时仍允许 fallback 到产兵。默认 `rule/off` 不变。
- Pending-static wait guarded Power-only confirmatory A/B 已完成：no-filter coverage-teacher 1W/2T/3L，tactic-rule 1W/1T/4L。tactic Robo payoff 改善到 ready 5/6、Observer 5/6、Immortal 1/6（no-filter 3/6、3/6、0/6），但 observer delay 更慢，first `ADD_GATEWAYS` 平均退到 121.9s（no-filter 97.1s），`ADD_GATEWAYS` count 15 vs 28，base_threat_rows 72 vs 33，`BUILD_STATIC_DEFENSE` 15 vs 33，filter_change_rows 85。不要推广，不采 tactic-aware 数据，不训练。
- 当时验证：focused tactic tests 53 passed，active-threat / power / strategy diagnostics focused tests 12 passed，full pytest 196 passed，`check_env.py` OK。该 SC2 A/B 使用 guard pid 26628，只通过 `scripts\evaluate.py` / `scripts\safe_launch.py` 启动；anti-air 窄修复本身未启动 SC2，未运行 PPO/训练。
- 开发路线修正：冻结当前 tactic-rule runtime，不再用“小 patch + 小样本 A/B”连续追噪声；先用离线 outcome slice 证明某个 tactic/context 的 fallback 有害，再做一个可回滚、显式 opt-in、默认不变的最小 runtime 修复。
- 新增 `rl/active_threat_outcome_diagnostics.py`、`scripts/diagnose_active_threat_outcomes.py`、`tests/test_active_threat_outcome_diagnostics.py`。它按 active threat 下 `BUILD_STATIC_DEFENSE -> <other>` 的 tactic/context 分组，看 +30/+60/+90/+120s 的 threat/static/army outcome。
- 最新 active-threat outcome 产物：`runs\20260624_active_threat_outcome_pending_static_wait_v1\artifacts\active_threat_outcomes.txt` 和 `.json`。关键发现：`ANTI_AIR_RESPONSE ready_static_low_minerals -> PRODUCE_ARMY` 11 行在 +60s 仍 11/11 threat persisted，+120s 仍 5/11 persisted，且 army_count_delta 为负；`RECOVERY ready_static_low_minerals -> PRODUCE_ARMY` 到 +120s 8/8 cleared，不能一刀切。
- 最新最小 runtime 修复：只在显式 opt-in tactic filter 下，`ANTI_AIR_RESPONSE`、active base threat、`base_under_air_threat > 0`、已有 ready static、无 pending static、minerals < 100 时，把 `BUILD_STATIC_DEFENSE` 的 fallback 从 `PRODUCE_ARMY` 改为 `STAY_COURSE`。离线 replay 只改变旧 pending-static wait tactic A/B 的 11 行坏桶；ground-only threat 和 `RECOVERY` 不变。
- Anti-air ready-static guarded Power-only A/B 已完成：no-filter coverage-teacher 2W/2T/2L，tactic-rule 0W/0T/6L。tactic first `ADD_GATEWAYS` 仍为 91.4s，Robo/Observer 覆盖为 6/6、6/6，但 Immortal 0/6，`BUILD_STATIC_DEFENSE` 仍被压低、`PRODUCE_ARMY` 和 `STAY_COURSE` under threat 仍过多。不要推广，不采 tactic-aware 数据，不训练/PPO。
- 最新 A/B 产物：`runs\20260624_power_ab_anti_air_ready_static_tactic_timeline_v1\artifacts\`、`runs\20260624_power_ab_anti_air_ready_static_power_tactics_v1\artifacts\`、`runs\20260624_strategy_outcome_power_ab_anti_air_ready_static_v1\artifacts\`、`runs\20260624_active_threat_outcome_anti_air_ready_static_v1\artifacts\`。
- 已新增离线 active-threat suppression diagnostics：`rl\active_threat_suppression_diagnostics.py`、`scripts\diagnose_active_threat_suppression.py`、`tests\test_active_threat_suppression_diagnostics.py`。它专门看 `RECOVERY` / `TECH_POWER` 下 `BUILD_STATIC_DEFENSE` 与 `TECH_ROBO` 被 tactic filter 改写为 `PRODUCE_ARMY` / `STAY_COURSE` 后的 +30/+60/+90/+120s outcome，并输出 per-context、per-file timeline 和 replay-only candidate impact。
- 最新 active-threat suppression 产物：`runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\active_threat_suppression.txt` 和 `.json`。输入为 `data\trajectories\power_ab_anti_air_ready_static_no_filter_strategy_v1` 与 `data\trajectories\power_ab_anti_air_ready_static_tactic_strategy_v1`。
- 最新离线结论：no-filter target suppression rows=0；tactic-rule target suppression rows=50。Replay-only pass-through 会把动作计数改为 `BUILD_STATIC_DEFENSE +42`、`TECH_ROBO +8`、`PRODUCE_ARMY -29`、`STAY_COURSE -21`，但只有 6/50 candidate rows 立即可执行。最大坏桶是 `RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY / ready_static_low_minerals / ground_threat`：18 行在 +30s 15/18 threat persisted，+60s 10/18 persisted，+120s 仍 2/18 persisted，且 static / army / Robo payoff 为负。`TECH_POWER` 与 `RECOVERY` 的 `pending_static_waiting -> STAY_COURSE` 虽然让 static 落地，但 +120s 仍分别 3/5、2/4 threat persisted。`TECH_ROBO` suppression 8 行都是 no-threat `first_robo_mineral_short`，不是资源已足的第一座 Robo 被误压；`TECH_POWER TECH_ROBO -> STAY_COURSE` 后续能见到 pending/ready Robo，但仍没有 Observer/Immortal payoff。
- 本轮不加 runtime patch；继续冻结 tactic-rule runtime。下一步只有在 replay diff 足够小、context 足够窄且 outcome 证据清晰时，才考虑新的显式 opt-in 修复。
- 当前实现主线已完成 P1/P2/P3：`StrategyExecutor.execute()` 返回执行结果，新 strategy trajectory 会记录 action 是否尝试、产生了什么 effect、被什么 blocker 卡住；`scripts\audit_strategy_candidate.py` 可比较 baseline/candidate trajectory 并输出 promotion gate；`scripts\diagnose_strategy_replay_candidate.py` 可离线回放 before-filter candidate，统计 changed rows、executable rows、outcome slice，并输出 machine-readable `gate_decision`。
- 最新 anti-air ready-static gate 结论：promotion audit 为 `promotable=false`，replay gate 为 `gate_decision=hold_runtime_patch`、`runtime_patch_candidate=false`、`changed_rows=66`、`candidate_executable=15/66`；最大分组是 `RECOVERY PRODUCE_ARMY -> BUILD_STATIC_DEFENSE / ready_static_low_minerals / ground_threat`，18 行且 0/18 可立即执行。不要从该 surface 做 runtime patch。
- 最新验证：`tests\test_strategy_candidate_audit.py tests\test_strategy_replay_candidate.py` 8 passed；full pytest 209 passed；`scripts\check_env.py` OK；`Get-Process SC2,SC2_x64` 无输出。本轮未启动 SC2，未训练/PPO，未采 tactic-aware 数据。
- 训练 / tactic-aware 数据 / PPO 不是永久禁区：如果 guarded A/B 和 outcome diagnostics 支持，可以采小规模 tactic-aware 数据、训练 action-outcome / veto / imitation 模型；PPO 仍需等 SC2 env、reward、baseline 对照和安全启动边界清楚后再进入。

## 安全规则

用户可能正在同一台机器上工作。不要暴露 SC2 或 Battle.net 窗口。

任何可能启动 SC2 的命令之前，必须先确认或启动 hidden-window guard：

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

只能使用 `scripts/evaluate.py` 或 `scripts/safe_launch.py` 启动对局。除非用户明确要求可见调试，不要运行可见/debug 模式。

## Completed Framework

- rule-based Protoss bot
- safe launch/window guard
- `ArmyPolicy` abstraction
- rule, coverage-teacher, checkpoint-backed, and experimental LLM army policies
- `ArmyMemory`
- low-frequency `StrategyPolicy` interface
- `StrategyAction` v1 macro intent enum
- default `RuleStrategyPolicy` returning `STAY_COURSE`
- `StrategyExecutor` skeleton for expand, gateways, robo, forge/upgrades, static defense, army/worker production
- conservative GG/surrender policy that sends `gg` and leaves only in near-unrecoverable states
- `strategy_v2` observation schema for low-frequency macro decisions, with compatibility defaulting for `strategy_v1` diagnostics
- explicit strategy trajectory recording via `--strategy-trajectory-path` / `--strategy-trajectory-dir`
- opt-in `StrategyCoverageTeacher` via `--strategy-policy coverage-teacher`
- explicit strategy imitation dataset/training/checkpoint support for `strategy_v2` + `StrategyAction`
- checkpoint-backed `RLStrategyPolicy` via `--strategy-policy checkpoint --strategy-checkpoint <path>`
- schema v3 observation
- trajectory recording
- dataset loading
- diagnostics with observation feature stats
- diagnostics `--kind strategy` mode for strategy observation/action coverage
- strategy timing diagnostics for action timing, threat-state actions, TECH_ROBO
  signal timing, pending-repeat checks, and Hard defeat file summaries
- offline strategy agreement diagnostics for comparing recorded labels, current
  coverage-teacher labels, and checkpoint predictions by time/state/file bucket
- official built-in AI `AIBuild` support in `run.py` and guarded evaluation
- `opponent_ai_build` metadata in eval records, experiment config, summary groups,
  and army/strategy trajectory rows
- local `TacticSpec` / `TacticState` scaffolding and `RuleTacticSelector`
  for future explicit tactic-aware strategy policies
- explicit opt-in tactic-aware coverage-teacher mode via
  `--strategy-policy coverage-teacher --strategy-tactic-mode rule`; current
  online filter is Power-targeted and still experimental
- tactic metadata diagnostics via `scripts/diagnose_tactics.py`
- Power-build tactic failure diagnostics via `scripts/diagnose_power_tactics.py`
- Power tactic diagnostics can classify `TECH_ROBO -> STAY_COURSE` Robo-banking
  filter changes by affordability and Robo state
- strategy action outcome diagnostics via `scripts/diagnose_strategy_outcomes.py`
- active-threat suppression diagnostics via
  `scripts/diagnose_active_threat_suppression.py`
- experiment metadata
- model/checkpoint metadata
- observation normalization
- imitation learning
- guarded batch evaluation

PPO missing pieces:

- Gymnasium SC2 environment
- reward implementation
- PPO training script
- PPO checkpoint adapter/evaluator
- curriculum controller

## Army Action Space

```text
0 RALLY
1 ATTACK_MAIN
2 RETREAT_HOME
3 DEFEND_BASE
4 HOLD
```

Economy, production, buildings, and workers remain rule-based.

## Strategy Action Space

This is a separate low-frequency macro layer and does not replace `ArmyAction`.
The default `RuleStrategyPolicy` returns `STAY_COURSE`, preserving the current
rule baseline until a strategy teacher or learned policy is explicitly enabled.

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

## Current Candidate

Training data:

```text
data\trajectories\coverage_teacher_v3_retreat_focused_v3
```

Combined data diagnostics:

```text
runs\20260622_151332_coverage_teacher_v3_retreat_focused_v3_extend1\artifacts\trajectory_diagnostics_combined.json
```

Combined data summary:

```text
72 games
16749 rows
16677 training rows
schema v3 only
rows_defaulted_observation_fields=0
RALLY=5892
ATTACK_MAIN=1468
RETREAT_HOME=33
DEFEND_BASE=805
HOLD=8479
```

Trained candidate:

```text
runs\20260622_154050_imitation_v3_candidate\checkpoints\policy.pt
```

Training metrics:

```text
16677 examples
observation_dim=26
schema v3 only
rows_defaulted_observation_fields=0
all actions present
validation RETREAT_HOME=5/5
```

## Current Comparison

Same-scenario guarded evals, 18 games each:

```text
imitation:
  run: runs\20260622_160753_eval_imitation_v3_candidate_v2
  result: 8 Victory / 8 Defeat / 2 Tie
  actions: RALLY=1991, ATTACK_MAIN=352, RETREAT_HOME=2, DEFEND_BASE=237, HOLD=1589

rule:
  run: runs\20260622_162334_eval_rule_baseline_v3_compare
  result: 6 Victory / 8 Defeat / 4 Tie
  actions: RALLY=3017, ATTACK_MAIN=1521

coverage-teacher:
  run: runs\20260622_163914_eval_coverage_teacher_v3_compare
  result: 5 Victory / 9 Defeat / 4 Tie
  actions: RALLY=1422, ATTACK_MAIN=589, RETREAT_HOME=3, DEFEND_BASE=267, HOLD=2288
```

Conclusion:

- `imitation_v3_candidate` is runnable and not obviously weaker than rule in this sample.
- It is a schema-v3 smoke/comparison baseline.
- It is not yet a stable PPO initializer: online `RETREAT_HOME` remains token-count and `RALLY/HOLD` dominate.

## Current Strategy Data Test

Guarded coverage-strategy collection:

```text
run: runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1
eval: runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\eval.jsonl
trajectory: data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1
diagnostics: runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\strategy_trajectory_diagnostics.json
```

Summary:

```text
18 games
return_code=0 for all games
13 Victory / 3 Defeat / 2 Tie
927 rows
909 training rows
strategy_v2 only
rows_defaulted_observation_fields=0
action_coverage=100%
STAY_COURSE=490
EXPAND=11
ADD_GATEWAYS=69
TECH_ROBO=130
FORGE_UPGRADES=24
BUILD_STATIC_DEFENSE=50
PRODUCE_ARMY=69
BOOST_WORKERS=66
low_count_actions=<none>
```

Conclusion:

- Strategy trajectory recording and diagnostics are healthy.
- `strategy_v2` adds pending macro/upgrade features so teacher labels do not repeat Forge/upgrades while construction or research is already pending.
- `FORGE_UPGRADES` long-run spam is fixed: latest v2 data has 24 one-step Forge labels and no low-count actions.
- A later `strategy_coverage_teacher_v2_pending_robo_tuned_v1` attempt added a Robo minerals gate and was worse: 11 Victory / 2 Defeat / 5 Tie, `TECH_ROBO=15`, `STAY_COURSE=628`; do not treat it as the recommended dataset.
- Next strategy work can move to an explicit strategy imitation training path, but do not reuse the army imitation checkpoint path for strategy checkpoints.

## Current Strategy Imitation Candidate

Previous v1 training run:

```text
runs\20260623_113345_strategy_imitation_v2_candidate
checkpoint: runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt
metrics: runs\20260623_113345_strategy_imitation_v2_candidate\artifacts\metrics.json
```

Training summary:

```text
examples=909
observation_dim=40
schema=strategy_v2 only
rows_defaulted_observation_fields=0
missing_action_names=[]
train_accuracy=0.602
validation_accuracy=0.533
```

Guarded online eval:

```text
run: runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1
eval: runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\eval.jsonl
strategy trajectories: data\trajectories\strategy_imitation_v2_candidate_eval_v1
strategy diagnostics: runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\strategy_trajectory_diagnostics.json
```

Eval summary:

```text
9 games, return_code=0 for all games
5 Victory / 3 Defeat / 1 Tie
Easy: 3/0/0
Medium: 2/0/1
Hard: 0/3/0
strategy action coverage=100%
STAY_COURSE=148
EXPAND=26
ADD_GATEWAYS=71
TECH_ROBO=22
FORGE_UPGRADES=49
BUILD_STATIC_DEFENSE=63
PRODUCE_ARMY=35
BOOST_WORKERS=73
```

Conclusion:

- The strategy checkpoint path is runnable and did not collapse online.
- It is not yet a stable strategy baseline: validation accuracy is modest and Hard games all lost in the first guarded sample.
- Continue strategy data/teacher/eval work before treating this as a stable initializer.

Timing diagnostics:

```text
imitation timing:
  runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\strategy_timing_diagnostics.json
coverage-teacher timing:
  runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\strategy_timing_diagnostics.json
imitation agreement:
  runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\strategy_agreement_diagnostics.json
coverage-teacher agreement:
  runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\strategy_agreement_diagnostics.json
```

Timing conclusion:

- Hard failures are not simple online action collapse.
- `strategy_imitation_v2_candidate` starts `ADD_GATEWAYS` much later than the
  recommended coverage-teacher data: first 411.4s / avg 582.7s vs first 91.4s /
  avg 312.2s.
- Under threat, imitation almost always emits `BUILD_STATIC_DEFENSE`
  (`ADD_GATEWAYS=1`, `BUILD_STATIC_DEFENSE=62`), while teacher keeps a more
  varied threat response (`TECH_ROBO=13`, `BUILD_STATIC_DEFENSE=50`, plus a few
  `STAY_COURSE`, `BOOST_WORKERS`, and `FORGE_UPGRADES`).
- TECH_ROBO timing is not currently a clear teacher bug: in the coverage-teacher
  data, every armored/cloaked signal file already has TECH_ROBO before the
  signal, with no no-tech signal files.
- Agreement diagnostics do not show teacher label drift: coverage-teacher data has
  `stored_vs_teacher_accuracy=1.000`.
- `strategy_imitation_v2_candidate` has weak offline agreement with teacher:
  `checkpoint_vs_teacher_accuracy=0.589` on teacher training trajectories and
  `0.511` on candidate online eval states.
- Biggest agreement gaps are `TECH_ROBO` and pending/tech states:
  teacher data has `TECH_ROBO` mismatches 96/130 and `pending_robo`
  bucket accuracy 0.207; candidate online states have `TECH_ROBO` mismatches
  124/143 and `tech_robo_needed` bucket accuracy 0.302.
- Do not change `CoverageStrategyPolicy` from this evidence alone.

Current v2 candidate:

```text
Hard-focused data:
  run: runs\20260623_122724_strategy_coverage_teacher_v2_hard_focus_v1
  trajectory: data\trajectories\strategy_coverage_teacher_v2_hard_focus_v1
  result: 4 Victory / 1 Defeat / 1 Tie, return_code=0 for all

Combined training data:
  data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1
  data\trajectories\strategy_coverage_teacher_v2_hard_focus_v1
  24 games, 1247 training rows, strategy_v2 only, defaulted=0,
  action_coverage=100%, low_count_actions=<none>

Training:
  run: runs\20260623_123449_strategy_imitation_v2_candidate_v2
  checkpoint: runs\20260623_123449_strategy_imitation_v2_candidate_v2\checkpoints\policy.pt
  examples=1247, validation_accuracy=0.711

Offline agreement on combined teacher data:
  v1 checkpoint_vs_teacher_accuracy=0.576
  v2 checkpoint_vs_teacher_accuracy=0.751
  tech_robo_needed bucket: 0.403 -> 0.835
  pending_robo bucket: 0.217 -> 0.639

Hard smoke eval:
  run: runs\20260623_123541_eval_strategy_imitation_v2_candidate_v2_hard_smoke
  trajectory: data\trajectories\strategy_imitation_v2_candidate_v2_hard_smoke
  result: 2 Victory / 1 Defeat, return_code=0 for all
  schema=strategy_v2 only, defaulted=0, action_coverage=100%

Hard Terran compare:
  v2 run: runs\20260623_124528_strategy_v2_hard_terran_compare_v1
    result: 0 Victory / 2 Defeat
  v1 run: runs\20260623_124835_strategy_v1_hard_terran_compare_v1
    result: 1 Victory / 1 Defeat
  rule run: runs\20260623_125103_rule_strategy_hard_terran_compare_v1
    result: 2 Victory / 0 Defeat
  coverage-teacher run: runs\20260623_134344_coverage_teacher_strategy_hard_terran_compare_v2
    result: 1 Victory / 0 Defeat / 1 Tie
```

Conclusion:

- `strategy_imitation_v2_candidate_v2` is a better offline candidate than v1 and
  passed a tiny guarded Hard smoke eval, but it regressed in the focused Hard
  Terran comparison.
- It is not a stable strategy baseline: v2 lost both focused Hard Terran games,
  while rule won both and coverage-teacher went 1 Victory / 1 Tie.
- Current evidence suggests strategy-layer macro actions can disrupt the rule
  baseline's naturally strong Hard Terran timing. Do not promote v2.
- Next step is to diagnose why rule/no-op wins Hard Terran while strategy
  policies lose or tie; likely compare macro executor side effects, expansion/
  tech/static-defense interference, and army-rule timing under strategy actions.

## Current AIBuild Smoke

Implementation status:

```text
run.py: --ai-build RandomBuild|Rush|Timing|Power|Macro|Air, default RandomBuild
scripts/evaluate.py: --ai-builds ..., default RandomBuild
eval/summary/experiment config: opponent_ai_build recorded and grouped
army/strategy trajectories: opponent_ai_build recorded as metadata
trajectory filenames: include opponent_ai_build for new evaluate collections
```

Validation:

```text
pytest: 124 passed
check_env: all OK
guard pid: 21392
smoke run: runs\20260623_144443_eval_aibuild_smoke_v1
eval: runs\20260623_144443_eval_aibuild_smoke_v1\artifacts\eval.jsonl
summary: runs\20260623_144443_eval_aibuild_smoke_v1\artifacts\summary.json
army trajectory: data\trajectories\aibuild_smoke_army_v1
strategy trajectory: data\trajectories\aibuild_smoke_strategy_v1
result: 1 game, return_code=0, Result.Tie, opponent_ai_build=Rush everywhere
```

## Current AIBuild Matrix

Hard Terran, AcropolisLE, one game per official build:

```text
rule/no-op:
  run: runs\20260623_145519_eval_rule_aibuild_hard_terran_v1
  strategy trajectory: data\trajectories\rule_aibuild_hard_terran_strategy_v1
  results:
    Rush: Victory
    Timing: Victory
    Power: Defeat
    Macro: Defeat
    Air: Tie

coverage-teacher strategy:
  run: runs\20260623_150115_eval_coverage_teacher_aibuild_hard_terran_v1
  strategy trajectory: data\trajectories\coverage_teacher_aibuild_hard_terran_strategy_v1
  results:
    Rush: Victory
    Timing: Victory
    Power: Defeat
    Macro: Victory
    Air: Tie
```

Interpretation:

- The first useful build-labeled split is Macro: coverage-teacher won while
  rule/no-op lost.
- Power is still a shared failure in this tiny sample.
- coverage-teacher Macro was lightweight: early `ADD_GATEWAYS`, repeated
  `PRODUCE_ARMY`, only 3 `TECH_ROBO`, and one static-defense action.
- coverage-teacher Power over-emitted `TECH_ROBO` (38 labels) and then
  `BUILD_STATIC_DEFENSE` under threat; TacticSpec should cap pending/cooldown
  repeats and protect production rhythm before any tactic-aware rollout.

## Current TacticSpec Skeleton

Code:

```text
rl\tactics.py
bot\managers\tactic_selector.py
tests\test_tactics.py
```

First tactic IDs:

```text
SAFE_MACRO
ANTI_RUSH_DEFENSE
GATEWAY_PRESSURE
ROBO_TIMING
TECH_POWER
ANTI_AIR_RESPONSE
RECOVERY
```

Status:

- Wired only through explicit opt-in `--strategy-tactic-mode rule`.
- Default `--strategy-tactic-mode off` keeps runtime unchanged.
- No new action space.
- No observation schema change.
- `RuleTacticSelector` uses `opponent_ai_build` and strategy observation metadata
  to choose a tactic with a switch cooldown and emergency overrides.
- `filter_strategy_action()` can conservatively replace disallowed/repeated
  strategy actions using tactic constraints.

## Current Tactic-Aware Smoke

Hard Terran Macro/Power, AcropolisLE, coverage-teacher with tactic filter:

```text
run: runs\20260623_152644_eval_tactic_coverage_aibuild_hard_terran_v1
eval: runs\20260623_152644_eval_tactic_coverage_aibuild_hard_terran_v1\artifacts\eval.jsonl
strategy trajectory: data\trajectories\tactic_coverage_aibuild_hard_terran_strategy_v1
strategy timing: runs\20260623_152644_eval_tactic_coverage_aibuild_hard_terran_v1\artifacts\strategy_timing_diagnostics.json
tactic diagnostics: runs\20260623_152644_eval_tactic_coverage_aibuild_hard_terran_v1\artifacts\tactic_diagnostics.json
config: strategy_policy=coverage-teacher, strategy_tactic_mode=rule, ai_builds=Macro/Power
results:
  Macro: Victory
  Power: Victory
return_code=0 for both games
```

Metadata/action-filter check:

```text
rows=88
training_rows=86
rows_with_tactic_metadata=88
rows_with_filter_metadata=88
filter_change_rows=29
training_filter_change_rows=28
unique_tactics:
  SAFE_MACRO
  ANTI_RUSH_DEFENSE
  GATEWAY_PRESSURE
  ANTI_AIR_RESPONSE
  TECH_POWER

Power:
  terminal-inclusive TECH_POWER, TECH_ROBO -> PRODUCE_ARMY: 19
  training rows TECH_POWER, TECH_ROBO -> PRODUCE_ARMY: 18
Macro:
  several TECH_ROBO -> PRODUCE_ARMY changes under defensive/pressure tactics
```

Interpretation:

- The opt-in tactic filter preserved the previous Macro win in this tiny smoke.
- Power flipped from defeat in the previous coverage-teacher sample to victory in
  this tiny smoke, with repeated TECH_ROBO labels filtered to PRODUCE_ARMY.
- This is not enough to promote tactic-aware mode; use it as a promising smoke
  result and expand guarded comparisons before collecting training data.

## Current Tactic-Aware Follow-up

The broader guarded follow-up did not support promoting the tactic filter.

Hard Terran all-build one-game samples:

```text
v1 broad filter:
  run: runs\20260623_154821_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v1
  results: Rush=Defeat, Timing=Defeat, Power=Victory, Macro=Defeat, Air=Defeat

v2 safer spec filter:
  run: runs\20260623_160120_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v2
  results: Rush=Defeat, Timing=Victory, Power=Defeat, Macro=Tie, Air=Tie

v3 Power-targeted filter:
  run: runs\20260623_160924_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v3
  results: Rush=Victory, Timing=Defeat, Power=Defeat, Macro=Victory, Air=Defeat
  tactic diagnostics: filter_changes=12, all on Power rows
```

Power-only three-game recheck:

```text
coverage-teacher no filter:
  run: runs\20260623_161606_eval_coverage_teacher_aibuild_hard_terran_power_recheck_v1
  results: Tie, Victory, Defeat
  summary: 1W / 1T / 1L

Power-targeted tactic filter:
  run: runs\20260623_162021_eval_tactic_power_targeted_hard_terran_power_recheck_v1
  results: Tie, Tie, Defeat
  summary: 0W / 2T / 1L
  tactic diagnostics: filter_changes=46, all Power rows
```

Interpretation:

- Broad filtering harms non-Power builds and should not be used for collection.
- Power-targeted filtering is safer for non-Power builds because it records
  tactic metadata without changing actions, but it did not outperform no-filter
  coverage-teacher in the 3-game Power recheck.
- Do not promote tactic-aware mode or collect tactic-aware training data yet.
- Next tactic work should be offline/spec refinement around Power failure modes,
  not PPO and not broad data collection.

## Current Power Tactic Diagnostics

Offline diagnostics, no SC2 launch:

```text
run/artifacts: runs\20260623_power_tactic_diagnostics_v1\artifacts
report: runs\20260623_power_tactic_diagnostics_v1\artifacts\power_tactic_diagnostics.txt
json: runs\20260623_power_tactic_diagnostics_v1\artifacts\power_tactic_diagnostics.json
inputs:
  data\trajectories\coverage_teacher_aibuild_hard_terran_power_recheck_strategy_v1
  data\trajectories\tactic_power_targeted_hard_terran_power_recheck_strategy_v1
files=6
rows=437
training_rows=431
results=1 Victory / 3 Tie / 2 Defeat
opponent_ai_build=Power for all training rows
filter_changes=46 terminal-inclusive, 46 training rows
pytest: 144 passed
check_env: all OK
```

Key offline findings:

```text
No-filter coverage-teacher:
  1 Tie / 1 Victory / 1 Defeat
  first TECH_ROBO around 240.0-251.4s
  first ready_robo around 331.4-365.7s
  Victory file had no base_under_threat rows and produced steadily.

Power-targeted tactic filter:
  2 Tie / 1 Defeat
  counterfactual filter delta:
    TECH_ROBO -14
    BUILD_STATIC_DEFENSE -12
    ADD_GATEWAYS -16
    PRODUCE_ARMY +26
    FORGE_UPGRADES +11
    BOOST_WORKERS +5
  largest changes:
    TECH_POWER TECH_ROBO -> PRODUCE_ARMY: 15
    TECH_POWER ADD_GATEWAYS -> FORGE_UPGRADES: 11
    TECH_POWER BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 8
  One Tie delayed actual TECH_ROBO until 628.6s despite pending/ready Robo only
  at 640.0s/685.7s.
  The tactic Defeat did make an Observer at 411.4s, but still no Immortal.
```

Interpretation:

- The current Power filter is likely over-suppressing `TECH_ROBO` and
  `ADD_GATEWAYS` while adding many `PRODUCE_ARMY` / `FORGE_UPGRADES` actions.
- `BUILD_STATIC_DEFENSE` suppression reduced static-defense spam, but the
  replacement did not improve the three-game Power recheck.
- Do not promote tactic-aware mode and do not collect tactic-aware training data.
- Next step should be an offline counterfactual/spec adjustment for `TECH_POWER`,
  especially preserving one timely Robo and gateway production rhythm before any
  guarded paired comparison.

## Current Revised TECH_POWER Spec

Offline spec/test adjustment only, no SC2 launch:

```text
files:
  rl\tactics.py
  tests\test_tactics.py
  tests\test_tactic_strategy_policy.py
focused pytest: 19 passed
full pytest: 150 passed
check_env: all OK
```

Behavior changes in the explicit opt-in tactic filter:

```text
TECH_POWER preferred fallback order:
  TECH_ROBO -> PRODUCE_ARMY -> ADD_GATEWAYS -> FORGE_UPGRADES

If TECH_POWER proposes TECH_ROBO with no ready/pending Robo but low minerals and
enough vespene, the filter now returns STAY_COURSE to save for the first Robo
instead of spending the step on PRODUCE_ARMY.

If ADD_GATEWAYS is capped by pending gateways and no Robo has started, the
fallback is now TECH_ROBO instead of FORGE_UPGRADES.

If ADD_GATEWAYS is capped after Robo is already ready/started, the fallback is
PRODUCE_ARMY instead of FORGE_UPGRADES.

If BUILD_STATIC_DEFENSE is rejected under threat, the fallback is PRODUCE_ARMY.

If TECH_ROBO is proposed after a Robo is already ready, the fallback is
PRODUCE_ARMY, which keeps attention on Observer/Immortal-capable production
instead of repeating Robo tech.
```

Boundaries:

```text
Default --strategy-policy rule / --strategy-tactic-mode off remains unchanged.
No observation schema change.
No training.
No PPO.
Guarded comparison has now been run; revised filter is not promoted.
```

## Current Revised Power A/B

Guarded Power-only paired comparison, fresh dirs:

```text
guard pid: 21392
map/difficulty/opponent/build: AcropolisLE / Hard / Terran / Power
games per side: 3

no-filter coverage-teacher:
  run: runs\20260623_170757_20260623_eval_power_ab_no_filter_revised_v1
  army trajectory: data\trajectories\power_ab_no_filter_revised_army_v1
  strategy trajectory: data\trajectories\power_ab_no_filter_revised_strategy_v1
  result: 1 Victory / 2 Defeat / 0 Tie
  return_code=0 for all games

revised Power tactic filter:
  run: runs\20260623_171134_20260623_eval_power_ab_revised_tactic_v1
  army trajectory: data\trajectories\power_ab_revised_tactic_army_v1
  strategy trajectory: data\trajectories\power_ab_revised_tactic_strategy_v1
  result: 0 Victory / 2 Defeat / 1 Tie
  return_code=0 for all games

diagnostics:
  runs\20260623_power_ab_revised_diagnostics_v1\artifacts\power_ab_diagnostics.txt
  runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_tactic_diagnostics.json
  runs\20260623_power_ab_revised_diagnostics_v1\artifacts\no_filter_power_diagnostics.json
  runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_power_diagnostics.json
```

Diagnostic conclusion:

```text
Robo:
  revised filter removed the previous catastrophic first-Robo delay:
    revised first TECH_ROBO = 274.3 / 297.1 / 274.3s
    revised ready_robo = 331.4 / 354.3 / 331.4s
  but no-filter still starts TECH_ROBO earlier:
    no-filter first TECH_ROBO = 251.4 / 251.4 / 251.4s

Gateway:
  not fixed. no-filter ADD_GATEWAYS appears early (min 91.4s, avg 110.5s),
  while revised actual ADD_GATEWAYS is late (min 502.9s) and SAFE_MACRO still
  changes ADD_GATEWAYS -> BOOST_WORKERS 7 times.

Filter deltas:
  FORGE_UPGRADES -13, TECH_ROBO +10, BUILD_STATIC_DEFENSE -15,
  PRODUCE_ARMY +22, ADD_GATEWAYS -12.

Observer/Immortal:
  revised produced Observer/Immortal only in the Tie file
  (Observer 560.0s, Immortal 605.7s); both Defeat files still had neither.

Outcome:
  revised filter did not outperform no-filter and should remain diagnostic-only.
  Do not collect tactic-aware training data from it.
```

## Current Strategy Outcome Diagnostics

Implementation-only/offline pass, no SC2 launch:

```text
files:
  rl\strategy_outcome_diagnostics.py
  scripts\diagnose_strategy_outcomes.py
  tests\test_strategy_outcome_diagnostics.py

focused pytest:
  .\.venv\Scripts\python.exe -m pytest tests\test_strategy_outcome_diagnostics.py -q
  4 passed

full pytest:
  .\.venv\Scripts\python.exe -m pytest -q
  154 passed

check_env:
  .\.venv\Scripts\python.exe scripts\check_env.py
  OK
```

Power A/B outcome report:

```text
runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.txt
runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.json

inputs:
  data\trajectories\power_ab_no_filter_revised_strategy_v1
  data\trajectories\power_ab_revised_tactic_strategy_v1

files=6, rows=393, training_rows=387
results=1 Victory / 1 Tie / 4 Defeat
```

Key outcome findings:

```text
No-filter:
  first ADD_GATEWAYS=91.4s
  first TECH_ROBO=251.4s
  3 files: 1 Victory / 2 Defeat

Revised tactic:
  filter_change_rows=42
  first actual ADD_GATEWAYS=502.9s at source level
  first TECH_ROBO=274.3s
  3 files: 0 Victory / 1 Tie / 2 Defeat

SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS:
  count=7, all before 240s.
  This removes early strategy-level ADD_GATEWAYS labels, though the rule baseline
  still builds some Gateways underneath.

TECH_POWER ADD_GATEWAYS -> PRODUCE_ARMY:
  count=2.
  It does not improve immediate army count at +30/+60s, but both samples show
  Observer/Immortal payoff by +120s; sample is too small to promote.

Static-defense suppression:
  TECH_POWER BUILD_STATIC_DEFENSE -> PRODUCE_ARMY cleared threat in all 3 samples
  by +30s.
  RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY was mixed: at +60s, 7/8 threat
  cleared and 1/8 persisted; at +120s, 8/8 cleared.

Robo payoff:
  revised Defeat files had ready_robo at 331.4s / 354.3s but no Observer or
  Immortal. Rows after ready_robo show robo_idle_count=1 repeatedly, and the
  policy mostly emits STAY_COURSE / PRODUCE_ARMY / FORGE_UPGRADES rather than
  TECH_ROBO, so StrategyExecutor never gets a Robo-unit production trigger.
  The revised Tie file eventually produced Observer at 560.0s and Immortal at
  605.7s.
```

## 推荐下一步

不要先做 PPO。

推荐顺序：

1. 保持默认 rule baseline 行为不变，所有 strategy / tactic 扩展都必须显式 opt-in。
2. 不要从当前 tactic filter 采集 tactic-aware 训练数据。
3. 在再次修改 runtime 行为前，使用离线 `StrategyOutcomeDiagnostics` 作为证据：
   `rl/strategy_outcome_diagnostics.py`,
   `scripts/diagnose_strategy_outcomes.py`,
   `tests/test_strategy_outcome_diagnostics.py`.
4. 已生成首批 Power A/B outcome 诊断：
   `runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.txt`
   和 `strategy_outcomes.json`。
5. 用 outcome diagnostics 继续回答 `ADD_GATEWAYS`、`TECH_ROBO`、`PRODUCE_ARMY`、
   `BUILD_STATIC_DEFENSE`、`FORGE_UPGRADES`、`EXPAND`、`BOOST_WORKERS` 是否在
   +30/+60/+90/+120s 内落地。
6. 将 tactic filter 调整为 guardrail-first：默认 pass-through，只拦截 repeated /
   pending-capped / clearly harmful actions。
7. 优先处理 ready_robo 后没有 Observer / Immortal 的生产触发缺口，考虑显式 opt-in
   production-bias hook 或更窄的 guardrail，而不是扩大 action rewrite。
8. `strategy_imitation_v2_candidate_v2` 只作为诊断 candidate，不作为 baseline。
9. 只有 army 和 strategy learned baseline 稳定，并且 reward / environment 边界清楚后，才考虑 PPO。

## 常用命令

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

环境检查：

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

诊断当前 candidate eval：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py data\trajectories\imitation_v3_candidate_eval_v2 --show-files
```

诊断 strategy trajectory：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py <strategy-trajectory-dir> --kind strategy --show-files
```

诊断 tactic metadata / filter changes：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_tactics.py <strategy-trajectory-dir> --show-files --json-output <run-artifacts-dir>\tactic_diagnostics.json
```

诊断 Power tactic failure modes：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_power_tactics.py <strategy-trajectory-dir> [<strategy-trajectory-dir> ...] --show-files --json-output <run-artifacts-dir>\power_tactic_diagnostics.json --text-output <run-artifacts-dir>\power_tactic_diagnostics.txt
```

诊断 strategy action outcomes：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_strategy_outcomes.py <strategy-trajectory-dir> [<strategy-trajectory-dir> ...] --show-files --json-output <run-artifacts-dir>\strategy_outcomes.json --text-output <run-artifacts-dir>\strategy_outcomes.txt
```

审计 strategy candidate promotion gate：

```powershell
.\.venv\Scripts\python.exe scripts\audit_strategy_candidate.py <baseline-strategy-dir> <candidate-strategy-dir> --json-output <run-artifacts-dir>\promotion_gate.json --text-output <run-artifacts-dir>\strategy_candidate_audit.txt
```

诊断 strategy replay candidate：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_strategy_replay_candidate.py <strategy-trajectory-dir> --show-files --show-timeline --json-output <run-artifacts-dir>\strategy_replay_candidate.json --text-output <run-artifacts-dir>\strategy_replay_candidate.txt
```

诊断 RECOVERY / TECH_POWER active-threat suppression outcomes：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_active_threat_suppression.py <strategy-trajectory-dir> [<strategy-trajectory-dir> ...] --show-files --show-timeline --json-output <run-artifacts-dir>\active_threat_suppression.json --text-output <run-artifacts-dir>\active_threat_suppression.txt
```

评测并单独记录 strategy trajectory：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --trajectory-dir <army-dir> --strategy-trajectory-dir <strategy-dir> ...
```

采集显式 opt-in 的 coverage strategy 标签：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --strategy-policy coverage-teacher --strategy-trajectory-dir <strategy-dir> ...
```

训练 strategy imitation：

```powershell
.\.venv\Scripts\python.exe scripts\train_strategy_imitation.py data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 --run-root runs --run-name strategy_imitation_v2_candidate --epochs 12 --batch-size 128 --class-weighting balanced
```

评测 strategy checkpoint：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --strategy-policy checkpoint --strategy-checkpoint runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt --strategy-trajectory-dir <strategy-eval-dir> ...
```

评测当前 candidate：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Medium Hard Harder --opponents Protoss Terran Zerg --games-per-combo 2 --trajectory-dir data\trajectories\imitation_v3_candidate_eval_v2 --run-root runs --run-name eval_imitation_v3_candidate_v2 --policy-name imitation_v3_candidate --policy-checkpoint runs\20260622_154050_imitation_v3_candidate\checkpoints\policy.pt --record-decision-interval 16 --game-time-limit 900
```

## 环境备注

- OS：Windows 11
- Python：`.venv` 中的 3.14.5
- SC2 client：网易 / 中国 5.0.15.96999，`Base96999`
- SC2 安装路径：同级目录 `..\StarCraft II`
- `bot/config.py` 会设置 `SC2PATH`；entry script 中应先 import 它，再 import `sc2.*`
- 已安装 RL 依赖包括 `gymnasium`、`stable-baselines3`、`torch`
- `rg.exe` 可能出现 `Access is denied`；必要时用 PowerShell `Get-ChildItem` / `Select-String`
