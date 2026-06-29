# 星海争霸 II AI Bot

这是一个 StarCraft II 神族 Bot 项目，目标是逐步做出一个能感知战局、解释思路、并根据局势调整打法的对战 AI。

当前项目重点不是直接跳到 PPO，而是先把“高层军队决策”的数据、诊断、模仿学习和安全评测链路打稳，并在不破坏默认 baseline 的前提下逐步扩展低频宏观策略层。经济、建筑、农民、补人口和造兵仍由规则 Bot 负责；学习策略目前只选择少量高层军队动作。

“不训练 / 不 PPO / 不采 tactic-aware 数据”不是永久禁令。当前原则是门槛式推进：先用 guarded A/B 和 outcome diagnostics 证明 opt-in strategy/tactic 改动至少不拖累 no-filter / rule baseline；证据支持后，可以采小规模 fresh tactic-aware 数据、训练 action-outcome / veto / imitation 模型。PPO 也可以进入路线图，但需要先补齐 SC2 env、reward、baseline 对照和安全启动边界。

## 当前状态

已完成：

- 神族规则 baseline Bot
- 隐藏窗口安全启动和 window guard
- `ArmyPolicy` 策略抽象
- 5 个高层军队动作
- `StrategyPolicy` 低频宏观策略抽象
- 8 个宏观策略动作，默认 `STAY_COURSE`
- 默认 no-op `RuleStrategyPolicy`
- `StrategyExecutor` 初版，可执行扩张、补兵营、转机械台、锻炉升级、静态防守、补农民/出兵等宏观意图
- `StrategyCoverageTeacher`，显式启用后可生成宏观策略标签
- strategy imitation 数据加载、训练脚本和独立 checkpoint 格式
- checkpoint-backed `StrategyPolicy`，通过 `--strategy-policy checkpoint` 显式启用
- 保守 GG / 投降逻辑：只在接近不可恢复的残局发送 `gg` 并退出
- schema v3 数值观测，26 维
- `strategy_v2` 宏观策略观测，40 维，并兼容诊断旧 `strategy_v1`
- trajectory 记录和诊断
- 独立 strategy trajectory 记录，需显式传入 `--strategy-trajectory-path` / `--strategy-trajectory-dir`
- diagnostics `--kind strategy` 模式，可诊断 strategy schema 和 action coverage
- strategy timing diagnostics，可查看宏观动作时机、威胁状态动作、TECH_ROBO 信号时机和 pending 重复动作
- strategy agreement diagnostics，可离线比较 recorded label、当前 coverage-teacher label 和 checkpoint 预测
- 实验元数据记录
- 官方内置 AI `AIBuild` 评测/采集维度：`RandomBuild`、`Rush`、`Timing`、`Power`、`Macro`、`Air`
- `opponent_ai_build` 已记录到 `eval.jsonl`、`summary.json`、experiment config、army trajectory、strategy trajectory
- `TacticSpec` / `TacticState` 战术池雏形、`RuleTacticSelector`，以及显式 opt-in 的 tactic-aware coverage-teacher mode
- 模型和 checkpoint 元数据记录
- observation normalization
- imitation learning
- checkpoint-backed 在线策略推理
- coverage-teacher 数据采集
- 实验性 LLM-backed army policy，可输出简短解释

未完成：

- Gymnasium SC2 environment
- PPO reward function
- PPO training script
- PPO checkpoint adapter
- curriculum controller

当前 baseline 结论：

- `imitation_v3_candidate` 已经可以在线运行。
- 它可作为 schema v3 的 smoke / comparison baseline。
- 它还不是稳定的 PPO 初始化策略。
- 在线 `RETREAT_HOME` 仍然低频，`RALLY` / `HOLD` 仍然占主导。
- `strategy_imitation_v2_candidate` 已经可以在线运行，且小批量 eval 没有 action collapse。
- `strategy_imitation_v2_candidate_v2` 已训练完成，离线指标明显好于 v1，并通过一个很小的 Hard smoke eval；但 Hard Terran focused compare 退化，不能提升为稳定 strategy baseline。
- 在 v3 baseline 更稳定，或策略层被有计划地扩展之前，不建议开始 PPO。

## 下一步计划

近期重点不是盲目堆训练轮数，也不是直接进入 PPO，更不是继续给
`rl\tactics.py` 堆手写 guardrail。当前主线见 `doc\DEVELOPMENT_PLAN.md`：
冻结 tactic-rule runtime；strategy action execution observability 和第一版
candidate audit / replay-only gate 已落地。每次发出 `ADD_GATEWAYS`、`TECH_ROBO`、
`PRODUCE_ARMY`、`BUILD_STATIC_DEFENSE` 之后，都要能看清 executor 是否尝试执行、
产生了什么 effect、被什么 blocker 卡住，以及 +30/+60/+90/+120 秒后是否真的
补到了产能、科技、部队或防守收益。

当前推荐路线：

- 官方内置 AI 的 `AIBuild` 已接入评测和采集：
  - `RandomBuild`
  - `Rush`
  - `Timing`
  - `Power`
  - `Macro`
  - `Air`
- 默认仍为 `RandomBuild`，旧 evaluate 命令保持兼容。
- 已通过 1 局 guarded smoke：`runs\20260623_144443_eval_aibuild_smoke_v1`，`opponent_ai_build=Rush` 写入 eval、summary、army trajectory、strategy trajectory。
- 已跑第一组 Hard Terran build-labeled 小矩阵：
  - rule/no-op：Rush 胜、Timing 胜、Power 败、Macro 败、Air 平。
  - coverage-teacher：Rush 胜、Timing 胜、Power 败、Macro 胜、Air 平。
- 第一版 `TacticSpec` 战术池雏形已建立，并可通过 `--strategy-policy coverage-teacher --strategy-tactic-mode rule` 显式启用。
- 默认 `--strategy-tactic-mode off`，默认 rule/no-op baseline 不变。
- Hard Terran Macro/Power tactic-aware smoke 两局都完成且 return_code=0。
- `scripts\diagnose_tactics.py` 已可汇总 tactic metadata、build 分布和 filter-change 统计。
- 后续 all-build / Power-only recheck 没有支持推广 tactic filter；当前 tactic-aware 只作为实验诊断，不建议采训练数据。
- `scripts\diagnose_power_tactics.py` 已新增，可离线诊断 Power recheck 的逐文件结果、action timing、tactic timeline、filter counterfactual、Robo/Forge/Observer/Immortal/static defense、威胁状态、兵力阈值、资源银行和 idle gateway。
- `scripts\diagnose_active_threat_outcomes.py` 已新增，可专门诊断 active threat 下 `BUILD_STATIC_DEFENSE` 被 tactic filter 改写后，在 +30/+60/+90/+120s 是否清 threat、是否补 static、是否真正涨兵。
- 最新离线诊断显示 Power-targeted filter 过度压低了 `TECH_ROBO` / `ADD_GATEWAYS`，并把大量动作改成 `PRODUCE_ARMY` / `FORGE_UPGRADES`；当前 tactic-aware filter 仍不能推广。
- 最新 anti-air ready-static promotion/replay gate 已生成：`runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\promotion_gate.json`、`strategy_candidate_audit.txt`、`strategy_replay_candidate.json`、`strategy_replay_candidate.txt`。候选审计为 `promotable=false`；replay gate 为 `hold_runtime_patch`，`changed_rows=66`，`candidate_executable=15/66`，最大分组 18 行且 0/18 可立即执行。当前不做 runtime patch、不推广、不采 tactic-aware 数据、不训练/PPO。
- 已基于该诊断离线微调 Power / `TECH_POWER` 的 TacticSpec，并补 focused tests。
- 已跑 guarded Power-only A/B：revised filter 修掉一部分 Robo delay，但没修好 Gateway 节奏，也没有赢过 no-filter；当前仍不能推广。
- 离线 `StrategyOutcomeDiagnostics` 已实现，可按 +30/+60/+90/+120s 复盘 strategy action 是否真正落地。
- 首批 Power A/B outcome 诊断已生成：`runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.txt` 和 `.json`。
- outcome 诊断确认 revised tactic filter 明显推迟实际 `ADD_GATEWAYS`，并暴露出 ready_robo 后 Robo 长时间 idle、缺少 Observer/Immortal 生产触发的问题。
- 已离线修第一层 guardrail：`SAFE_MACRO` 在 240s 前允许 1 个 pending Gateway 时继续放行 `ADD_GATEWAYS`，到 2 个 pending Gateway 才 fallback；`PRODUCE_ARMY` 现在会在 ready Robo 上优先补 Observer / Immortal，同时保留 Gateway 出兵委托。
- guarded Power-only A/B 已复查：no-filter coverage-teacher 为 1W/2L，tactic guardrail 也是 1W/2L；guardrail 改善了 source-level early Gateway 和 Observer payoff，但仍没有 Immortal payoff，不能推广。
- `scripts\diagnose_tactics.py` 已支持逐行 filter timeline，可输出 `original_action -> selected_action`、tactic、game_time、资源、pending/ready Gateway、threat 和 idle 产能字段。
- 最新第三局复查显示 `ADD_GATEWAYS=571.4s` 主要是 selected-label 视角的后期重建信号：102.9s 的 early `ADD_GATEWAYS` 被压掉时已有 2 个 pending Gateway，后续 ready Gateways 到过 4。
- `StrategyOutcomeDiagnostics` 已扩展 ready-Robo payoff 分类；当前 tactic guardrail 三局 Immortal 未落地均归因于资源/供给阻塞。
- 已加一处显式 opt-in 的 first-Immortal bias：ready Robo + Observer 已有 + Immortal 缺失且无基地威胁时，tactic filter 会优先让资源流向第一只 Immortal；默认 `--strategy-policy rule / --strategy-tactic-mode off` 不变。
- first-Immortal bias guarded Power-only A/B 已完成：no-filter coverage-teacher 为 0W/3L，tactic-rule 也是 0W/3L。tactic 第 1 局产出 Immortal 且更久，但整体没有赢过 no-filter；`ADD_GATEWAYS` timing 和 static-defense/threat 处理退化，暂不推广，也不进入 tactic-aware 训练。
- 已离线修复 first-Immortal bias 暴露的两个 guardrail 回归：Gateway 未达到 `own_bases * 4` 时保留 `ADD_GATEWAYS`；`TECH_POWER` 在 active base threat 且 static-defense slot 可用时保留/留矿 `BUILD_STATIC_DEFENSE`。
- guardrail-retention guarded Power-only A/B 已完成：no-filter coverage-teacher 为 2W/1L，tactic-rule 为 0W/3L。诊断显示 tactic filter 仍过度压低 `TECH_ROBO`；第三局 tactic 没有建出 ready Robo。
- 已离线补一个更窄的 initial-Robo precedence：`TECH_POWER` 在无 ready/pending Robo、Core ready、无威胁且资源足够时，会把 `STAY_COURSE` 改成 `TECH_ROBO` 抢回第一座 Robo 窗口。
- initial-Robo guarded Power-only A/B 已完成：no-filter coverage-teacher 为 1W/1T/1L，tactic-rule 为 0W/1T/2L。tactic 三局都建出了 ready Robo，两局产出 Immortal，但 `ADD_GATEWAYS` 仍被明显推迟。
- 已离线补 midgame Gateway cap：`TECH_POWER` 在 Robo 已 ready/pending、无基地威胁、Gateway 低于 `own_bases * 4` 时，允许最多 2 个 pending Gateway，避免 2 base/4 Gateway 附近把 `ADD_GATEWAYS` 改成 `PRODUCE_ARMY`。
- midgame Gateway cap guarded Power-only A/B 已完成：no-filter coverage-teacher 为 1W/2L，tactic-rule 为 1W/2L。tactic first `ADD_GATEWAYS` 相比 initial-Robo A/B 有改善，但仍晚于 no-filter；仍有 no-ready-Robo 和 missed-Immortal 文件，不推广，不采 tactic-aware 数据。
- 已离线补 first-Robo banking guard：`TECH_POWER` 在无 ready/pending Robo、Core ready、无威胁时，会把可能打散第一座 Robo 资源节奏的 `STAY_COURSE` / `FORGE_UPGRADES` / `ADD_GATEWAYS` / `PRODUCE_ARMY` / `BOOST_WORKERS` 等动作改成 `TECH_ROBO` 或留矿；安全 `EXPAND` 仍保留。
- first-Robo banking guard guarded Power-only A/B 已完成：no-filter coverage-teacher 为 1W/1T/1L，tactic-rule 为 2W/1T/0L。tactic 三局都有 ready Robo + Observer，1 局出了 Immortal，base threat 也更少；但 first `ADD_GATEWAYS` 退到 411.4s，仍不能采 tactic-aware 训练数据。
- 已离线补 Gateway preservation follow-up：`SAFE_MACRO` 在 120s 前、无 ready Gateway、已有 2 pending Gateway、minerals>=250、vespene<100、无威胁时，允许第 3 个 pending Gateway；Robo gas ready 后或 active base threat 下仍保持原 cap。
- Gateway preservation guarded Power-only A/B 已完成：no-filter coverage-teacher 为 1W/2L，tactic-rule 为 2W/1T/0L。tactic first `ADD_GATEWAYS` 回到 91.4s，count=11；三局都有 ready Robo，2 局 Observer，1 局 Immortal。仍有 1 局 Victory 的 Observer 被资源/供给阻塞，不采 tactic-aware 数据。
- Gateway preservation 6-valid confirmatory A/B 已完成：no-filter 有效样本为 2W/1T/3L（另有 1 个 `NO_RESULT code=1` 后用 fresh dir top-up），tactic-rule 为 2W/1T/3L。Gateway timing 保住，tactic first `ADD_GATEWAYS=91.4s`；Robo payoff 改善到 ready 5/6、Observer 5/6、Immortal 2/6。但 tactic `base_threat_rows=44` 高于 no-filter 的 39，`BUILD_STATIC_DEFENSE=13` 低于 no-filter 的 39，`filter_change_rows=75`，不采 tactic-aware 数据、不训练。
- 已离线补 static-defense retention follow-up：`ANTI_AIR_RESPONSE` / `ANTI_RUSH_DEFENSE` / `ROBO_TIMING` / `RECOVERY` 在 active threat 且无 ready/pending static defense 时，会保留可负担的 `BUILD_STATIC_DEFENSE`，矿不足时 `STAY_COURSE` 留矿；已有 static 且矿不足时仍允许 fallback 到产兵。confirm6 replay 显示 3 行 `ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY` 会改为留矿，`RECOVERY` 低矿无 static 行也会留矿。
- `scripts\diagnose_power_tactics.py` 已扩展 Robo banking filter context：`TECH_ROBO -> STAY_COURSE` 现在会按 `first_robo_mineral_short` / `first_robo_affordable` / pending / ready Robo 等状态分类。旧 confirm6 和新 confirm6 的 26 条 `TECH_POWER TECH_ROBO -> STAY_COURSE` 都是 `first_robo_mineral_short`，应视为留矿诊断噪声，不是可负担第一座 Robo 被误压。
- Static-defense retention fresh-dir confirm6 A/B 已完成：no-filter coverage-teacher 为 3W/3L，tactic-rule 为 3W/1T/2L。tactic first `ADD_GATEWAYS=91.4s`，Robo payoff 改善到 ready 5/6、Observer 4/6、Immortal 2/6；但 `base_threat_rows=52` 高于 no-filter 的 46，`BUILD_STATIC_DEFENSE=6` 远低于 no-filter 的 43，`filter_change_rows=82`，仍不推广、不采 tactic-aware 数据、不训练。
- `scripts\diagnose_power_tactics.py` 已继续扩展 static-defense filter context：active threat 下 `BUILD_STATIC_DEFENSE` 被改写时，会区分 `no_static_mineral_short`、`ready_static_low_minerals`、`pending_static_waiting`、`pending_static_with_ready` 等。复查显示没有 `no_static_affordable`；5 行 `pending_static_waiting` 会被最小修复从 `PRODUCE_ARMY/TECH_ROBO` 改为等待。
- 最新 opt-in runtime 修复：active threat 下如果 static 已 pending 但还没有 ready static，tactic filter 返回 `STAY_COURSE` 等它落地；已有 ready static 时仍可 fallback 到产兵。默认 `--strategy-policy rule / --strategy-tactic-mode off` 不变。
- Pending-static wait guarded Power-only confirmatory A/B 已完成：no-filter coverage-teacher 为 1W/2T/3L，tactic-rule 为 1W/1T/4L。tactic Robo payoff 改善到 ready 5/6、Observer 5/6、Immortal 1/6，但 first `ADD_GATEWAYS` 平均退到 121.9s、`ADD_GATEWAYS` count 15 vs no-filter 28，且 `base_threat_rows=72` vs 33、`BUILD_STATIC_DEFENSE=15` vs 33。不要推广 tactic filter，不采 tactic-aware 数据，不训练/PPO。
- 最新验证：`tests\test_strategy_candidate_audit.py tests\test_strategy_replay_candidate.py` 为 `8 passed`，full pytest 为 `209 passed`，`scripts\check_env.py` OK，`Get-Process SC2,SC2_x64` 无输出。本轮未启动 SC2，未训练/PPO，未采 tactic-aware 数据。
- 开发路线已修正：冻结当前 tactic-rule runtime，不再连续堆手写 guardrail 追小样本 A/B；先用离线 outcome slice 证明某个 tactic/context 的 fallback 有害，再做显式 opt-in、可回滚、默认不变的最小 runtime 修复。
- 最新 active-threat outcome 诊断显示 `ANTI_AIR_RESPONSE ready_static_low_minerals -> PRODUCE_ARMY` 明确有害：11 行在 +60s 全部 threat persisted，+120s 仍有 5/11 persisted，且 army_count_delta 为负；而 `RECOVERY ready_static_low_minerals -> PRODUCE_ARMY` 到 +120s 为 8/8 cleared。已做一个仅限 ANTI_AIR 的 opt-in 窄修复：active air threat、已有 ready static、无 pending static、minerals < 100 时，不再 fallback 到 `PRODUCE_ARMY`，改为 `STAY_COURSE`；ground-only threat 和 `RECOVERY` 不变。
- Anti-air ready-static guarded Power-only A/B 已完成：no-filter coverage-teacher 为 2W/2T/2L，tactic-rule 为 0W/0T/6L。局部 anti-air `STAY_COURSE` 小桶在 +120s 2/2 清 threat，但整体 tactic-rule 仍把 `BUILD_STATIC_DEFENSE` 和 `TECH_ROBO` 压得太多、把 `PRODUCE_ARMY` / `STAY_COURSE` under threat 放得太多。不要推广 tactic filter，不采 tactic-aware 数据，不训练/PPO。
- 最新 A/B 诊断产物在 `runs\20260624_power_ab_anti_air_ready_static_tactic_timeline_v1\artifacts\`、`runs\20260624_power_ab_anti_air_ready_static_power_tactics_v1\artifacts\`、`runs\20260624_strategy_outcome_power_ab_anti_air_ready_static_v1\artifacts\`、`runs\20260624_active_threat_outcome_anti_air_ready_static_v1\artifacts\`。
- 已新增 active-threat suppression diagnostics：`scripts\diagnose_active_threat_suppression.py` / `rl\active_threat_suppression_diagnostics.py`。它离线定位 `RECOVERY` / `TECH_POWER` 下 `BUILD_STATIC_DEFENSE`、`TECH_ROBO` 被改写为 `PRODUCE_ARMY` / `STAY_COURSE` 后的 +30/+60/+90/+120s outcome，并输出 per-context、per-file timeline 和 replay-only candidate impact。
- 最新 suppression 诊断产物在 `runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\active_threat_suppression.txt` 和 `.json`。结论是 tactic-rule 有 50 条目标 suppression，no-filter 为 0；replay-only pass-through 会恢复 `BUILD_STATIC_DEFENSE +42`、`TECH_ROBO +8`，但只有 6/50 candidate rows 当场可执行。最大问题不是一个可一刀切修复的小桶：`RECOVERY ready_static_low_minerals -> PRODUCE_ARMY` 早期 threat 持续严重但到 +120s 多数清掉，`pending_static_waiting -> STAY_COURSE` 会补 static 却仍常让 threat 延续，`TECH_ROBO` suppression 多为 no-threat mineral-short。
- 本轮因此不继续堆 runtime patch；先冻结 tactic-rule runtime，等待更窄的 replay diff 和 outcome 证据。
- 当前开发计划已整理到 `doc\DEVELOPMENT_PLAN.md`。`strategy_execution_*` 字段、第一版 candidate audit、第一版 replay-only before-filter diagnostics 已完成；下一步是把 replay diagnostics 跑在冻结 tactic-rule trajectory 上，用 changed-row / executable-row / outcome slice 判断是否值得再做 runtime patch。
- 第一版战术池建议包括：
  - `SAFE_MACRO`
  - `ANTI_RUSH_DEFENSE`
  - `GATEWAY_PRESSURE`
  - `ROBO_TIMING`
  - `TECH_POWER`
  - `ANTI_AIR_RESPONSE`
  - `RECOVERY`

这个方向来自当前 Hard Terran 诊断：`strategy_imitation_v2_candidate_v2` 离线更像 teacher，但在线 focused compare 仍弱于 rule/no-op，说明策略层宏观动作可能干扰了 rule baseline 的资源和出兵节奏。战术池的目标是让 `EXPAND`、`TECH_ROBO`、`ADD_GATEWAYS`、`BUILD_STATIC_DEFENSE` 等动作服从同一个战术意图。

当前门槛：

- 不建议现在直接进入 PPO；先补 env / reward / baseline comparison / safe-launch 边界。
- 不建议把 `strategy_imitation_v2_candidate_v2` 当作稳定 strategy baseline。
- 不建议在没有 outcome / guarded A/B 证据时盲目重训。
- 如果后续 fresh-dir guarded A/B 稳定优于 no-filter，且 Gateway/Robo/static-defense 都没有明显退化，才可以进入小规模 tactic-aware 数据采集和训练实验。

当前下一步执行方案见 `doc\STRATEGY_OUTCOME_PLAN.md`。AIBuild / TacticSpec 背景计划见 `doc\TACTIC_POOL_PLAN.md`。历史策略层扩展路线已归档到 `doc\archive\STRATEGY_EXPANSION_PLAN.md`。

## 安全启动规则

用户可能正在同一台机器上工作。不要暴露 SC2 或 Battle.net 窗口。

任何可能启动 SC2 的命令之前，都必须先启动或确认隐藏窗口 guard：

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

优先使用：

```text
scripts/evaluate.py
scripts/safe_launch.py
```

除非用户明确要求可见调试，不要直接裸跑可见 `run.py`。

## 常用命令

环境检查：

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

安全跑一局：

```powershell
.\.venv\Scripts\python.exe scripts\safe_launch.py --keep-guard --guard-interval 0.02 -- --difficulty VeryEasy --opponent Protoss --hide-watch-seconds 120 --hide-watch-interval 0.02
```

AIBuild smoke / 采集示例：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Medium --opponents Terran --ai-builds Rush --games-per-combo 1 --run-root runs --run-name eval_aibuild_smoke_v1 --policy-name rule_aibuild_smoke --army-policy rule --strategy-policy rule --trajectory-dir data\trajectories\aibuild_smoke_army_v1 --strategy-trajectory-dir data\trajectories\aibuild_smoke_strategy_v1 --record-decision-interval 16 --game-time-limit 600
```

显式启用 tactic-aware coverage-teacher：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Macro Power --games-per-combo 1 --run-root runs --run-name eval_tactic_coverage_aibuild_hard_terran_v1 --policy-name tactic_coverage_aibuild_hard_terran_v1 --army-policy rule --strategy-policy coverage-teacher --strategy-tactic-mode rule --trajectory-dir data\trajectories\tactic_coverage_aibuild_hard_terran_army_v1 --strategy-trajectory-dir data\trajectories\tactic_coverage_aibuild_hard_terran_strategy_v1 --record-decision-interval 16 --game-time-limit 900
```

诊断 trajectory：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py data\trajectories\imitation_v3_candidate_eval_v2 --show-files
```

诊断 strategy trajectory：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py <strategy-trajectory-dir> --kind strategy --show-files
```

诊断 strategy action timing：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_strategy_timing.py <strategy-trajectory-dir> --show-files --json-output <run-artifacts-dir>\strategy_timing_diagnostics.json
```

诊断 tactic metadata / filter changes：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_tactics.py <strategy-trajectory-dir> --show-files --json-output <run-artifacts-dir>\tactic_diagnostics.json
```

诊断 Power tactic failure modes：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_power_tactics.py <strategy-trajectory-dir> [<strategy-trajectory-dir> ...] --show-files --json-output <run-artifacts-dir>\power_tactic_diagnostics.json --text-output <run-artifacts-dir>\power_tactic_diagnostics.txt
```

诊断 active-threat static-defense filter outcomes：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_active_threat_outcomes.py <strategy-trajectory-dir> --show-files --json-output <run-artifacts-dir>\active_threat_outcomes.json --text-output <run-artifacts-dir>\active_threat_outcomes.txt
```

诊断 RECOVERY / TECH_POWER active-threat suppression outcomes：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_active_threat_suppression.py <strategy-trajectory-dir> [<strategy-trajectory-dir> ...] --show-files --show-timeline --json-output <run-artifacts-dir>\active_threat_suppression.json --text-output <run-artifacts-dir>\active_threat_suppression.txt
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

诊断 strategy teacher/checkpoint agreement：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_strategy_agreement.py <strategy-trajectory-dir> --checkpoint <strategy-checkpoint.pt> --show-buckets --show-files --json-output <run-artifacts-dir>\strategy_agreement_diagnostics.json
```

评测时同时记录 army / strategy trajectory：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --trajectory-dir <army-dir> --strategy-trajectory-dir <strategy-dir> ...
```

采集 strategy coverage-teacher 标签：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --strategy-policy coverage-teacher --strategy-trajectory-dir <strategy-dir> ...
```

训练 strategy imitation policy：

```powershell
.\.venv\Scripts\python.exe scripts\train_strategy_imitation.py data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 --run-root runs --run-name strategy_imitation_v2_candidate --epochs 12 --batch-size 128 --class-weighting balanced
```

评测 strategy checkpoint：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --strategy-policy checkpoint --strategy-checkpoint runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt --strategy-trajectory-dir <strategy-eval-dir> ...
```

用 trajectory 训练 imitation policy：

```powershell
.\.venv\Scripts\python.exe scripts\train_imitation.py data\trajectories\coverage_teacher_v3_retreat_focused_v3 --run-root runs --run-name imitation_v3_candidate --epochs 8 --batch-size 128 --class-weighting balanced
```

评测当前 v3 imitation checkpoint：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Medium Hard Harder --opponents Protoss Terran Zerg --games-per-combo 2 --trajectory-dir data\trajectories\imitation_v3_candidate_eval_v2 --run-root runs --run-name eval_imitation_v3_candidate_v2 --policy-name imitation_v3_candidate --policy-checkpoint runs\20260622_154050_imitation_v3_candidate\checkpoints\policy.pt --record-decision-interval 16 --game-time-limit 900
```

## 当前军队动作空间

```text
0 RALLY
1 ATTACK_MAIN
2 RETREAT_HOME
3 DEFEND_BASE
4 HOLD
```

当前学习策略只控制这些军队层面的动作。宏观经济和生产仍由规则 Bot 执行。

## 当前策略动作空间

策略层是低频宏观意图，不替代军队动作。默认 `RuleStrategyPolicy` 始终返回 `STAY_COURSE`，所以正常 rule baseline 行为保持不变。

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

## 状态观测结构

当前 schema：v3。

模型看到的是压缩后的数值状态，不是游戏画面。主要特征包括：

- 时间、矿物、气体、人口
- 农民、基地、兵营、是否有控制核心
- 狂热者、追猎者、总兵力
- 是否正在进攻
- 已知敌方单位和建筑
- 我军到家、到敌方出生点的距离
- 基地威胁、最近敌军到家的距离
- 我军空闲/忙碌数量
- 攻击阶段兵力峰值和峰值后损失
- 最近兵力变化

旧 v1 / v2 trajectory 仅用于诊断或 smoke training 时的兼容补齐。旧 v1 / v2 checkpoint 与当前 runtime 故意不兼容。

## 策略观测结构

当前 strategy schema：`strategy_v2`，40 维。

它独立于 army schema v3，不影响现有 checkpoint 和默认 trajectory 路径。主要特征包括：

- 时间、资源、人口、农民和基地数
- 基地、Gateway / Robotics Facility / Forge / 静态防守的 ready 与 pending 状态
- Cybernetics Core 状态
- Zealot / Stalker / Immortal / Observer / Sentry 数量
- 地面攻防升级等级和升级 pending 状态
- 已知敌方空军、重甲、隐形单位信号
- 农民饱和度、Gateway/Robo 空闲数量
- 基地空中/地面威胁和最近敌军到家距离

## 当前关键产物

训练数据：

```text
data\trajectories\coverage_teacher_v3_retreat_focused_v3
```

combined diagnostics：

```text
runs\20260622_151332_coverage_teacher_v3_retreat_focused_v3_extend1\artifacts\trajectory_diagnostics_combined.json
```

当前 candidate：

```text
runs\20260622_154050_imitation_v3_candidate\checkpoints\policy.pt
```

当前 comparison evals：

```text
runs\20260622_160753_eval_imitation_v3_candidate_v2
runs\20260622_162334_eval_rule_baseline_v3_compare
runs\20260622_163914_eval_coverage_teacher_v3_compare
```

当前 strategy imitation candidate：

```text
runs\20260623_123449_strategy_imitation_v2_candidate_v2\checkpoints\policy.pt
```

当前 strategy imitation guarded eval：

```text
runs\20260623_123541_eval_strategy_imitation_v2_candidate_v2_hard_smoke
data\trajectories\strategy_imitation_v2_candidate_v2_hard_smoke
```

当前 AIBuild smoke：

```text
runs\20260623_144443_eval_aibuild_smoke_v1
data\trajectories\aibuild_smoke_army_v1
data\trajectories\aibuild_smoke_strategy_v1
return_code=0, Result.Tie, opponent_ai_build=Rush
```

当前 AIBuild Hard Terran 小矩阵：

```text
rule/no-op:
  runs\20260623_145519_eval_rule_aibuild_hard_terran_v1
  data\trajectories\rule_aibuild_hard_terran_strategy_v1
  Rush=Victory, Timing=Victory, Power=Defeat, Macro=Defeat, Air=Tie

coverage-teacher:
  runs\20260623_150115_eval_coverage_teacher_aibuild_hard_terran_v1
  data\trajectories\coverage_teacher_aibuild_hard_terran_strategy_v1
  Rush=Victory, Timing=Victory, Power=Defeat, Macro=Victory, Air=Tie
```

当前 tactic-aware coverage-teacher smoke：

```text
runs\20260623_152644_eval_tactic_coverage_aibuild_hard_terran_v1
data\trajectories\tactic_coverage_aibuild_hard_terran_strategy_v1
strategy_tactic_mode=rule
Macro=Victory, Power=Victory
tactic_diagnostics: rows_with_tactic_metadata=88/88, filter_change_rows=29
training_filter_change_rows=28
Power: TECH_POWER / TECH_ROBO -> PRODUCE_ARMY = 19 terminal-inclusive, 18 training rows
```

后续 tactic-aware recheck：

```text
v1 broad filter:
  runs\20260623_154821_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v1
  Rush=Defeat, Timing=Defeat, Power=Victory, Macro=Defeat, Air=Defeat

v2 safer spec filter:
  runs\20260623_160120_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v2
  Rush=Defeat, Timing=Victory, Power=Defeat, Macro=Tie, Air=Tie

v3 Power-targeted filter:
  runs\20260623_160924_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v3
  Rush=Victory, Timing=Defeat, Power=Defeat, Macro=Victory, Air=Defeat
  filter_changes=12, all on Power rows

Power-only recheck:
  no-filter coverage-teacher: runs\20260623_161606_eval_coverage_teacher_aibuild_hard_terran_power_recheck_v1
    1W / 1T / 1L
  Power-targeted tactic filter: runs\20260623_162021_eval_tactic_power_targeted_hard_terran_power_recheck_v1
    0W / 2T / 1L
```

当前结论：不要推广 tactic-aware filter，不要用当前 tactic-aware 数据训练；先继续离线诊断/细化 `TECH_POWER`。

当前 Power tactic 离线诊断：

```text
runs\20260623_power_tactic_diagnostics_v1\artifacts\power_tactic_diagnostics.txt
runs\20260623_power_tactic_diagnostics_v1\artifacts\power_tactic_diagnostics.json

inputs:
  data\trajectories\coverage_teacher_aibuild_hard_terran_power_recheck_strategy_v1
  data\trajectories\tactic_power_targeted_hard_terran_power_recheck_strategy_v1

6 files, 437 rows, 431 training rows
results: 1 Victory / 3 Tie / 2 Defeat
opponent_ai_build=Power for all training rows
filter changes: 46 training rows

counterfactual filter delta:
  TECH_ROBO -14
  BUILD_STATIC_DEFENSE -12
  ADD_GATEWAYS -16
  PRODUCE_ARMY +26
  FORGE_UPGRADES +11
  BOOST_WORKERS +5
```

离线解读：

```text
No-filter coverage-teacher 的 first TECH_ROBO 基本在 240.0-251.4s。
Power-targeted tactic filter 的一局 Tie 将 actual TECH_ROBO 推迟到 628.6s。
TECH_POWER 下最大改写是 TECH_ROBO -> PRODUCE_ARMY、ADD_GATEWAYS -> FORGE_UPGRADES、
BUILD_STATIC_DEFENSE -> PRODUCE_ARMY。

因此下一步不要扩大采集或训练，而是先做 TECH_POWER counterfactual/spec 调整：
保留一次及时 Robo，避免过度压制 gateway 节奏，再决定是否做 guarded Power-only paired comparison。
```

当前 revised `TECH_POWER` 离线调整：

```text
rl\tactics.py
tests\test_tactics.py
tests\test_tactic_strategy_policy.py

focused tests:
  .\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
  19 passed

full tests:
  .\.venv\Scripts\python.exe -m pytest -q
  150 passed
```

调整内容：

```text
TECH_POWER fallback 顺序改为：
  TECH_ROBO -> PRODUCE_ARMY -> ADD_GATEWAYS -> FORGE_UPGRADES

保留一次及时 Robo：
  no ready/pending Robo 且低矿高气时，TECH_ROBO 不再直接改成 PRODUCE_ARMY，
  而是 STAY_COURSE 等资源。

减少 ADD_GATEWAYS -> FORGE_UPGRADES：
  pending gateways 已 capped 且 no Robo 时，fallback 到 TECH_ROBO。
  Robo 已 started/ready 后，fallback 到 PRODUCE_ARMY。

static-defense repeat cap：
  threat 下 rejected BUILD_STATIC_DEFENSE fallback 到 PRODUCE_ARMY。

Observer/Immortal 缺失关注：
  ready Robo 已存在但 Observer/Immortal 仍缺时，不继续重复 TECH_ROBO，
  fallback 到 PRODUCE_ARMY，让产兵链路有机会补 Robo 单位。
```

当前 revised Power A/B：

```text
guard pid: 21392
场景: AcropolisLE / Hard / Terran / Power

no-filter coverage-teacher:
  runs\20260623_170757_20260623_eval_power_ab_no_filter_revised_v1
  data\trajectories\power_ab_no_filter_revised_strategy_v1
  1 Victory / 2 Defeat / 0 Tie
  return_code=0 for all games

revised Power tactic filter:
  runs\20260623_171134_20260623_eval_power_ab_revised_tactic_v1
  data\trajectories\power_ab_revised_tactic_strategy_v1
  0 Victory / 2 Defeat / 1 Tie
  return_code=0 for all games

diagnostics:
  runs\20260623_power_ab_revised_diagnostics_v1\artifacts\power_ab_diagnostics.txt
  runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_tactic_diagnostics.json
  runs\20260623_power_ab_revised_diagnostics_v1\artifacts\no_filter_power_diagnostics.json
  runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_power_diagnostics.json
```

诊断结论：

```text
Robo: revised 不再出现上一轮 628.6s 才 first TECH_ROBO 的灾难性延迟；
revised first TECH_ROBO 为 274.3 / 297.1 / 274.3s，ready_robo 为
331.4 / 354.3 / 331.4s。但 no-filter 的 first TECH_ROBO 仍更早，三局都是
251.4s。

Gateway: 未修好。no-filter ADD_GATEWAYS 很早出现（min 91.4s, avg 110.5s），
revised actual ADD_GATEWAYS 很晚（min 502.9s），且 SAFE_MACRO 仍有
ADD_GATEWAYS -> BOOST_WORKERS 7 次。

Observer/Immortal: revised 只有 Tie 那局补到 Observer 560.0s / Immortal
605.7s；两局 Defeat 仍然没有 Observer/Immortal。

当前结论：revised filter 只能保留为诊断实验，不能采 tactic-aware training data。
```

当前 strategy outcome diagnostics：

```text
runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.txt
runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.json

inputs:
  data\trajectories\power_ab_no_filter_revised_strategy_v1
  data\trajectories\power_ab_revised_tactic_strategy_v1

files=6, rows=393, training_rows=387
results=1 Victory / 1 Tie / 4 Defeat

source comparison:
  no-filter first ADD_GATEWAYS=91.4s, first TECH_ROBO=251.4s
  revised first actual ADD_GATEWAYS=502.9s, first TECH_ROBO=274.3s
  revised filter_change_rows=42

SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS:
  count=7, all before 240s.
  The rewrite suppresses early strategy-level ADD_GATEWAYS labels, though the
  default rule build loop still constructs some Gateways underneath.

TECH_POWER ADD_GATEWAYS -> PRODUCE_ARMY:
  count=2.
  No immediate +30/+60 army-count gain, but both samples show Robo-unit payoff
  by +120s; sample is too small to promote this rewrite.

TECH_POWER BUILD_STATIC_DEFENSE -> PRODUCE_ARMY:
  count=3.
  Threat cleared in all 3 samples by +30s.

RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY:
  count=8.
  Mixed early: at +60s, 7/8 cleared and 1/8 persisted; by +120s, 8/8 cleared.

Robo payoff:
  revised Defeat files had ready_robo at 331.4s / 354.3s but no Observer or
  Immortal. Rows after ready_robo show robo_idle_count=1 repeatedly; because
  later actions are mostly STAY_COURSE / PRODUCE_ARMY / FORGE_UPGRADES rather
  than TECH_ROBO, StrategyExecutor never gets the Robo-unit production trigger.
  The revised Tie file eventually produced Observer at 560.0s and Immortal at
  605.7s.
```

## 项目结构

```text
sc2-ai-bot/
├── bot/
│   ├── config.py
│   ├── protoss_rule_bot.py
│   ├── window_hider.py
│   └── managers/
├── rl/
├── scripts/
├── tests/
├── doc/
│   ├── CODEX.md
│   ├── README.md
│   ├── STATE.md
│   ├── STRATEGY_OUTCOME_PLAN.md
│   ├── TACTIC_POOL_PLAN.md
│   └── archive/
│       └── STRATEGY_EXPANSION_PLAN.md
├── run.py
└── requirements.txt
```

## 文档地图

- `doc\CODEX.md`：给 coding agent 的简洁交接文档。
- `doc\DEVELOPMENT_PLAN.md`：当前开发主线和 promotion gate。
- `doc\README.md`：项目概览和常用命令。
- `doc\STATE.md`：当前状态和关键实验账本。
- `doc\STRATEGY_OUTCOME_PLAN.md`：当前下一步方案，用离线 action-to-outcome 诊断修复 Gateway/Robo/生产收益闭环。
- `doc\TACTIC_POOL_PLAN.md`：AIBuild 评测/采集维度和 `TacticSpec` 战术池背景计划。
- `doc\archive\STRATEGY_EXPANSION_PLAN.md`：历史/背景计划，记录从 5 个军队动作拓展到低频宏观策略层的路线。
