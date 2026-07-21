# 星际争霸 II AI Bot

这是一个 StarCraft II 神族 Bot 项目，目标是逐步做出一个能感知战局、解释思路、并根据局势调整打法的对战 AI。


```powershell
.\.venv\Scripts\python.exe scripts\benchmark_strategy_lab.py --policies heuristic random stay-course --episodes-per-scenario 4
```

输出包含标准实验元数据、汇总报告和逐步 JSONL 决策轨迹，覆盖 reward 分解、动作阻塞、fallback、延迟和分场景结果。求职展示导览见 `doc\GAME_AI_ENGINEERING_PORTFOLIO.md`。

## 当前状态

已完成：

- 神族规则 baseline Bot
- 隐藏窗口安全启动和 window guard
- `ArmyPolicy` 策略抽象
- 5 个高层军队动作
- schema v3 数值观测，26 维
- trajectory 记录和诊断
- 实验元数据记录
- 模型和 checkpoint 元数据记录
- observation normalization
- imitation learning
- checkpoint-backed 在线策略推理
- coverage-teacher 数据采集
- 实验性 LLM-backed army policy，可输出简短解释
- 实验性 LLM-backed strategy policy，失败时回退到 `STAY_COURSE`
- Strategy PPO Gymnasium adapter、严格 transition contract 和可解释 reward
- Stable-Baselines3 PPO 训练接线与 checkpoint strategy adapter
- 五类 deterministic surrogate 场景，可在无 SC2 环境下验证完整策略链路
- Strategy Lab 统一评测、故障回退、延迟指标和 JSONL 决策追踪

未完成：

- live BurnySC2 / SC2 PPO backend
- PPO reward 调参与真实训练
- PPO 在线评测和 promotion gate
- curriculum controller

当前 baseline 结论：

- `imitation_v3_candidate` 已经可以在线运行。
- 它可作为 schema v3 的 smoke / comparison baseline。
- 它还不是稳定的 PPO 初始化策略。
- 在线 `RETREAT_HOME` 仍然低频，`RALLY` / `HOLD` 仍然占主导。
- 在 v3 baseline 更稳定，或策略层被有计划地扩展之前，不建议开始 PPO。

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

查看 PPO 框架配置（不会启动 SC2，也不会训练）：

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo.py --backend surrogate --dry-run
```

LLM strategy 显式启用示例：

```powershell
.\.venv\Scripts\python.exe scripts\safe_launch.py --keep-guard -- --strategy-policy llm
```

PPO checkpoint 在线推理接线：

```text
--strategy-policy ppo --strategy-checkpoint <stable-baselines3 .zip>
```

PPO / LLM 框架边界和扩展点见 `doc\PPO_LLM_FRAMEWORK.md`。

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

安全跑一局：

```powershell
.\.venv\Scripts\python.exe scripts\safe_launch.py --keep-guard --guard-interval 0.02 -- --difficulty VeryEasy --opponent Protoss --hide-watch-seconds 120 --hide-watch-interval 0.02
```

诊断 trajectory：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py data\trajectories\imitation_v3_candidate_eval_v2 --show-files
```

用 trajectory 训练 imitation policy：

```powershell
.\.venv\Scripts\python.exe scripts\train_imitation.py data\trajectories\coverage_teacher_v3_retreat_focused_v3 --run-root runs --run-name imitation_v3_candidate --epochs 8 --batch-size 128 --class-weighting balanced
```

评测当前 v3 imitation checkpoint：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Medium Hard Harder --opponents Protoss Terran Zerg --games-per-combo 2 --trajectory-dir data\trajectories\imitation_v3_candidate_eval_v2 --run-root runs --run-name eval_imitation_v3_candidate_v2 --policy-name imitation_v3_candidate --policy-checkpoint runs\20260622_154050_imitation_v3_candidate\checkpoints\policy.pt --record-decision-interval 16 --game-time-limit 900
```

## 当前动作空间

```text
0 RALLY
1 ATTACK_MAIN
2 RETREAT_HOME
3 DEFEND_BASE
4 HOLD
```

当前学习策略只控制这些军队层面的动作。宏观经济和生产仍由规则 Bot 执行。

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
├── CODEX.md
├── README.md
├── STATE.md
├── STRATEGY_EXPANSION_PLAN.md
├── run.py
└── requirements.txt
```

## 文档地图

- `CODEX.md`：给 coding agent 的简洁交接文档。
- `README.md`：项目概览和常用命令。
- `STATE.md`：当前状态和关键实验账本。
- `STRATEGY_EXPANSION_PLAN.md`：未来从 5 个军队动作拓展到宏观策略层的计划。
