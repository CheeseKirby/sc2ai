# CODEX.md

Concise handoff for coding agents.

## Read First

Read in this order:

1. `CODEX.md` - current truth, safety, next work.
2. `README.md` - user-facing commands and project overview.
3. `STATE.md` - compact experiment ledger.
4. `STRATEGY_EXPANSION_PLAN.md` - future strategy expansion plan, only when asked to expand strategy.

## Project Truth

- Project root: `D:\opus\data\raw\alpaca-gpt4\sc2\sc2-ai-bot`
- This folder is not a git repository.
- Current bot: Protoss rule bot plus high-level army policy infrastructure.
- Current learned-policy boundary: army-level decisions only.
- Current action space: 5 discrete actions.
- Current observation schema: v3, 26 numeric features.
- Current tests: `.\.venv\Scripts\python.exe -m pytest -q` -> `64 passed`.
- PPO is not implemented.
- `coverage-teacher` is for data collection, not a strong baseline strategy.
- Existing v1/v2 checkpoints are archived or diagnostic artifacts; current runtime intentionally rejects them.

## Safety Rule

The user may be at work. Do not expose SC2 or Battle.net windows.

Before any command that can launch SC2, verify/start the hidden-window guard:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Use `scripts/evaluate.py` or `scripts/safe_launch.py`. Do not run visible/debug mode unless explicitly requested.

## Completed Framework

- rule-based Protoss bot
- safe launch/window guard
- `ArmyPolicy` abstraction
- rule, coverage-teacher, checkpoint-backed, and experimental LLM army policies
- `ArmyMemory`
- schema v3 observation
- trajectory recording
- dataset loading
- diagnostics with observation feature stats
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

## Action Space

```text
0 RALLY
1 ATTACK_MAIN
2 RETREAT_HOME
3 DEFEND_BASE
4 HOLD
```

Economy, production, buildings, and workers remain rule-based.

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

## Recommended Next Work

Do not start PPO first.

Preferred next steps:

1. Continue larger guarded evals for `imitation_v3_candidate`.
2. Inspect online `RETREAT_HOME`, `ATTACK_MAIN`, and RALLY/HOLD dominance.
3. If online retreat remains token-count, improve retreat data/teacher.
4. If the goal is richer SC2 strategy, follow `STRATEGY_EXPANSION_PLAN.md` before PPO.
5. Consider PPO only after a stable learned baseline and reward/environment design exist.

## Common Commands

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Environment check:

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

Diagnose current candidate eval:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py data\trajectories\imitation_v3_candidate_eval_v2 --show-files
```

Evaluate current candidate:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Medium Hard Harder --opponents Protoss Terran Zerg --games-per-combo 2 --trajectory-dir data\trajectories\imitation_v3_candidate_eval_v2 --run-root runs --run-name eval_imitation_v3_candidate_v2 --policy-name imitation_v3_candidate --policy-checkpoint runs\20260622_154050_imitation_v3_candidate\checkpoints\policy.pt --record-decision-interval 16 --game-time-limit 900
```

## Environment Notes

- OS: Windows 11
- Python: 3.14.5 in `.venv`
- SC2 client: NetEase/China 5.0.15.96999, `Base96999`
- SC2 install path: sibling folder `..\StarCraft II`
- `bot/config.py` sets `SC2PATH`; import it before `sc2.*` imports in entry scripts.
- Installed RL dependencies include `gymnasium`, `stable-baselines3`, and `torch`.
- `rg.exe` may fail with `Access is denied`; use PowerShell `Get-ChildItem` / `Select-String` if needed.

