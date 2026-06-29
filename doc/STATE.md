# Project State

Compact current state and experiment ledger.

## Current Snapshot

- Project root: `D:\opus\data\raw\alpaca-gpt4\sc2\sc2-ai-bot`
- This folder is a git repository; the worktree may be dirty from ongoing feature work, so avoid destructive git operations.
- Latest recorded tests: `.\.venv\Scripts\python.exe -m pytest -q` -> `209 passed`.
- Latest recorded environment check: `.\.venv\Scripts\python.exe scripts\check_env.py` -> OK.
- SC2 client: NetEase/China `5.0.15.96999`, `Base96999`.
- SC2 launches must use hidden-window guard and `scripts/evaluate.py` / `scripts/safe_launch.py`.
- PPO is not implemented.
- Current learned-policy boundary: army-level decisions plus opt-in low-frequency strategy imitation.
- Default runtime remains rule/no-op unless a policy is explicitly selected.
- Current AIBuild status: official built-in AI `AIBuild` is wired into `run.py`, `scripts/evaluate.py`, eval summaries, experiment config, and army/strategy trajectory metadata.
- Current tactic status: first `TacticSpec` / `TacticState` / `RuleTacticSelector` skeleton is implemented and available only through explicit opt-in `--strategy-policy coverage-teacher --strategy-tactic-mode rule`; tactic-rule runtime is frozen and not promotable on current evidence.
- Current next plan: follow `doc\DEVELOPMENT_PLAN.md`; strategy execution observability, candidate audit, and replay-only runtime-patch gates are implemented. The frozen anti-air ready-static candidate failed both gates, so do not patch runtime from it.

## Safety

Before any SC2 launch:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Use `scripts/evaluate.py` or `scripts/safe_launch.py`. Do not expose SC2 or Battle.net windows unless explicitly requested.

## Architecture Snapshot

Current bot:

- `ProtossRuleBot`
- rule-based economy, production, buildings, workers
- pluggable high-level army policy

Policy implementations:

- `RuleArmyPolicy`
- `CoverageArmyPolicy`
- `RLArmyPolicy`
- experimental `LLMArmyPolicy`
- `RuleStrategyPolicy`
- `CoverageStrategyPolicy`
- `RLStrategyPolicy`

Current army action space:

```text
0 RALLY
1 ATTACK_MAIN
2 RETREAT_HOME
3 DEFEND_BASE
4 HOLD
```

Current observation schema:

- version: v3
- dimension: 26
- key additions over earlier schemas:
  - `attack_army_peak`
  - `army_lost_from_peak`
  - `army_lost_from_peak_ratio`
  - `army_count_delta`

Compatibility:

- v1/v2 trajectory rows can be defaulted for diagnostics or smoke work.
- v1/v2 checkpoints are intentionally incompatible with current runtime.

Current strategy observation schema:

- version: `strategy_v2`
- dimension: 40
- default strategy policy: `RuleStrategyPolicy`, which returns `STAY_COURSE`
- learned strategy policies are explicit opt-in via `--strategy-policy checkpoint`

## Current Candidate

Training trajectory:

```text
data\trajectories\coverage_teacher_v3_retreat_focused_v3
```

Combined training diagnostics:

```text
runs\20260622_151332_coverage_teacher_v3_retreat_focused_v3_extend1\artifacts\trajectory_diagnostics_combined.json
```

Combined training data:

```text
games: 72
rows: 16749
training_rows: 16677
terminal_rows: 72
observation_dim: 26
observation_schema_counts: v3=16749
rows_defaulted_observation_fields: 0
action_coverage: 100%
results: 32 Victory / 30 Defeat / 10 Tie
actions:
  RALLY=5892
  ATTACK_MAIN=1468
  RETREAT_HOME=33
  DEFEND_BASE=805
  HOLD=8479
```

Trained candidate:

```text
runs\20260622_154050_imitation_v3_candidate
runs\20260622_154050_imitation_v3_candidate\checkpoints\policy.pt
runs\20260622_154050_imitation_v3_candidate\artifacts\metrics.json
```

Training metrics:

```text
examples: 16677
observation_dim: 26
observation_schema_counts: v3=16677
rows_defaulted_observation_fields: 0
missing_action_names: none
class_weighting: balanced
train_accuracy: 0.999
validation_accuracy: 0.9985
validation_examples: 3335
validation RETREAT_HOME: 5/5
```

Training action counts:

```text
RALLY=5892
ATTACK_MAIN=1468
RETREAT_HOME=33
DEFEND_BASE=805
HOLD=8479
```

Per-action validation accuracy:

```text
RALLY=1.000
ATTACK_MAIN=0.987
RETREAT_HOME=1.000
DEFEND_BASE=1.000
HOLD=0.999
```

## Guarded Eval Ledger

### Smoke Eval

Run:

```text
runs\20260622_154143_eval_imitation_v3_candidate
```

Trajectory:

```text
data\trajectories\imitation_v3_candidate_eval
```

Results:

```text
games: 6
return_code: 6/6 zero
results: 2 Victory / 2 Defeat / 2 Tie
observation_schema_counts: v3=1448
rows_defaulted_observation_fields: 0
action_coverage: 100%
actions:
  RALLY=704
  ATTACK_MAIN=25
  RETREAT_HOME=1
  DEFEND_BASE=94
  HOLD=618
```

Conclusion: checkpoint loads and runs online; retreat remains low-count.

### Three-Way Comparison Eval

Same map/difficulty/opponent grid:

```text
map: AcropolisLE
difficulties: Medium / Hard / Harder
opponents: Protoss / Terran / Zerg
games_per_combo: 2
game_time_limit: 900
record_decision_interval: 16
```

#### imitation_v3_candidate

```text
run: runs\20260622_160753_eval_imitation_v3_candidate_v2
trajectory: data\trajectories\imitation_v3_candidate_eval_v2
diagnostics: runs\20260622_160753_eval_imitation_v3_candidate_v2\artifacts\trajectory_diagnostics.json
return_code: 18/18 zero
results: 8 Victory / 8 Defeat / 2 Tie
rows: 4189
training_rows: 4171
observation_schema_counts: v3=4189
rows_defaulted_observation_fields: 0
action_coverage: 100%
actions:
  RALLY=1991
  ATTACK_MAIN=352
  RETREAT_HOME=2
  DEFEND_BASE=237
  HOLD=1589
warning: low RETREAT_HOME count
```

#### rule baseline

```text
run: runs\20260622_162334_eval_rule_baseline_v3_compare
trajectory: data\trajectories\rule_baseline_eval_v3_compare
diagnostics: runs\20260622_162334_eval_rule_baseline_v3_compare\artifacts\trajectory_diagnostics.json
return_code: 18/18 zero
results: 6 Victory / 8 Defeat / 4 Tie
rows: 4556
training_rows: 4538
observation_schema_counts: v3=4556
rows_defaulted_observation_fields: 0
action_coverage: 40%
actions:
  RALLY=3017
  ATTACK_MAIN=1521
missing actions:
  RETREAT_HOME
  DEFEND_BASE
  HOLD
note: missing actions are expected for this narrow rule state machine.
```

#### coverage-teacher

```text
run: runs\20260622_163914_eval_coverage_teacher_v3_compare
trajectory: data\trajectories\coverage_teacher_eval_v3_compare
diagnostics: runs\20260622_163914_eval_coverage_teacher_v3_compare\artifacts\trajectory_diagnostics.json
return_code: 18/18 zero
results: 5 Victory / 9 Defeat / 4 Tie
rows: 4587
training_rows: 4569
observation_schema_counts: v3=4587
rows_defaulted_observation_fields: 0
action_coverage: 100%
actions:
  RALLY=1422
  ATTACK_MAIN=589
  RETREAT_HOME=3
  DEFEND_BASE=267
  HOLD=2288
warning: low RETREAT_HOME count
```

Comparison conclusion:

- All three evals completed with `return_code=0`.
- All trajectories are pure schema v3 with no defaulted fields.
- `imitation_v3_candidate` is not obviously weaker than rule in this sample.
- It is not a stable PPO initializer yet:
  - online `RETREAT_HOME` remains token-count
  - `RALLY/HOLD` dominate
  - richer strategy is still outside the current action space

## Historical Notes Worth Keeping

Earlier experiments showed:

- v1/v2 pipelines were useful framework smoke tests but are obsolete for current runtime checkpoints.
- early v2/v3 data produced only token retreat coverage.
- schema v3 and `ArmyMemory` were added specifically to expose attack peak/loss signals.
- `coverage-teacher` defaults were initially:
  - `RETREAT_PEAK_LOSS_RATIO=0.25`
  - `RETREAT_MIN_PEAK_ARMY=8`
  - `RETREAT_MIN_LOST_FROM_PEAK=3`
- the useful expanded training data came from a looser collection regime:
  - `--retreat-peak-loss-ratio 0.15`
  - `--retreat-min-lost-from-peak 2`
  - `--retreat-min-peak-army 8`

## Current Decision

Do not start PPO yet.

Current next work:

1. Implement `AIBuild` as an eval/data-collection dimension.
2. Store `opponent_ai_build` in eval records, experiment config, and army/strategy trajectory metadata.
3. Use that data to compare rule, coverage-teacher, and strategy policies by opponent build type.
4. Then design a local `TacticSpec` / `TacticState` / `TacticSelector` tactic pool.
5. Keep tactic behavior explicit opt-in and preserve default rule/no-op behavior.
6. Only consider PPO after stable army and strategy learned baselines, reward design, and environment boundaries exist.

## Documentation

- `doc\CODEX.md`: concise agent handoff.
- `doc\README.md`: project overview and commands.
- `doc\STATE.md`: this compact state ledger.
- `doc\STRATEGY_OUTCOME_PLAN.md`: current next plan for action-to-outcome diagnostics and guardrail-first tactic filtering.
- `doc\TACTIC_POOL_PLAN.md`: AIBuild and TacticSpec tactic-pool background plan.
- `doc\archive\STRATEGY_EXPANSION_PLAN.md`: original strategy expansion roadmap; useful background for implemented strategy-layer architecture.

## 2026-06-23 Strategy Expansion Phase 1/2

User request:

```text
参考 sc2-ai-bot/STRATEGY_EXPANSION_PLAN.md，进行策略扩展开发
```

Scope:

```text
No PPO.
No PPO/RL mainline architecture changes.
No SC2 launch.
Preserve default rule baseline, coverage-teacher, checkpoint policy, and LLM army policy behavior.
Keep ArmyAction v1 and schema v3 default path unchanged.
```

Implemented files:

```text
rl\strategy_actions.py
bot\managers\strategy_policy.py
bot\managers\rule_strategy_policy.py
bot\managers\strategy_executor.py
tests\test_strategy_policy.py
```

Modified files:

```text
bot\protoss_rule_bot.py
bot\managers\__init__.py
CODEX.md
README.md
STATE.md
```

StrategyAction v1:

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

Architecture changes:

```text
ProtossRuleBot now has a low-frequency StrategyPolicy slot.
STRATEGY_DECISION_INTERVAL defaults to 64 iterations.
Default strategy_policy is RuleStrategyPolicy.
RuleStrategyPolicy always returns STAY_COURSE.
StrategyExecutor executes only explicit macro intents.
ArmyPolicy, ArmyAction v1, schema v3 observation, trajectory format, and learned army checkpoint path were not changed.
```

StrategyExecutor initial capabilities:

```text
EXPAND: safely calls expand_now when below target bases and Nexus is affordable.
ADD_GATEWAYS: builds powered Gateways based on base count / target gateway count.
TECH_ROBO: after Cybernetics Core, builds Robotics Facility; ready Robo trains Observer before Immortal.
FORGE_UPGRADES: builds Forge, then researches Protoss ground weapons / armor level 1 when affordable.
BUILD_STATIC_DEFENSE: builds Shield Battery after Cybernetics Core or Photon Cannon after Forge, capped per base.
PRODUCE_ARMY: delegates to existing _train_army.
BOOST_WORKERS: trains Probes from idle ready Nexuses.
STAY_COURSE: no-op.
```

Tests added:

```text
StrategyAction int/name mapping.
RuleStrategyPolicy default no-op.
StrategyExecutor STAY_COURSE no-op.
EXPAND execution.
ADD_GATEWAYS powered build.
TECH_ROBO build and Robo unit training.
FORGE_UPGRADES build and research.
BUILD_STATIC_DEFENSE prerequisite behavior.
BOOST_WORKERS / PRODUCE_ARMY delegation.
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
74 passed in 2.05s
```

Conclusion:

```text
Strategy expansion phase 1 is in place and phase 2 has an initial rule executor skeleton.
Default runtime behavior remains behavior-preserving because RuleStrategyPolicy returns STAY_COURSE.
This is not yet a learned strategy system: no strategy observation schema, no strategy trajectory labels, no StrategyCoverageTeacher, and no strategy imitation candidate exist yet.
PPO remains not recommended.
```

## 2026-06-23 Hard Terran Strategy Compare

Scope:

```text
Continue from strategy_imitation_v2_candidate_v2 smoke eval.
Run a focused guarded Hard Terran comparison:
  v2 strategy checkpoint
  v1 strategy checkpoint
  default rule/no-op strategy
  coverage-teacher strategy
Do not touch PPO.
Do not change CoverageStrategyPolicy or StrategyExecutor.
Do not break default rule baseline.
```

Health checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results:

```text
pytest: 121 passed in 2.81s
check_env: all OK
```

Safety:

```text
Hidden-window guard was checked before every SC2 batch.
guard pid: 21392
All SC2 launches used scripts\evaluate.py -> scripts\safe_launch.py.
No visible run.py launch.
```

Commands:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --games-per-combo 2 --run-root runs --run-name strategy_v2_hard_terran_compare_v1 --policy-name strategy_v2_hard_terran_compare_v1 --strategy-policy checkpoint --strategy-checkpoint runs\20260623_123449_strategy_imitation_v2_candidate_v2\checkpoints\policy.pt --strategy-trajectory-dir data\trajectories\strategy_v2_hard_terran_compare_v1 --record-decision-interval 16 --game-time-limit 900

.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --games-per-combo 2 --run-root runs --run-name strategy_v1_hard_terran_compare_v1 --policy-name strategy_v1_hard_terran_compare_v1 --strategy-policy checkpoint --strategy-checkpoint runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt --strategy-trajectory-dir data\trajectories\strategy_v1_hard_terran_compare_v1 --record-decision-interval 16 --game-time-limit 900

.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --games-per-combo 2 --run-root runs --run-name rule_strategy_hard_terran_compare_v1 --policy-name rule_strategy_hard_terran_compare_v1 --strategy-policy rule --strategy-trajectory-dir data\trajectories\rule_strategy_hard_terran_compare_v1 --record-decision-interval 16 --game-time-limit 900

.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --games-per-combo 2 --run-root runs --run-name coverage_teacher_strategy_hard_terran_compare_v2 --policy-name coverage_teacher_strategy_hard_terran_compare_v2 --strategy-policy coverage-teacher --strategy-trajectory-dir data\trajectories\coverage_teacher_strategy_hard_terran_compare_v2 --record-decision-interval 16 --game-time-limit 900
```

Sleep/interruption handling:

```text
The first coverage-teacher compare attempt was interrupted by PC sleep:
  run: runs\20260623_125257_coverage_teacher_strategy_hard_terran_compare_v1
  eval rows: 1
  result: 1 Victory
  duration_seconds: 2576.5
  second trajectory file: 0 bytes

Residual evaluate/safe_launch/run.py/SC2 processes were inspected via
Win32_Process and then stopped. No python/SC2 processes remained afterward.

The interrupted v1 coverage-teacher compare should not be used for comparison.
It was replaced by coverage_teacher_strategy_hard_terran_compare_v2 using a fresh
run and trajectory directory.
```

Run paths:

```text
v2:
  run: runs\20260623_124528_strategy_v2_hard_terran_compare_v1
  eval: runs\20260623_124528_strategy_v2_hard_terran_compare_v1\artifacts\eval.jsonl
  trajectory: data\trajectories\strategy_v2_hard_terran_compare_v1

v1:
  run: runs\20260623_124835_strategy_v1_hard_terran_compare_v1
  eval: runs\20260623_124835_strategy_v1_hard_terran_compare_v1\artifacts\eval.jsonl
  trajectory: data\trajectories\strategy_v1_hard_terran_compare_v1

rule:
  run: runs\20260623_125103_rule_strategy_hard_terran_compare_v1
  eval: runs\20260623_125103_rule_strategy_hard_terran_compare_v1\artifacts\eval.jsonl
  trajectory: data\trajectories\rule_strategy_hard_terran_compare_v1

coverage-teacher:
  run: runs\20260623_134344_coverage_teacher_strategy_hard_terran_compare_v2
  eval: runs\20260623_134344_coverage_teacher_strategy_hard_terran_compare_v2\artifacts\eval.jsonl
  trajectory: data\trajectories\coverage_teacher_strategy_hard_terran_compare_v2
```

Eval results:

```text
v2:
  return_code: 0 for both
  Result.Defeat: 2

v1:
  return_code: 0 for both
  Result.Victory: 1
  Result.Defeat: 1

rule:
  return_code: 0 for both
  Result.Victory: 2

coverage-teacher:
  return_code: 0 for both
  Result.Victory: 1
  Result.Tie: 1
```

Strategy trajectory diagnostics:

```text
v2:
  rows=137, training_rows=135
  schema=strategy_v2 only
  rows_defaulted_observation_fields=0
  action_coverage=100%
  actions:
    STAY_COURSE=46
    EXPAND=4
    ADD_GATEWAYS=12
    TECH_ROBO=28
    FORGE_UPGRADES=5
    BUILD_STATIC_DEFENSE=23
    PRODUCE_ARMY=6
    BOOST_WORKERS=11

v1:
  rows=109, training_rows=107
  schema=strategy_v2 only
  rows_defaulted_observation_fields=0
  action_coverage=87.5%
  missing_actions: ADD_GATEWAYS
  actions:
    STAY_COURSE=42
    EXPAND=4
    TECH_ROBO=6
    FORGE_UPGRADES=8
    BUILD_STATIC_DEFENSE=18
    PRODUCE_ARMY=14
    BOOST_WORKERS=15

rule:
  rows=86, training_rows=84
  schema=strategy_v2 only
  rows_defaulted_observation_fields=0
  actions: STAY_COURSE=84

coverage-teacher:
  rows=123, training_rows=121
  schema=strategy_v2 only
  rows_defaulted_observation_fields=0
  action_coverage=100%
  actions:
    STAY_COURSE=58
    EXPAND=1
    ADD_GATEWAYS=14
    TECH_ROBO=13
    FORGE_UPGRADES=2
    BUILD_STATIC_DEFENSE=14
    PRODUCE_ARMY=10
    BOOST_WORKERS=9
```

Timing highlights:

```text
v2:
  ADD_GATEWAYS first=91.4 avg=371.4
  TECH_ROBO first=388.6 avg=444.9
  BUILD_STATIC_DEFENSE first=297.1 avg=518.3
  threat_actions: TECH_ROBO=2, BUILD_STATIC_DEFENSE=23
  results: 2 Defeat

v1:
  TECH_ROBO first=354.3 avg=360.0
  BUILD_STATIC_DEFENSE first=388.6 avg=505.4
  threat_actions: BUILD_STATIC_DEFENSE=17
  results: 1 Victory / 1 Defeat

rule:
  STAY_COURSE only
  results: 2 Victory

coverage-teacher:
  ADD_GATEWAYS first=91.4 avg=389.4
  TECH_ROBO first=308.6 avg=349.0
  BUILD_STATIC_DEFENSE first=400.0 avg=669.4
  threat_actions: BUILD_STATIC_DEFENSE=14
  results: 1 Victory / 1 Tie
```

Agreement highlights:

```text
v2 against current teacher:
  checkpoint_vs_teacher_accuracy=0.704
  checkpoint_vs_stored_accuracy=1.000
  tech_robo_needed bucket=0.667
  gateway_scaling_needed bucket=0.474

v1 against current teacher:
  checkpoint_vs_teacher_accuracy=0.523
  checkpoint_vs_stored_accuracy=1.000
  tech_robo_needed bucket=0.440
  gateway_scaling_needed bucket=0.143

coverage-teacher against v2 checkpoint:
  stored_vs_teacher_accuracy=1.000
  v2 checkpoint_vs_teacher_accuracy=0.760
  tech_robo_needed bucket=0.833
  gateway_scaling_needed bucket=0.889
```

Interpretation:

```text
The focused Hard Terran comparison does not support promoting
strategy_imitation_v2_candidate_v2.

v2 is better than v1 offline, but in online Hard Terran it went 0/2.
The default rule/no-op strategy went 2/2 with STAY_COURSE only.
Coverage-teacher went 1 Victory / 1 Tie.

This suggests the problem is not simply missing TECH_ROBO or action collapse.
The learned strategy layer and/or StrategyExecutor side effects may be stealing
resources or perturbing the rule bot's naturally strong Hard Terran timing.
Likely suspects:
  expansion/tech/static-defense actions changing resource allocation
  repeated ADD_GATEWAYS while pending_gateways > 0
  BUILD_STATIC_DEFENSE under threat arriving when army production/micro timing
    matters more
  StrategyExecutor macro actions competing with rule production priorities
```

Recommended next step:

```text
Do not train again yet.
Do not modify CoverageStrategyPolicy based only on this.
Do not enter PPO.

Next development should inspect StrategyExecutor side effects in Hard Terran:
  compare minerals/vespene/supply/army_count/ready_gateways/ready_robo over time
    between rule, v2, and coverage-teacher
  add a resource/timing diagnostic for strategy evals
  consider guarded executor constraints so strategy actions cannot overrule or
    starve the rule baseline's proven Hard Terran production cadence
```

## 2026-06-23 Strategy Agreement Diagnostics

Scope:

```text
Continue strategy-layer diagnostics after timing analysis.
Add offline teacher-vs-imitation agreement diagnostics before teacher changes
or new data collection.
No PPO.
No SC2 launch in this pass.
No rule baseline behavior change.
CoverageStrategyPolicy unchanged.
```

Code changes:

```text
rl\strategy_agreement_diagnostics.py
  Added offline diagnostics comparing:
    stored trajectory action
    current CoverageStrategyPolicy action from the same observation
    checkpoint prediction from the same observation
  Reports:
    overall stored_vs_teacher / checkpoint_vs_teacher / checkpoint_vs_stored accuracy
    action counts for stored / teacher / checkpoint
    teacher-to-checkpoint confusion matrix
    mismatch counts grouped by teacher action
    time buckets: 0-180, 180-360, 360-540, 540-720, 720+
    state buckets: difficulty/opponent, base threat, armored/cloaked signal,
      gateway_scaling_needed, tech_robo_needed, pending_* states, low saturation
    per-file summaries for Hard failure analysis

scripts\diagnose_strategy_agreement.py
  Added CLI with --checkpoint, --show-buckets, --show-files, and --json-output.

tests\test_strategy_agreement_diagnostics.py
  Added unit tests for time/state/file summaries and report formatting.
```

Commands:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_agreement_diagnostics.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_agreement_diagnostics.py tests\test_strategy_timing_diagnostics.py tests\test_strategy_datasets.py tests\test_rl_strategy_policy.py tests\test_coverage_strategy_policy.py -q
.\.venv\Scripts\python.exe scripts\diagnose_strategy_agreement.py data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 --checkpoint runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt --show-buckets --show-files --json-output runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\strategy_agreement_diagnostics.json
.\.venv\Scripts\python.exe scripts\diagnose_strategy_agreement.py data\trajectories\strategy_imitation_v2_candidate_eval_v1 --checkpoint runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt --show-buckets --show-files --json-output runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\strategy_agreement_diagnostics.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Validation:

```text
focused agreement tests: 2 passed
related strategy tests: 25 passed
full suite: 121 passed in 2.84s
check_env: all OK
SC2 launched: no
Hidden-window guard needed: no, because no SC2 command was run
PPO touched: no
CoverageStrategyPolicy changed: no
```

Artifacts:

```text
coverage-teacher agreement diagnostics:
runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\strategy_agreement_diagnostics.json

candidate online-state agreement diagnostics:
runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\strategy_agreement_diagnostics.json
```

Coverage-teacher data agreement:

```text
inputs: data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1
checkpoint: runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt
files: 18
rows: 909
observation_schema_counts: strategy_v2=909
rows_defaulted_observation_fields: 0
stored_vs_teacher_accuracy: 1.000
checkpoint_vs_teacher_accuracy: 0.589
checkpoint_vs_stored_accuracy: 0.589

teacher/stored action counts:
  STAY_COURSE=490
  EXPAND=11
  ADD_GATEWAYS=69
  TECH_ROBO=130
  FORGE_UPGRADES=24
  BUILD_STATIC_DEFENSE=50
  PRODUCE_ARMY=69
  BOOST_WORKERS=66

checkpoint action counts:
  STAY_COURSE=353
  EXPAND=48
  ADD_GATEWAYS=67
  TECH_ROBO=38
  FORGE_UPGRADES=115
  BUILD_STATIC_DEFENSE=70
  PRODUCE_ARMY=79
  BOOST_WORKERS=139

mismatch_counts_by_teacher:
  STAY_COURSE=173
  ADD_GATEWAYS=35
  TECH_ROBO=96
  FORGE_UPGRADES=3
  BUILD_STATIC_DEFENSE=1
  PRODUCE_ARMY=51
  BOOST_WORKERS=15

important buckets:
  0-180: rows=288, checkpoint_vs_teacher=0.708, ADD_GATEWAYS mismatches=32
  180-360: rows=288, checkpoint_vs_teacher=0.524, TECH_ROBO mismatches=55
  360-540: rows=210, checkpoint_vs_teacher=0.481, TECH_ROBO mismatches=33
  difficulty:Hard: rows=349, checkpoint_vs_teacher=0.610
  gateway_scaling_needed: rows=110, checkpoint_vs_teacher=0.636, ADD_GATEWAYS mismatches=35
  tech_robo_needed: rows=185, checkpoint_vs_teacher=0.438, TECH_ROBO mismatches=96
  pending_robo: rows=58, checkpoint_vs_teacher=0.207
```

Candidate online-state agreement:

```text
inputs: data\trajectories\strategy_imitation_v2_candidate_eval_v1
checkpoint: runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt
files: 9
rows: 487
observation_schema_counts: strategy_v2=487
rows_defaulted_observation_fields: 0
stored_vs_teacher_accuracy: 0.511
checkpoint_vs_teacher_accuracy: 0.511
checkpoint_vs_stored_accuracy: 1.000

teacher counterfactual action counts:
  STAY_COURSE=154
  EXPAND=7
  ADD_GATEWAYS=34
  TECH_ROBO=143
  FORGE_UPGRADES=28
  BUILD_STATIC_DEFENSE=55
  PRODUCE_ARMY=33
  BOOST_WORKERS=33

checkpoint/stored action counts:
  STAY_COURSE=148
  EXPAND=26
  ADD_GATEWAYS=71
  TECH_ROBO=22
  FORGE_UPGRADES=49
  BUILD_STATIC_DEFENSE=63
  PRODUCE_ARMY=35
  BOOST_WORKERS=73

mismatch_counts_by_teacher:
  STAY_COURSE=41
  ADD_GATEWAYS=22
  TECH_ROBO=124
  FORGE_UPGRADES=20
  BUILD_STATIC_DEFENSE=1
  PRODUCE_ARMY=24
  BOOST_WORKERS=6

important buckets:
  180-360: rows=144, checkpoint_vs_teacher=0.479, TECH_ROBO mismatches=55
  360-540: rows=116, checkpoint_vs_teacher=0.388, TECH_ROBO mismatches=45
  720+: rows=34, checkpoint_vs_teacher=0.294, TECH_ROBO mismatches=16
  difficulty:Hard: rows=191, checkpoint_vs_teacher=0.607
  opponent:Terran: rows=167, checkpoint_vs_teacher=0.455, TECH_ROBO mismatches=55
  gateway_scaling_needed: rows=79, checkpoint_vs_teacher=0.430
  tech_robo_needed: rows=205, checkpoint_vs_teacher=0.302, TECH_ROBO mismatches=124
  pending_bases: rows=42, checkpoint_vs_teacher=0.214
  pending_robo: rows=8, checkpoint_vs_teacher=0.250
```

Hard candidate file summaries:

```text
Hard Protoss defeat:
  file: data\trajectories\strategy_imitation_v2_candidate_eval_v1\20260623_114025_AcropolisLE_Hard_Protoss_001.jsonl
  rows=70
  checkpoint_vs_teacher=0.686
  mismatches_by_teacher: STAY_COURSE=8, ADD_GATEWAYS=4, TECH_ROBO=3,
    BUILD_STATIC_DEFENSE=1, PRODUCE_ARMY=5, BOOST_WORKERS=1

Hard Terran defeat:
  file: data\trajectories\strategy_imitation_v2_candidate_eval_v1\20260623_114148_AcropolisLE_Hard_Terran_001.jsonl
  rows=47
  checkpoint_vs_teacher=0.660
  mismatches_by_teacher: STAY_COURSE=3, ADD_GATEWAYS=7, TECH_ROBO=5, BOOST_WORKERS=1

Hard Zerg defeat:
  file: data\trajectories\strategy_imitation_v2_candidate_eval_v1\20260623_114240_AcropolisLE_Hard_Zerg_001.jsonl
  rows=74
  checkpoint_vs_teacher=0.500
  mismatches_by_teacher: STAY_COURSE=15, ADD_GATEWAYS=3, TECH_ROBO=4,
    FORGE_UPGRADES=9, PRODUCE_ARMY=4, BOOST_WORKERS=2
```

Interpretation:

```text
This diagnostic does not support changing CoverageStrategyPolicy right now.
The recommended teacher data exactly matches the current teacher logic
(stored_vs_teacher_accuracy=1.000).

The checkpoint has broad offline disagreement with teacher labels, especially:
  TECH_ROBO under tech_robo_needed states
  pending_robo states
  early ADD_GATEWAYS / gateway_scaling_needed states
  late-game candidate online states

The candidate online trace confirms the checkpoint is deterministic/reproducible
against stored online actions (checkpoint_vs_stored=1.000), so the gap is not
an evaluation logging mismatch. It is learned-policy behavior.
```

Recommended next step:

```text
Do not enter PPO.
Do not modify CoverageStrategyPolicy yet.
Collect a new Hard-focused strategy_v2 coverage-teacher dataset using a fresh
directory/run name, then run:
  diagnose_trajectories --kind strategy
  diagnose_strategy_timing
  diagnose_strategy_agreement
Only train strategy_imitation_v2_candidate_v2 after the new data has healthy
schema/action coverage and agreement/timing diagnostics support it.
```

## 2026-06-23 Hard-Focused Strategy Data and Candidate v2

Scope:

```text
Use the agreement diagnostic conclusion to collect a small Hard-focused
strategy_v2 coverage-teacher extension, diagnose it, combine it with the
recommended pending-tuned teacher data, train strategy_imitation_v2_candidate_v2,
and run a tiny guarded Hard smoke eval.
No PPO.
CoverageStrategyPolicy unchanged.
Default rule baseline unchanged.
Strategy checkpoint remains explicit opt-in.
```

Safety:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Result:

```text
hidden-window guard pid: 21392
SC2 was launched only through scripts\evaluate.py / scripts\safe_launch.py.
No visible run.py was used.
```

Hard-focused coverage-teacher collection command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Protoss Terran Zerg --games-per-combo 2 --run-root runs --run-name strategy_coverage_teacher_v2_hard_focus_v1 --tag strategy-v2 --tag coverage-strategy --tag hard-focus --policy-name strategy_coverage_teacher_v2_hard_focus_v1 --strategy-policy coverage-teacher --strategy-trajectory-dir data\trajectories\strategy_coverage_teacher_v2_hard_focus_v1 --record-decision-interval 16 --game-time-limit 900
```

Hard-focused coverage-teacher paths:

```text
run:
runs\20260623_122724_strategy_coverage_teacher_v2_hard_focus_v1

eval:
runs\20260623_122724_strategy_coverage_teacher_v2_hard_focus_v1\artifacts\eval.jsonl

strategy trajectories:
data\trajectories\strategy_coverage_teacher_v2_hard_focus_v1

diagnostics:
runs\20260623_122724_strategy_coverage_teacher_v2_hard_focus_v1\artifacts\strategy_trajectory_diagnostics.json
runs\20260623_122724_strategy_coverage_teacher_v2_hard_focus_v1\artifacts\strategy_timing_diagnostics.json
runs\20260623_122724_strategy_coverage_teacher_v2_hard_focus_v1\artifacts\strategy_agreement_diagnostics.json
```

Hard-focused coverage-teacher result:

```text
games: 6
return_code: 0 for all
Victory: 4
Defeat: 1
Tie: 1
schema: strategy_v2 only
rows: 344
training_rows: 338
rows_defaulted_observation_fields: 0
action_coverage: 100%
STAY_COURSE: 183
EXPAND: 4
ADD_GATEWAYS: 27
TECH_ROBO: 32
FORGE_UPGRADES: 8
BUILD_STATIC_DEFENSE: 33
PRODUCE_ARMY: 34
BOOST_WORKERS: 17
low_count_actions (<10): EXPAND, FORGE_UPGRADES
```

Hard-focused timing summary:

```text
ADD_GATEWAYS: count=27 first=91.4 avg=332.7
TECH_ROBO: count=32 first=251.4 avg=406.4
BUILD_STATIC_DEFENSE: count=33 first=400.0 avg=571.4
threat_action_counts: BUILD_STATIC_DEFENSE=33
TECH_ROBO before armored/cloaked signal in all signal files:
  armored_signal: files_with_signal=4, tech_before=4, no_tech=0
  cloaked_signal: files_with_signal=3, tech_before=3, no_tech=0
hard_defeat_files:
  data\trajectories\strategy_coverage_teacher_v2_hard_focus_v1\20260623_122854_AcropolisLE_Hard_Protoss_002.jsonl
```

Combined data diagnostics command:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 data\trajectories\strategy_coverage_teacher_v2_hard_focus_v1 --kind strategy --show-files --json-output runs\20260623_122724_strategy_coverage_teacher_v2_hard_focus_v1\artifacts\strategy_trajectory_diagnostics_combined.json
.\.venv\Scripts\python.exe scripts\diagnose_strategy_timing.py data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 data\trajectories\strategy_coverage_teacher_v2_hard_focus_v1 --json-output runs\20260623_122724_strategy_coverage_teacher_v2_hard_focus_v1\artifacts\strategy_timing_diagnostics_combined.json
.\.venv\Scripts\python.exe scripts\diagnose_strategy_agreement.py data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 data\trajectories\strategy_coverage_teacher_v2_hard_focus_v1 --checkpoint runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt --show-buckets --json-output runs\20260623_122724_strategy_coverage_teacher_v2_hard_focus_v1\artifacts\strategy_agreement_diagnostics_combined_vs_v1.json
```

Combined data result:

```text
inputs:
  data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1
  data\trajectories\strategy_coverage_teacher_v2_hard_focus_v1

games: 24
rows: 1271
training_rows: 1247
terminal_rows: 24
schema: strategy_v2 only
rows_defaulted_observation_fields: 0
action_coverage: 100%
low_count_actions: <none>
results: 17 Victory / 4 Defeat / 3 Tie

actions:
  STAY_COURSE=673
  EXPAND=15
  ADD_GATEWAYS=96
  TECH_ROBO=162
  FORGE_UPGRADES=32
  BUILD_STATIC_DEFENSE=83
  PRODUCE_ARMY=103
  BOOST_WORKERS=83
```

Combined timing summary:

```text
ADD_GATEWAYS: count=96 first=91.4 avg=318.0
TECH_ROBO: count=162 first=251.4 avg=390.8
BUILD_STATIC_DEFENSE: count=83 first=400.0 avg=538.7
threat_action_counts:
  STAY_COURSE=7
  TECH_ROBO=13
  FORGE_UPGRADES=1
  BUILD_STATIC_DEFENSE=83
  BOOST_WORKERS=4
TECH_ROBO signal timing:
  armored_signal: files_with_signal=17, tech_before=17, no_tech=0
  cloaked_signal: files_with_signal=9, tech_before=9, no_tech=0
```

Combined agreement against v1 checkpoint:

```text
checkpoint: runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt
stored_vs_teacher_accuracy: 1.000
checkpoint_vs_teacher_accuracy: 0.576
tech_robo_needed bucket: 0.403
pending_robo bucket: 0.217
gateway_scaling_needed bucket: 0.618
```

Training command:

```powershell
.\.venv\Scripts\python.exe scripts\train_strategy_imitation.py data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 data\trajectories\strategy_coverage_teacher_v2_hard_focus_v1 --run-root runs --run-name strategy_imitation_v2_candidate_v2 --tag strategy-v2 --tag hard-focus --tag agreement-diagnosed --epochs 16 --batch-size 128 --class-weighting balanced
```

Training paths:

```text
run:
runs\20260623_123449_strategy_imitation_v2_candidate_v2

checkpoint:
runs\20260623_123449_strategy_imitation_v2_candidate_v2\checkpoints\policy.pt

metrics:
runs\20260623_123449_strategy_imitation_v2_candidate_v2\artifacts\metrics.json

train-data agreement:
runs\20260623_123449_strategy_imitation_v2_candidate_v2\artifacts\strategy_agreement_diagnostics_train_combined.json
```

Training result:

```text
examples: 1247
observation_dim: 40
schema: strategy_v2 only
rows_defaulted_observation_fields: 0
missing_action_names: []
class_weighting: balanced
train_accuracy: 0.761
validation_accuracy: 0.711

per_action_accuracy_by_name:
  STAY_COURSE: 0.727
  EXPAND: 1.000
  ADD_GATEWAYS: 0.955
  TECH_ROBO: 0.643
  FORGE_UPGRADES: 0.667
  BUILD_STATIC_DEFENSE: 1.000
  PRODUCE_ARMY: 0.400
  BOOST_WORKERS: 0.600
```

v2 agreement on combined teacher data:

```text
checkpoint_vs_teacher_accuracy: 0.751

Improvement vs v1 on same combined data:
  overall: 0.576 -> 0.751
  tech_robo_needed bucket: 0.403 -> 0.835
  pending_robo bucket: 0.217 -> 0.639
  gateway_scaling_needed bucket: 0.618 -> 0.882

Remaining weak spots:
  PRODUCE_ARMY validation accuracy: 0.400
  BOOST_WORKERS validation accuracy: 0.600
  pending_bases bucket agreement: 0.587
```

v2 Hard smoke eval command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Protoss Terran Zerg --games-per-combo 1 --run-root runs --run-name eval_strategy_imitation_v2_candidate_v2_hard_smoke --policy-name strategy_imitation_v2_candidate_v2 --strategy-policy checkpoint --strategy-checkpoint runs\20260623_123449_strategy_imitation_v2_candidate_v2\checkpoints\policy.pt --strategy-trajectory-dir data\trajectories\strategy_imitation_v2_candidate_v2_hard_smoke --record-decision-interval 16 --game-time-limit 900
```

v2 Hard smoke paths:

```text
run:
runs\20260623_123541_eval_strategy_imitation_v2_candidate_v2_hard_smoke

eval:
runs\20260623_123541_eval_strategy_imitation_v2_candidate_v2_hard_smoke\artifacts\eval.jsonl

strategy trajectories:
data\trajectories\strategy_imitation_v2_candidate_v2_hard_smoke

diagnostics:
runs\20260623_123541_eval_strategy_imitation_v2_candidate_v2_hard_smoke\artifacts\strategy_trajectory_diagnostics.json
runs\20260623_123541_eval_strategy_imitation_v2_candidate_v2_hard_smoke\artifacts\strategy_timing_diagnostics.json
runs\20260623_123541_eval_strategy_imitation_v2_candidate_v2_hard_smoke\artifacts\strategy_agreement_diagnostics.json
```

v2 Hard smoke result:

```text
games: 3
return_code: 0 for all
Victory: 2
Defeat: 1
schema: strategy_v2 only
rows: 164
training_rows: 161
rows_defaulted_observation_fields: 0
action_coverage: 100%
actions:
  STAY_COURSE=45
  EXPAND=5
  ADD_GATEWAYS=22
  TECH_ROBO=46
  FORGE_UPGRADES=4
  BUILD_STATIC_DEFENSE=12
  PRODUCE_ARMY=8
  BOOST_WORKERS=19

per-game:
  Hard Protoss: Victory
  Hard Terran: Defeat
  Hard Zerg: Victory
```

v2 Hard smoke timing/agreement:

```text
timing:
  ADD_GATEWAYS first=91.4 avg=393.2
  TECH_ROBO first=217.1 avg=366.5
  BUILD_STATIC_DEFENSE first=262.9 avg=569.5
  armored_signal files=3, tech_before=3, no_tech=0

agreement:
  checkpoint_vs_teacher_accuracy: 0.733
  checkpoint_vs_stored_accuracy: 1.000
  tech_robo_needed bucket: 0.846
  base_under_threat bucket: 1.000

Hard Terran loss:
  file: data\trajectories\strategy_imitation_v2_candidate_v2_hard_smoke\20260623_123629_AcropolisLE_Hard_Terran_001.jsonl
  checkpoint_vs_teacher_accuracy: 0.576
  timeline includes early TECH_ROBO at 217-240, then EXPAND, Forge,
  long ADD_GATEWAYS 434-560, and late static defense/BOOST_WORKERS.
```

Conclusion:

```text
strategy_imitation_v2_candidate_v2 is a better offline candidate than v1 and
passed a tiny guarded Hard smoke eval without action collapse.

It should not yet be promoted to stable strategy baseline:
  only 3 online v2 smoke games were run
  Hard Terran still lost
  PRODUCE_ARMY / BOOST_WORKERS remain weaker in validation

Next step:
  run a larger guarded same-scenario comparison for v2, v1, rule, and
  coverage-teacher, prioritizing Hard Terran and then the original
  Easy/Medium/Hard x three-race sample.
PPO remains not recommended.
```

## 2026-06-23 Strategy Imitation v2 Candidate + Guarded Data Collection

User request:

```text
好。继续策略开发与数据收集
```

Scope:

```text
Continue strategy development and data collection.
No PPO.
No PPO/RL mainline architecture changes.
Keep default rule baseline behavior-preserving.
Use hidden-window guard before SC2 launch.
Do not mix old data directories.
```

Code changes:

```text
rl\strategy_datasets.py
  Added strategy-specific JSONL dataset loader for rows with:
    strategy_observation
    strategy_action
    strategy_action_name
  Supports strategy_v2 directly and defaulting old strategy_v1 rows for diagnostics/training compatibility.

rl\strategy_checkpoints.py
  Added strategy checkpoint save/load helpers.
  Metadata validates:
    policy_family=strategy
    observation_schema_version=strategy_v2
    strategy observation field order
    StrategyAction names
    normalizer fields/schema

rl\strategy_imitation.py
  Added StrategyImitationTrainConfig, StrategyImitationTrainMetrics,
  and train_strategy_imitation_policy().
  Uses strategy action_dim=8 and strategy_v2 observation_dim=40.

scripts\train_strategy_imitation.py
  Added CLI for strategy imitation training.

bot\managers\rl_strategy_policy.py
  Added checkpoint-backed RLStrategyPolicy.
  Runtime inference builds strategy_v2 observation and returns StrategyAction.

run.py
scripts\evaluate.py
  Added explicit strategy checkpoint path:
    --strategy-policy checkpoint
    --strategy-checkpoint <path>
    --strategy-device cpu
  Kept army --policy-checkpoint semantics unchanged.

rl\normalization.py
  Generalized ObservationNormalizer to accept explicit field/schema metadata.
  Army defaults remain unchanged.

tests
  Added strategy dataset, strategy imitation, and RLStrategyPolicy tests.
  Updated evaluate tests for strategy checkpoint forwarding.
```

Validation before training:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results:

```text
pytest before training: 108 passed in 19.26s
check_env: all OK
```

Strategy training input re-diagnosis:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 --kind strategy --show-files --json-output runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\strategy_trajectory_diagnostics_rerun_for_imitation.json
```

Diagnostics output:

```text
diagnostics json: runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\strategy_trajectory_diagnostics_rerun_for_imitation.json
files: 18
rows: 927
training_rows: 909
terminal_rows: 18
observation_dim: 40
observation_schemas: strategy_v2=927
rows_defaulted_observation_fields: 0
action_coverage: 100%
missing_actions: none
low_count_actions: none
results: Victory=13, Defeat=3, Tie=2
```

Input action distribution:

```text
STAY_COURSE: 490
EXPAND: 11
ADD_GATEWAYS: 69
TECH_ROBO: 130
FORGE_UPGRADES: 24
BUILD_STATIC_DEFENSE: 50
PRODUCE_ARMY: 69
BOOST_WORKERS: 66
```

Training command:

```powershell
.\.venv\Scripts\python.exe scripts\train_strategy_imitation.py data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 --run-root runs --run-name strategy_imitation_v2_candidate --tag strategy-v2 --tag imitation --epochs 12 --batch-size 128 --class-weighting balanced
```

Training artifacts:

```text
run: runs\20260623_113345_strategy_imitation_v2_candidate
checkpoint: runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt
metrics: runs\20260623_113345_strategy_imitation_v2_candidate\artifacts\metrics.json
normalizer: runs\20260623_113345_strategy_imitation_v2_candidate\artifacts\normalizer.json
```

Training metrics:

```text
examples: 909
train_examples: 727
validation_examples: 182
observation_dim: 40
observation_schema_counts: strategy_v2=909
rows_defaulted_observation_fields: 0
missing_action_names: []
class_weighting: balanced
train_accuracy: 0.602
validation_accuracy: 0.533
```

Validation per-action accuracy:

```text
STAY_COURSE: 0.598
EXPAND: 1.000
ADD_GATEWAYS: 0.316
TECH_ROBO: 0.259
FORGE_UPGRADES: 0.667
BUILD_STATIC_DEFENSE: 1.000
PRODUCE_ARMY: 0.353
BOOST_WORKERS: 0.615
```

Interpretation:

```text
The strategy checkpoint is structurally valid and all 8 actions are present.
Validation accuracy is modest, with TECH_ROBO / ADD_GATEWAYS / PRODUCE_ARMY still confused.
This is a runnable candidate, not a stable baseline.
```

Hidden-window guard before SC2 eval:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Result:

```text
21392
```

Guarded eval/data collection command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Easy Medium Hard --opponents Protoss Terran Zerg --games-per-combo 1 --trajectory-dir data\trajectories\strategy_imitation_v2_candidate_eval_v1_army --strategy-trajectory-dir data\trajectories\strategy_imitation_v2_candidate_eval_v1 --run-root runs --run-name eval_strategy_imitation_v2_candidate_v1 --tag strategy-v2 --tag strategy-imitation --tag guarded-eval --policy-name strategy_imitation_v2_candidate --strategy-policy checkpoint --strategy-checkpoint runs\20260623_113345_strategy_imitation_v2_candidate\checkpoints\policy.pt --strategy-device cpu --record-decision-interval 16 --game-time-limit 900
```

Eval/data artifacts:

```text
run: runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1
eval: runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\eval.jsonl
summary: runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\summary.json
strategy trajectories: data\trajectories\strategy_imitation_v2_candidate_eval_v1
strategy diagnostics: runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\strategy_trajectory_diagnostics.json
army trajectories: data\trajectories\strategy_imitation_v2_candidate_eval_v1_army
army diagnostics: runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\army_trajectory_diagnostics.json
```

Eval result:

```text
games: 9
return_code=0 for all 9 games
Victory: 5
Defeat: 3
Tie: 1
Easy: 3 Victory / 0 Defeat / 0 Tie
Medium: 2 Victory / 0 Defeat / 1 Tie
Hard: 0 Victory / 3 Defeat / 0 Tie
```

Online strategy diagnostics:

```text
files: 9
rows: 496
training_rows: 487
terminal_rows: 9
observation_dim: 40
observation_schemas: strategy_v2=496
rows_defaulted_observation_fields: 0
action_coverage: 100%
missing_actions: none
low_count_actions: none
warnings: none
```

Online strategy action distribution:

```text
STAY_COURSE: 148
EXPAND: 26
ADD_GATEWAYS: 71
TECH_ROBO: 22
FORGE_UPGRADES: 49
BUILD_STATIC_DEFENSE: 63
PRODUCE_ARMY: 35
BOOST_WORKERS: 73
```

Online army diagnostics from the same eval:

```text
army schema: v3=1943
rows_defaulted_observation_fields: 0
RALLY: 1344
ATTACK_MAIN: 590
RETREAT_HOME / DEFEND_BASE / HOLD: not covered
```

Interpretation:

```text
strategy_imitation_v2_candidate is runnable and did not action-collapse online.
Online strategy action coverage is healthy in this 9-game sample.
Performance is not yet stable: Hard Protoss/Terran/Zerg were all Defeat.
The default rule baseline remains behavior-preserving because strategy checkpoint use is explicit opt-in.
Do not enter PPO from this evidence. Continue comparing strategy candidate vs rule/coverage-teacher and inspect Hard-game failure timing.
```

Final validation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results:

```text
pytest final: 114 passed in 2.95s
check_env final: all OK
PPO touched: no
```

## 2026-06-23 Strategy Observation + Diagnostics

User request:

```text
按计划继续开发
```

Scope:

```text
Continue STRATEGY_EXPANSION_PLAN.md after phase 1/2.
No PPO.
No SC2 launch.
Do not change ArmyAction v1 or army schema v3 defaults.
Do not change PPO/RL mainline architecture.
```

Implemented files:

```text
rl\strategy_observations.py
tests\test_strategy_observations.py
```

Modified files:

```text
rl\diagnostics.py
scripts\diagnose_trajectories.py
tests\test_trajectory_diagnostics.py
CODEX.md
README.md
STATE.md
```

Strategy observation schema:

```text
schema: strategy_v1
dimension: 33
```

Fields:

```text
game_time
minerals
vespene
supply_used
supply_cap
supply_left
workers
own_bases
ready_gateways
ready_robo
ready_forge
ready_static_defense
has_cybernetics_core
zealots
stalkers
immortals
observers
sentries
army_count
ground_weapon_level
ground_armor_level
enemy_units_known
enemy_structures_known
enemy_air_units_known
enemy_armored_units_known
enemy_cloaked_units_seen
worker_saturation_ratio
gateway_idle_count
robo_idle_count
base_under_air_threat
base_under_ground_threat
base_under_threat
enemy_to_home_distance
```

Diagnostics changes:

```text
diagnose_trajectories now supports trajectory_kind="army" (default) and "strategy".
scripts\diagnose_trajectories.py now supports --kind army|strategy.
Default army diagnostics still read action/observation and use ArmyAction names.
Strategy diagnostics read strategy_action and strategy_observation, falling back to observation for strategy-only rows.
Strategy diagnostics report strategy_v1 schema counts and StrategyAction coverage.
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_observations.py tests\test_trajectory_diagnostics.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Results:

```text
strategy/diagnostics focused tests: 10 passed
full suite: 79 passed in 2.06s
```

Conclusion:

```text
Phase 3 is complete for a first strategy_v1 schema and diagnostics support.
The army schema v3 default path remains unchanged.
No trajectory recording changes were made yet, so real strategy datasets are not being emitted by the bot yet.
PPO remains not recommended.
```

Recommended next work:

```text
1. Extend trajectory recording so strategy decisions are recorded separately from army decisions.
2. Keep old army-only trajectory loader compatible.
3. Add tests for mixed/strategy trajectory rows and terminal rows.
4. Then implement StrategyCoverageTeacher and diagnose strategy action coverage before any strategy imitation training.
```

## 2026-06-23 Strategy Trajectory Recording

User request:

```text
继续开发
```

Scope:

```text
Continue STRATEGY_EXPANSION_PLAN.md phase 4.
No PPO.
No SC2 launch.
Keep default army-only trajectory path unchanged.
Keep old army imitation loader compatible.
Do not change ArmyAction v1 or army schema v3 defaults.
```

Implemented / modified files:

```text
rl\trajectory_recorder.py
bot\protoss_rule_bot.py
run.py
scripts\evaluate.py
tests\test_rl_trajectory_modules.py
tests\test_strategy_trajectory_recording.py
tests\test_evaluate.py
CODEX.md
README.md
STATE.md
```

Trajectory format:

```text
Existing army rows are unchanged:
  observation
  action
  action_name

New strategy rows use StrategyTrajectoryStep:
  strategy_observation
  strategy_action
  strategy_action_name
  army_observation
  army_action
  army_action_name
  reward
  done
  result
```

Runtime integration:

```text
ProtossRuleBot accepts strategy_trajectory_recorder=None by default.
_manage_strategy records one strategy row only when a strategy recorder is present.
on_end writes a terminal strategy row only when a strategy recorder is present.
Default --trajectory-path remains army-only.
run.py adds --strategy-trajectory-path for explicit strategy JSONL capture.
scripts\evaluate.py adds --strategy-trajectory-dir and forwards per-game --strategy-trajectory-path.
EvalRecord now includes strategy_trajectory_path.
```

Compatibility:

```text
Old army-only trajectory loader still reads action/observation rows.
Strategy rows are written to a separate file/path, so army imitation datasets are not polluted.
Strategy diagnostics already support --kind strategy from the previous phase.
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rl_trajectory_modules.py tests\test_strategy_trajectory_recording.py tests\test_evaluate.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Results:

```text
strategy trajectory / eval focused tests: 18 passed
full suite: 83 passed in 2.33s
```

Conclusion:

```text
Phase 4 is complete for explicit, separate strategy trajectory recording.
No real strategy datasets were collected in this step because SC2 was not launched.
Default rule baseline and army imitation recording remain unchanged.
PPO remains not recommended.
```

Recommended next work:

```text
1. Implement StrategyCoverageTeacher as an explicit opt-in strategy policy.
2. Add run.py / evaluate.py strategy-policy selection only for rule / coverage strategy once the teacher exists.
3. Collect a small guarded strategy dataset with --strategy-trajectory-dir.
4. Diagnose it with scripts\diagnose_trajectories.py <dir> --kind strategy --show-files.
5. Only train strategy imitation after strategy action coverage is acceptable.
```

## 2026-06-23 StrategyCoverageTeacher

User request:

```text
可以，继续开发
```

Scope:

```text
Continue STRATEGY_EXPANSION_PLAN.md phase 5.
No PPO.
No SC2 launch.
Keep default strategy policy as rule no-op.
Keep army policy / army schema v3 / army trajectory defaults unchanged.
```

Implemented / modified files:

```text
bot\managers\coverage_strategy_policy.py
bot\managers\__init__.py
run.py
scripts\evaluate.py
tests\test_coverage_strategy_policy.py
tests\test_evaluate.py
CODEX.md
README.md
STATE.md
```

Strategy teacher:

```text
CoverageStrategyPolicy
  decide_strategy(bot) -> StrategyAction
  decide_from_observation(strategy_v1 dict) -> StrategyAction
```

Rule labels:

```text
BUILD_STATIC_DEFENSE:
  base_under_threat and static defense below per-base target

BOOST_WORKERS:
  worker_saturation_ratio below floor with minerals and supply available

EXPAND:
  below target bases, saturated enough, safe, and enough minerals

ADD_GATEWAYS:
  ready_gateways below bases * gateways_per_base and enough minerals

TECH_ROBO:
  Cyber ready, no Robo, enough gas, and armored/cloaked/midgame/army signal

FORGE_UPGRADES:
  no Forge, enough minerals, and midgame

PRODUCE_ARMY:
  idle Gateway/Robo or low army count with resources and supply

STAY_COURSE:
  fallback when no coverage rule fires
```

CLI integration:

```text
run.py:
  --strategy-policy rule
  --strategy-policy coverage-teacher

scripts\evaluate.py:
  --strategy-policy rule
  --strategy-policy coverage-teacher
```

Default:

```text
--strategy-policy rule
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_coverage_strategy_policy.py tests\test_evaluate.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Results:

```text
coverage strategy / evaluate focused tests: 16 passed
full suite: 92 passed in 2.30s
```

Conclusion:

```text
Phase 5 has an initial opt-in StrategyCoverageTeacher.
It is ready for small guarded data collection, but no SC2 collection was run in this step.
Default runtime behavior remains behavior-preserving because --strategy-policy defaults to rule.
PPO remains not recommended.
```

Recommended next work:

```text
1. Run a small guarded strategy coverage-teacher eval with --strategy-policy coverage-teacher and --strategy-trajectory-dir.
2. Diagnose the new strategy trajectories with --kind strategy.
3. Check strategy_v1 schema counts, rows_defaulted_observation_fields, action coverage, and per-action counts.
4. Tune CoverageStrategyPolicy only if action coverage is poor or labels are not supported by observation_feature_stats.
5. Do not train strategy imitation until coverage is acceptable.
```

## 2026-06-23 Strategy Framework Review

User request:

```text
先审查一遍策略框架。无误后进入测试
```

Review scope:

```text
StrategyAction / StrategyPolicy / RuleStrategyPolicy
StrategyExecutor
strategy_v1 observation
StrategyTrajectoryStep and strategy diagnostics
StrategyCoverageTeacher
run.py / scripts\evaluate.py opt-in strategy policy and trajectory flags
```

Findings fixed:

```text
1. StrategyExecutor TECH_ROBO checked Observer count through structures instead of units.
   Fix: add _unit_count() and use units.of_type({OBSERVER}) before training another Observer.

2. StrategyExecutor FORGE_UPGRADES skipped pending upgrades but not completed upgrades.
   Fix: add _upgrade_researched() against bot.state.upgrades and skip completed upgrades.

3. CoverageStrategyPolicy did not emit FORGE_UPGRADES when Forge already existed but level-1 ground upgrades were incomplete.
   Fix: keep emitting FORGE_UPGRADES when ground_weapon_level < 1 or ground_armor_level < 1.
```

Tests added:

```text
test_strategy_executor_tech_robo_trains_immortal_when_observer_exists
test_strategy_executor_forge_upgrades_skips_completed_upgrades
test_coverage_strategy_continues_forge_upgrades_when_forge_exists
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_policy.py tests\test_coverage_strategy_policy.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
git diff --check
```

Results:

```text
strategy focused tests: 21 passed
full suite: 95 passed in 2.27s
check_env: all checks passed
git diff --check: no whitespace errors; only LF/CRLF warnings
```

Conclusion:

```text
No remaining blocking strategy-framework findings from this review.
No SC2 launch was performed.
PPO remains not recommended.
Next step is guarded strategy coverage-teacher data collection and diagnostics.
```

## 2026-06-23 Strategy Coverage Data Test

User request:

```text
进行数据采集测试
```

Scope:

```text
Run a larger guarded strategy coverage-teacher data collection test after the strategy framework review.
No PPO.
No code changes.
Keep default strategy policy behavior-preserving; coverage strategy remains explicit opt-in.
Use scripts\evaluate.py and hidden-window guard before SC2 launch.
```

Health checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results:

```text
pytest: 95 passed in 2.30s
check_env: all checks passed
```

Hidden-window guard:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Result:

```text
21392
```

Collection command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Easy Medium Hard --opponents Protoss Terran Zerg --games-per-combo 2 --run-root runs --run-name strategy_coverage_teacher_v1_larger --tag strategy-v1 --tag coverage-strategy --tag data-test --policy-name strategy_coverage_teacher_v1_larger --strategy-policy coverage-teacher --strategy-trajectory-dir data\trajectories\strategy_coverage_teacher_v1_larger --record-decision-interval 16 --game-time-limit 900
```

Generated paths:

```text
run: runs\20260623_092727_strategy_coverage_teacher_v1_larger
eval: runs\20260623_092727_strategy_coverage_teacher_v1_larger\artifacts\eval.jsonl
strategy trajectories: data\trajectories\strategy_coverage_teacher_v1_larger
diagnostics json: runs\20260623_092727_strategy_coverage_teacher_v1_larger\artifacts\strategy_trajectory_diagnostics.json
```

Eval result:

```text
games: 18
return_codes: 0 for all 18 games
Victory: 15
Defeat: 3
Tie: 0
```

Per-combo result:

```text
Easy Protoss: 2 Victory
Easy Terran: 2 Victory
Easy Zerg: 2 Victory
Medium Protoss: 2 Victory
Medium Terran: 2 Victory
Medium Zerg: 2 Victory
Hard Protoss: 2 Defeat
Hard Terran: 1 Victory / 1 Defeat
Hard Zerg: 2 Victory
```

Diagnostics command:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py data\trajectories\strategy_coverage_teacher_v1_larger --kind strategy --show-files --json-output runs\20260623_092727_strategy_coverage_teacher_v1_larger\artifacts\strategy_trajectory_diagnostics.json
```

Diagnostics summary:

```text
files: 18
rows: 893
training_rows: 875
terminal_rows: 18
empty_files: 0
files_missing_terminal: 0
observation_dim: 33
observation_schemas: strategy_v1=893
rows_defaulted_observation_fields: 0
action_coverage: 100.0%
missing_actions: none
low_count_actions (<10): FORGE_UPGRADES
results: Victory=15, Defeat=3
```

Action distribution:

```text
STAY_COURSE: 382
EXPAND: 14
ADD_GATEWAYS: 103
TECH_ROBO: 216
FORGE_UPGRADES: 5
BUILD_STATIC_DEFENSE: 60
PRODUCE_ARMY: 40
BOOST_WORKERS: 55
```

Selected observation feature stats:

```text
own_bases: min=0 max=2 avg=1.235
ready_gateways: min=0 max=8 avg=3.253
ready_robo: min=0 max=1 avg=0.168
ready_forge: min=0 max=1 avg=0.031
ready_static_defense: min=0 max=1 avg=0.087
army_count: min=0 max=42 avg=8.809
immortals: min=0 max=0 avg=0
observers: min=0 max=0 avg=0
sentries: min=0 max=0 avg=0
ground_weapon_level: min=0 max=0 avg=0
ground_armor_level: min=0 max=0 avg=0
enemy_air_units_known: min=0 max=8 avg=0.406
enemy_armored_units_known: min=0 max=8 avg=0.410
enemy_cloaked_units_seen: min=0 max=2 avg=0.110
worker_saturation_ratio: min=0 max=1.455 avg=0.880
gateway_idle_count: min=0 max=6 avg=0.857
robo_idle_count: min=0 max=1 avg=0.168
base_under_threat: min=0 max=1 avg=0.067
enemy_to_home_distance: min=0 max=160.756 avg=30.560
```

Conclusion:

```text
The data collection pipeline is healthy: all SC2 processes returned 0, all rows use strategy_v1, no observation fields defaulted, and all 8 strategy actions were covered.

This is still not a good strategy imitation training set. FORGE_UPGRADES has only 5 rows, and the feature stats support the explanation: Forge almost never becomes ready, ground upgrades never appear, and TECH_ROBO / STAY_COURSE dominate the midgame labels.

Do not train strategy imitation from this dataset yet.
Recommended next step is to tune CoverageStrategyPolicy or collection parameters so Forge/upgrades labels appear more reliably, then recollect another guarded strategy dataset. PPO remains not recommended.
```

## 2026-06-23 StrategyCoverageTeacher Forge Coverage Tune

User request:

```text
可以，继续开发
```

Scope:

```text
Continue after strategy coverage data showed FORGE_UPGRADES remained low-count.
No PPO.
No PPO/RL mainline architecture changes.
Keep default RuleStrategyPolicy behavior-preserving.
Only tune explicit opt-in CoverageStrategyPolicy and tests.
Use hidden-window guard before SC2 collection.
```

Evidence from previous dataset:

```text
data: data\trajectories\strategy_coverage_teacher_v1_larger
training rows: 875
old action distribution:
  STAY_COURSE=382
  EXPAND=14
  ADD_GATEWAYS=103
  TECH_ROBO=216
  FORGE_UPGRADES=5
  BUILD_STATIC_DEFENSE=60
  PRODUCE_ARMY=40
  BOOST_WORKERS=55

Forge-eligible rows under old observations: 61
Forge-eligible labels under old teacher:
  ADD_GATEWAYS=49
  TECH_ROBO=6
  FORGE_UPGRADES=5
  BUILD_STATIC_DEFENSE=1
```

Interpretation:

```text
FORGE_UPGRADES was not low because Forge states never appeared at all.
Most potentially valid Forge/upgrades labels were being preempted by ADD_GATEWAYS because the teacher wanted bases * 4 ready Gateways before considering Forge.
Some were also preempted by TECH_ROBO.
```

Code changes:

```text
bot\managers\coverage_strategy_policy.py
  Added forge_min_gateways=4.
  Added urgent_robo_signal / robo_needed factoring.
  Let urgent cloaked-unit Robo labels fire before Forge/Gateway expansion.
  Let midgame Forge/upgrades fire before filling all extra Gateways once at least 4 Gateways are ready.
  Kept defense, worker boost, and expansion above Forge.

tests\test_coverage_strategy_policy.py
  Added test_coverage_strategy_prioritizes_midgame_forge_before_extra_gateways.
  Added test_coverage_strategy_techs_robo_for_cloaked_signal_before_forge.
```

TDD check:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_coverage_strategy_policy.py -q
```

Before code change:

```text
2 failed, 9 passed
```

After code change:

```text
11 passed
```

Offline replay on previous observations with new teacher:

```text
rows: 875
old:
  STAY_COURSE=382
  BOOST_WORKERS=55
  ADD_GATEWAYS=103
  PRODUCE_ARMY=40
  TECH_ROBO=216
  BUILD_STATIC_DEFENSE=60
  EXPAND=14
  FORGE_UPGRADES=5

new projected:
  STAY_COURSE=382
  BOOST_WORKERS=55
  ADD_GATEWAYS=57
  PRODUCE_ARMY=40
  TECH_ROBO=223
  BUILD_STATIC_DEFENSE=60
  EXPAND=14
  FORGE_UPGRADES=44

changed:
  ADD_GATEWAYS -> FORGE_UPGRADES: 35
  ADD_GATEWAYS -> TECH_ROBO: 11
  TECH_ROBO -> FORGE_UPGRADES: 4
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results:

```text
pytest: 97 passed in 2.36s
check_env: all checks passed
```

Hidden-window guard:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Result:

```text
21392
```

Collection command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Easy Medium Hard --opponents Protoss Terran Zerg --games-per-combo 2 --run-root runs --run-name strategy_coverage_teacher_v1_forge_tuned_v1 --tag strategy-v1 --tag coverage-strategy --tag forge-tuned --policy-name strategy_coverage_teacher_v1_forge_tuned_v1 --strategy-policy coverage-teacher --strategy-trajectory-dir data\trajectories\strategy_coverage_teacher_v1_forge_tuned_v1 --record-decision-interval 16 --game-time-limit 900
```

Generated paths:

```text
run: runs\20260623_094943_strategy_coverage_teacher_v1_forge_tuned_v1
eval: runs\20260623_094943_strategy_coverage_teacher_v1_forge_tuned_v1\artifacts\eval.jsonl
strategy trajectories: data\trajectories\strategy_coverage_teacher_v1_forge_tuned_v1
diagnostics json: runs\20260623_094943_strategy_coverage_teacher_v1_forge_tuned_v1\artifacts\strategy_trajectory_diagnostics.json
```

Eval result:

```text
games: 18
return_codes: 0 for all 18 games
Victory: 13
Defeat: 2
Tie: 3
```

Per-combo result:

```text
Easy Protoss: 2 Victory
Easy Terran: 2 Tie
Easy Zerg: 2 Victory
Medium Protoss: 1 Victory / 1 Tie
Medium Terran: 2 Victory
Medium Zerg: 2 Victory
Hard Protoss: 1 Victory / 1 Defeat
Hard Terran: 1 Victory / 1 Defeat
Hard Zerg: 2 Victory
```

Diagnostics command:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py data\trajectories\strategy_coverage_teacher_v1_forge_tuned_v1 --kind strategy --show-files --json-output runs\20260623_094943_strategy_coverage_teacher_v1_forge_tuned_v1\artifacts\strategy_trajectory_diagnostics.json
```

Diagnostics summary:

```text
files: 18
rows: 958
training_rows: 940
terminal_rows: 18
empty_files: 0
files_missing_terminal: 0
observation_dim: 33
observation_schemas: strategy_v1=958
rows_defaulted_observation_fields: 0
action_coverage: 100.0%
missing_actions: none
low_count_actions (<10): none
warnings: none
results: Victory=13, Tie=3, Defeat=2
```

Action distribution:

```text
STAY_COURSE: 411
EXPAND: 16
ADD_GATEWAYS: 69
TECH_ROBO: 178
FORGE_UPGRADES: 114
BUILD_STATIC_DEFENSE: 50
PRODUCE_ARMY: 44
BOOST_WORKERS: 58
```

Selected observation feature stats:

```text
own_bases: min=0 max=2 avg=1.258
ready_gateways: min=0 max=7 avg=2.929
ready_robo: min=0 max=1 avg=0.228
ready_forge: min=0 max=1 avg=0.169
ready_static_defense: min=0 max=4 avg=0.296
army_count: min=0 max=76 avg=12.675
ground_weapon_level: min=0 max=1 avg=0.048
ground_armor_level: min=0 max=1 avg=0.023
enemy_air_units_known: min=0 max=7 avg=0.344
enemy_armored_units_known: min=0 max=10 avg=0.344
enemy_cloaked_units_seen: min=0 max=3 avg=0.104
worker_saturation_ratio: min=0.364 max=1.727 avg=0.901
gateway_idle_count: min=0 max=7 avg=0.718
robo_idle_count: min=0 max=1 avg=0.228
base_under_threat: min=0 max=1 avg=0.059
enemy_to_home_distance: min=0 max=156.424 avg=30.831
```

Conclusion:

```text
Forge/upgrades coverage issue is fixed for the current teacher/data loop.
The latest strategy dataset is structurally healthy and all 8 StrategyAction labels have non-low-count coverage.
Do not use the existing army imitation path for strategy checkpoints because it encodes army schema/action metadata.
Next step is Phase 6: implement explicit strategy imitation dataset/training/checkpoint support for strategy_v1 + StrategyAction, then train a strategy imitation candidate from data\trajectories\strategy_coverage_teacher_v1_forge_tuned_v1.
PPO remains not recommended.
```

## 2026-06-23 Strategy v2 Pending Features + Data Collection

User request:

```text
可以，继续优化策略并收集数据。
```

Scope:

```text
Continue strategy optimization and collect data.
No PPO.
No PPO/RL mainline architecture changes.
Keep default RuleStrategyPolicy behavior-preserving.
Improve explicit opt-in CoverageStrategyPolicy label quality.
Use hidden-window guard before each SC2 collection.
```

Problem found in previous `strategy_coverage_teacher_v1_forge_tuned_v1`:

```text
FORGE_UPGRADES coverage was healthy, but several games had long consecutive Forge labels.
Top Forge runs:
  23 consecutive FORGE_UPGRADES in an Easy Terran Tie
  22 consecutive FORGE_UPGRADES in a Medium Protoss Tie

Root cause:
  strategy_v1 could see completed ground upgrade levels, but could not see pending buildings or pending upgrades.
  The teacher therefore kept emitting FORGE_UPGRADES while Forge construction or level-1 research was already pending.
```

Code changes retained:

```text
rl\strategy_observations.py
  Bumped current strategy schema from strategy_v1 to strategy_v2.
  Added pending macro features:
    pending_bases
    pending_gateways
    pending_robo
    pending_forge
    pending_static_defense
    ground_weapon_upgrade_pending
    ground_armor_upgrade_pending
  Kept STRATEGY_OBSERVATION_FIELDS_V1 and defaults so old strategy_v1 rows remain diagnosable.

rl\diagnostics.py
  Added pending strategy fields to strategy observation feature stats.

bot\managers\coverage_strategy_policy.py
  Uses ready + pending counts for expand, gateways, robo, forge, and static defense labels.
  Suppresses duplicate FORGE_UPGRADES when Forge or ground-upgrade research is already pending.
  Suppresses duplicate TECH_ROBO when Robo is already pending.

tests\test_strategy_observations.py
tests\test_coverage_strategy_policy.py
tests\test_trajectory_diagnostics.py
  Added v2 field/defaulting and pending-label behavior coverage.
```

Validation after retained changes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_observations.py tests\test_coverage_strategy_policy.py tests\test_trajectory_diagnostics.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Results:

```text
focused: 26 passed
full suite: 102 passed in 3.05s
```

Hidden-window guard before v2 collection:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Result:

```text
21392
```

Collection command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Easy Medium Hard --opponents Protoss Terran Zerg --games-per-combo 2 --run-root runs --run-name strategy_coverage_teacher_v2_pending_tuned_v1 --tag strategy-v2 --tag coverage-strategy --tag pending-tuned --policy-name strategy_coverage_teacher_v2_pending_tuned_v1 --strategy-policy coverage-teacher --strategy-trajectory-dir data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 --record-decision-interval 16 --game-time-limit 900
```

Generated paths:

```text
run: runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1
eval: runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\eval.jsonl
strategy trajectories: data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1
diagnostics json: runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\strategy_trajectory_diagnostics.json
```

Eval result:

```text
games: 18
return_codes: 0 for all 18 games
Victory: 13
Defeat: 3
Tie: 2
```

Diagnostics summary:

```text
files: 18
rows: 927
training_rows: 909
terminal_rows: 18
empty_files: 0
files_missing_terminal: 0
observation_dim: 40
observation_schemas: strategy_v2=927
rows_defaulted_observation_fields: 0
action_coverage: 100.0%
missing_actions: none
low_count_actions (<10): none
warnings: none
```

Action distribution:

```text
STAY_COURSE: 490
EXPAND: 11
ADD_GATEWAYS: 69
TECH_ROBO: 130
FORGE_UPGRADES: 24
BUILD_STATIC_DEFENSE: 50
PRODUCE_ARMY: 69
BOOST_WORKERS: 66
```

Selected v2 feature stats:

```text
pending_bases: min=0 max=1 avg=0.073
pending_gateways: min=0 max=3 avg=0.454
pending_robo: min=0 max=1 avg=0.063
pending_forge: min=0 max=1 avg=0.046
pending_static_defense: min=0 max=2 avg=0.030
ground_weapon_upgrade_pending: min=0 max=1 avg=0.128
ground_armor_upgrade_pending: min=0 max=0 avg=0
```

Label-quality check:

```text
All FORGE_UPGRADES runs in strategy_coverage_teacher_v2_pending_tuned_v1 have length 1.
The previous 22/23-step Forge label spam is fixed.
```

Negative experiment:

```text
An additional attempted optimization added a minerals threshold for TECH_ROBO.
It was tested and then reverted because online data got worse.
```

Rejected code change:

```text
CoverageStrategyPolicy.robo_minerals = 150.0
TECH_ROBO required minerals >= 150 and vespene >= 100
```

Rejected test:

```text
test_coverage_strategy_waits_for_robo_minerals_before_tech_robo
```

Negative collection command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Easy Medium Hard --opponents Protoss Terran Zerg --games-per-combo 2 --run-root runs --run-name strategy_coverage_teacher_v2_pending_robo_tuned_v1 --tag strategy-v2 --tag coverage-strategy --tag pending-robo-tuned --policy-name strategy_coverage_teacher_v2_pending_robo_tuned_v1 --strategy-policy coverage-teacher --strategy-trajectory-dir data\trajectories\strategy_coverage_teacher_v2_pending_robo_tuned_v1 --record-decision-interval 16 --game-time-limit 900
```

Negative experiment paths:

```text
run: runs\20260623_103656_strategy_coverage_teacher_v2_pending_robo_tuned_v1
eval: runs\20260623_103656_strategy_coverage_teacher_v2_pending_robo_tuned_v1\artifacts\eval.jsonl
strategy trajectories: data\trajectories\strategy_coverage_teacher_v2_pending_robo_tuned_v1
diagnostics json: runs\20260623_103656_strategy_coverage_teacher_v2_pending_robo_tuned_v1\artifacts\strategy_trajectory_diagnostics.json
```

Negative experiment result:

```text
games: 18
return_codes: 0 for all 18 games
Victory: 11
Defeat: 2
Tie: 5
schema: strategy_v2 only
rows_defaulted_observation_fields: 0
action_coverage: 100%
STAY_COURSE: 628
EXPAND: 13
ADD_GATEWAYS: 101
TECH_ROBO: 15
FORGE_UPGRADES: 23
BUILD_STATIC_DEFENSE: 65
PRODUCE_ARMY: 96
BOOST_WORKERS: 56
```

Rejected interpretation:

```text
The Robo minerals gate over-suppressed TECH_ROBO and inflated STAY_COURSE.
It should not be retained as current teacher behavior.
The generated dataset is useful as a negative experiment, but should not be the recommended training dataset.
```

Final retained status:

```text
Current strategy schema: strategy_v2, 40 features.
Recommended strategy dataset: data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1.
Current tests: 102 passed.
Do not train strategy imitation through the existing army imitation path.
Next step is explicit strategy imitation dataset/training/checkpoint support for strategy_v2 + StrategyAction.
PPO remains not recommended.
```

## 2026-06-23 Conservative GG / Surrender Policy

User request:

```text
加一个 "gg" 功能，让 ai 在识别到本局游戏几乎没有可能获胜时直接投降、退出游戏。
```

Scope:

```text
Add a conservative runtime surrender feature.
No PPO.
No PPO/RL mainline architecture changes.
Do not change checkpoint formats or strategy/army policy interfaces.
Do not launch SC2 for this unit-level change.
```

Code changes:

```text
bot\managers\surrender_policy.py
  Added SurrenderPolicy and maybe_surrender().
  Default policy only fires after 360 seconds when:
    ready Nexus count is 0
    pending Nexus count is 0
    workers <= 4
    army <= 2
    ready + pending production structures <= 1
    and there is enemy pressure evidence, scouted enemy structures, or no workers remain
  Also fires with one remaining ready Nexus only when workers are 0, army <= 2,
    production <= 1, no Nexus is pending, and enemy pressure is visible.
  If workers can still rebuild a Nexus with >=400 minerals, the policy does not surrender.
  maybe_surrender() sends "gg" once, then awaits bot.client.leave().
  A one-shot _gg_surrendered guard prevents repeated chat/leave calls.

bot\protoss_rule_bot.py
  Instantiates SurrenderPolicy with GG_* class thresholds.
  Calls maybe_surrender(self) at the top of on_step() and returns early if it fires.

bot\managers\__init__.py
  Exports SurrenderPolicy.

tests\test_surrender_policy.py
  Added coverage for minimum game time, recoverable states, pending Nexus recovery,
  one-base terminal pressure, hopeless GG+leave behavior, and one-shot behavior.

CODEX.md / README.md
  Documented the conservative GG feature.
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_surrender_policy.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results:

```text
focused surrender tests: 6 passed
full test suite: 108 passed in 3.91s
environment check: all OK
SC2 launched: no
PPO touched: no
```

Current status:

```text
The bot now has conservative GG/surrender support in normal on_step runtime.
This is rule-side safety behavior and does not affect army/strategy action spaces,
trajectory schemas, imitation checkpoints, or PPO/RL architecture.
PPO remains not recommended.
```

## 2026-06-23 Next Strategy Development Plan

User request:

```text
可以，把接下来的规划更新到文档里。写一段下一步的提示词出来
```

Current planning state:

```text
strategy_imitation_v2_candidate is runnable and did not action-collapse online.
It is not yet a stable strategy baseline:
  validation_accuracy=0.533
  first guarded eval=5 Victory / 3 Defeat / 1 Tie
  Hard Protoss / Hard Terran / Hard Zerg all Defeat
The next step should explain Hard failures before more training or larger collection.
PPO remains not recommended.
```

Documentation updates:

```text
CODEX.md
  Expanded Recommended Next Work.

README.md
  Added a user-facing 下一步计划 section.

STATE.md
  Appended this planning note so tail readers see the current direction.
```

Next development plan:

```text
1. Keep default rule baseline behavior-preserving.
2. Do not enter PPO.
3. First tidy STATE.md ordering if needed so newest strategy imitation state is at the tail.
4. Add strategy timing diagnostics before more blind training:
   - first/average/min/max game_time per StrategyAction
   - per-file action timeline summary
   - action distribution when base_under_threat=1
   - latency from enemy_armored_units_known or enemy_cloaked_units_seen signal to first TECH_ROBO
   - repeated macro actions while corresponding pending_* features are active
5. Run timing diagnostics on:
   - data\trajectories\strategy_imitation_v2_candidate_eval_v1
   - data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1
6. Focus analysis on the three Hard defeats from:
   runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1
7. Modify CoverageStrategyPolicy only if diagnostics show specific teacher timing defects.
8. If modified, add focused tests and run full pytest.
9. Only after diagnostics/teacher fixes, plan Hard-focused strategy-v2 data collection and train strategy_imitation_v2_candidate_v2.
```

## 2026-06-23 Strategy Timing Diagnostics for Hard Failures

Scope:

```text
Diagnose why strategy_imitation_v2_candidate lost all Hard games in the first
guarded eval before changing CoverageStrategyPolicy or collecting new data.
No PPO.
No SC2 launch in this diagnostics/code-only pass.
No rule baseline behavior change.
```

Code changes:

```text
rl\strategy_timing_diagnostics.py
  Added timing-oriented diagnostics for strategy trajectories:
    action first/min/max/avg game_time
    per-file consecutive action timeline
    threat-state action distribution
    TECH_ROBO timing against armored/cloaked signals
    pending_* repeat counts
    Hard defeat file summaries
  Fixed Hard defeat detection to accept both "Defeat" and "Result.Defeat".
  Split TECH_ROBO signal timing into:
    files_with_tech_after_signal
    files_with_tech_before_signal
    files_without_tech

scripts\diagnose_strategy_timing.py
  Added CLI and JSON artifact output for strategy timing diagnostics.

tests\test_strategy_timing_diagnostics.py
  Added focused tests for timing summaries, timelines, missing tech,
  tech-before-signal classification, bare "Defeat" results, and report sections.
```

Commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_timing_diagnostics.py -q
.\.venv\Scripts\python.exe scripts\diagnose_strategy_timing.py data\trajectories\strategy_imitation_v2_candidate_eval_v1 --show-files --json-output runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\strategy_timing_diagnostics.json
.\.venv\Scripts\python.exe scripts\diagnose_strategy_timing.py data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1 --show-files --json-output runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\strategy_timing_diagnostics.json
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_timing_diagnostics.py tests\test_trajectory_diagnostics.py tests\test_strategy_datasets.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Validation:

```text
Initial health check from this pass: 114 passed; check_env all OK.
Focused timing tests after implementation: 5 passed.
Related diagnostics/dataset tests: 16 passed.
Final full suite: 119 passed in 3.23s.
Final check_env: all OK.
SC2 launched: no.
Hidden-window guard needed: no, because no SC2 command was run.
PPO touched: no.
CoverageStrategyPolicy changed: no.
```

Artifacts:

```text
imitation timing diagnostics:
runs\20260623_113438_eval_strategy_imitation_v2_candidate_v1\artifacts\strategy_timing_diagnostics.json

coverage-teacher timing diagnostics:
runs\20260623_101638_strategy_coverage_teacher_v2_pending_tuned_v1\artifacts\strategy_timing_diagnostics.json
```

strategy_imitation_v2_candidate timing summary:

```text
files: 9
rows: 496
training_rows: 487
results: 5 Victory / 3 Defeat / 1 Tie
Hard defeat files:
  data\trajectories\strategy_imitation_v2_candidate_eval_v1\20260623_114025_AcropolisLE_Hard_Protoss_001.jsonl
  data\trajectories\strategy_imitation_v2_candidate_eval_v1\20260623_114148_AcropolisLE_Hard_Terran_001.jsonl
  data\trajectories\strategy_imitation_v2_candidate_eval_v1\20260623_114240_AcropolisLE_Hard_Zerg_001.jsonl

action_timing:
  STAY_COURSE: count=148 first=0.0 avg=154.9 min=0.0 max=800.0
  EXPAND: count=26 first=217.1 avg=254.9 min=217.1 max=297.1
  ADD_GATEWAYS: count=71 first=411.4 avg=582.7 min=411.4 max=880.0
  TECH_ROBO: count=22 first=354.3 avg=383.9 min=308.6 max=468.6
  FORGE_UPGRADES: count=49 first=365.7 avg=438.5 min=331.4 max=891.4
  BUILD_STATIC_DEFENSE: count=63 first=388.6 avg=538.8 min=194.3 max=777.1
  PRODUCE_ARMY: count=35 first=205.7 avg=270.4 min=171.4 max=514.3
  BOOST_WORKERS: count=73 first=57.1 avg=203.8 min=57.1 max=834.3

threat_action_counts:
  ADD_GATEWAYS: 1
  BUILD_STATIC_DEFENSE: 62

pending_repeat_counts:
  EXPAND: 8
  ADD_GATEWAYS: 38
  TECH_ROBO: 1
  FORGE_UPGRADES: 3
  BUILD_STATIC_DEFENSE: 6

TECH_ROBO signal timing:
  armored_signal: files_with_signal=6, tech_after=1, tech_before=5, no_tech=0, avg_delay=80.0, avg_early_lead=22.9
  cloaked_signal: files_with_signal=2, tech_after=0, tech_before=2, no_tech=0, avg_early_lead=114.3
```

coverage-teacher timing summary:

```text
files: 18
rows: 927
training_rows: 909
results: 13 Victory / 3 Defeat / 2 Tie
Hard defeat files:
  data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1\20260623_102913_AcropolisLE_Hard_Protoss_002.jsonl
  data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1\20260623_103024_AcropolisLE_Hard_Terran_001.jsonl
  data\trajectories\strategy_coverage_teacher_v2_pending_tuned_v1\20260623_103133_AcropolisLE_Hard_Terran_002.jsonl

action_timing:
  STAY_COURSE: count=490 first=0.0 avg=264.5 min=0.0 max=891.4
  EXPAND: count=11 first=285.7 avg=250.4 min=217.1 max=297.1
  ADD_GATEWAYS: count=69 first=91.4 avg=312.2 min=91.4 max=617.1
  TECH_ROBO: count=130 first=240.0 avg=386.9 min=240.0 max=685.7
  FORGE_UPGRADES: count=24 first=457.1 avg=429.5 min=365.7 max=537.1
  BUILD_STATIC_DEFENSE: count=50 first=262.9 avg=517.0 min=262.9 max=880.0
  PRODUCE_ARMY: count=69 first=205.7 avg=266.8 min=114.3 max=628.6
  BOOST_WORKERS: count=66 first=57.1 avg=259.4 min=57.1 max=891.4

threat_action_counts:
  STAY_COURSE: 7
  TECH_ROBO: 13
  FORGE_UPGRADES: 1
  BUILD_STATIC_DEFENSE: 50
  BOOST_WORKERS: 4

pending_repeat_counts:
  ADD_GATEWAYS: 54
  BUILD_STATIC_DEFENSE: 14

TECH_ROBO signal timing:
  armored_signal: files_with_signal=13, tech_after=0, tech_before=13, no_tech=0, avg_early_lead=121.3
  cloaked_signal: files_with_signal=6, tech_after=0, tech_before=6, no_tech=0, avg_early_lead=173.3
```

Hard failure interpretation:

```text
The candidate did not collapse to a single action online, but its timing differs
from teacher in ways that plausibly explain Hard weakness.

Largest timing gap:
  imitation ADD_GATEWAYS first=411.4 avg=582.7
  teacher    ADD_GATEWAYS first=91.4  avg=312.2

Threat response:
  imitation under threat is almost entirely BUILD_STATIC_DEFENSE.
  teacher under threat still includes TECH_ROBO and occasional non-static actions.

Hard Terran imitation defeat:
  No ADD_GATEWAYS, no TECH_ROBO, no FORGE_UPGRADES.
  Long BUILD_STATIC_DEFENSE run from about 331s to 469s.

TECH_ROBO teacher bug not supported:
  In coverage-teacher files with armored/cloaked signals, TECH_ROBO always
  happened before the signal. The earlier "missing" metric was misleading
  because it conflated early tech with no tech.
```

Conclusion:

```text
Do not change CoverageStrategyPolicy yet.
Do not collect blind data solely from this timing pass.
Next best step is an offline teacher-vs-imitation agreement diagnostic by time
bucket and state bucket. If that confirms the learned checkpoint misses early
Gate/Tech or over-selects static defense under pressure, collect a new
Hard-focused strategy_v2 coverage-teacher dataset using a new directory name
and retrain strategy_imitation_v2_candidate_v2 only after diagnostics are healthy.
PPO remains not recommended.
```

## 2026-06-23 AIBuild / Tactic Pool Planning

Scope:

```text
Write a next-development plan for tactic-pool work.
No SC2 launch.
No PPO.
No runtime code changes.
```

Docs:

```text
TACTIC_POOL_PLAN.md
  Added a staged plan to first connect official built-in AI AIBuild as an
  evaluation/data-collection dimension, then design a local TacticSpec pool.

CODEX.md
  Added TACTIC_POOL_PLAN.md to the read-first map for tactic-pool work.

README.md
  Added TACTIC_POOL_PLAN.md to the documentation map.
```

Plan summary:

```text
1. Add run.py --ai-build and scripts/evaluate.py --ai-builds.
2. Store opponent_ai_build in eval records, experiment config, and army/strategy trajectories.
3. Summarize and diagnose evals by opponent_ai_build.
4. Use official Rush / Timing / Power / Macro / Air only as scenario labels and data buckets.
5. After AIBuild data is healthy, design TacticSpec / TacticState / TacticSelector.
6. Keep tactic behavior explicit opt-in and preserve default rule/no-op behavior.
7. Do not enter PPO.
```

## 2026-06-23 AIBuild Evaluation Dimension Implementation

Scope:

```text
Continue strategy-layer development.
Do not implement PPO.
Do not change RL/PPO mainline.
Do not break default rule baseline.
Keep strategy/tactic explicit opt-in; default remains rule/no-op.
Use hidden-window guard and scripts/evaluate.py for SC2 smoke.
Do not mix new collection with old trajectory directories.
```

Initial health checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Initial results:

```text
pytest: 121 passed in 2.84s
check_env: all OK
```

Code changes:

```text
run.py
  Added AIBUILD_MAP for RandomBuild / Rush / Timing / Power / Macro / Air.
  Added --ai-build, default RandomBuild.
  Passed Computer(..., ai_build=AIBUILD_MAP[args.ai_build]).
  Added opponent_ai_build to episode_metadata.

scripts\evaluate.py
  Added --ai-builds, default RandomBuild.
  Expanded eval matrix to map x difficulty x opponent race x ai_build x game.
  Added opponent_ai_build to EvalRecord / eval.jsonl.
  Added ai_builds to experiment config.
  Forwarded --ai-build to run.py through scripts\safe_launch.py.
  Added opponent_ai_build to new trajectory and LLM-log filenames.

scripts\summarize_eval.py
  Added opponent_ai_build to EvalSummary.
  Grouped summaries by opponent_ai_build and displayed build in the table.
  Defaults missing old eval records to RandomBuild.

rl\trajectory_recorder.py
  Added opponent_ai_build metadata to TrajectoryStep and StrategyTrajectoryStep.
  Kept a default of RandomBuild for backward-compatible construction.

bot\protoss_rule_bot.py
  Records opponent_ai_build from episode_metadata into army and strategy
  trajectory rows.
  Observation schemas were not changed.
```

Tests updated:

```text
tests\test_evaluate.py
  Covers --ai-builds default, config metadata, command forwarding, record field,
  and build-labeled trajectory filenames.

tests\test_summarize_eval.py
  Covers grouping by opponent_ai_build and defaulting old rows to RandomBuild.

tests\test_strategy_trajectory_recording.py
  Covers ProtossRuleBot army and strategy trajectory build metadata.

tests\test_rl_trajectory_modules.py
  Covers JSONL serialization for army/strategy opponent_ai_build metadata.
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_evaluate.py tests\test_summarize_eval.py tests\test_strategy_trajectory_recording.py tests\test_rl_trajectory_modules.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results:

```text
focused tests: 26 passed
full pytest: 124 passed in 3.11s
check_env: all OK
```

Smoke safety:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Result:

```text
guard pid: 21392
trajectory dirs before smoke:
  data\trajectories\aibuild_smoke_army_v1 missing
  data\trajectories\aibuild_smoke_strategy_v1 missing
```

Guarded smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Medium --opponents Terran --ai-builds Rush --games-per-combo 1 --run-root runs --run-name eval_aibuild_smoke_v1 --policy-name rule_aibuild_smoke --army-policy rule --strategy-policy rule --trajectory-dir data\trajectories\aibuild_smoke_army_v1 --strategy-trajectory-dir data\trajectories\aibuild_smoke_strategy_v1 --record-decision-interval 16 --game-time-limit 600
```

Smoke result:

```text
run: runs\20260623_144443_eval_aibuild_smoke_v1
eval: runs\20260623_144443_eval_aibuild_smoke_v1\artifacts\eval.jsonl
summary: runs\20260623_144443_eval_aibuild_smoke_v1\artifacts\summary.json
army trajectory: data\trajectories\aibuild_smoke_army_v1\20260623_144443_AcropolisLE_Medium_Terran_Rush_001.jsonl
strategy trajectory: data\trajectories\aibuild_smoke_strategy_v1\20260623_144443_AcropolisLE_Medium_Terran_Rush_001.jsonl
return_code: 0
result: Result.Tie
duration_seconds: 56.2
```

Metadata check:

```text
eval.jsonl:
  opponent_ai_build=Rush
  return_code=0

summary.json:
  opponent_ai_build=Rush
  games=1
  ties=1
  failures=0

metadata.json config:
  ai_builds={Rush}
  army_policy=rule
  strategy_policy=rule

army trajectory:
  files=1
  rows=212
  unique opponent_ai_build=Rush

strategy trajectory:
  files=1
  rows=54
  unique opponent_ai_build=Rush
```

Compatibility / behavior:

```text
Default run.py --ai-build remains RandomBuild.
Default scripts/evaluate.py --ai-builds remains RandomBuild, so older evaluate
commands remain compatible.
Default army policy remains rule unless explicitly changed.
Default strategy policy remains rule/no-op unless explicitly changed.
No tactic layer was implemented.
No PPO was implemented or touched.
SC2 was launched only through scripts/evaluate.py -> scripts/safe_launch.py after
the hidden-window guard was confirmed.
```

Recommended next step:

```text
Run a small guarded AIBuild matrix for rule and coverage-teacher using new
run/trajectory directories. Compare by opponent_race + opponent_ai_build before
starting the TacticSpec / TacticSelector rule design.
```

## 2026-06-23 AIBuild Hard Terran Matrix + TacticSpec Skeleton

Scope:

```text
Continue strategy-layer development after AIBuild metadata implementation.
Do not implement PPO.
Do not change PPO/RL mainline.
Do not break default rule/no-op baseline.
Run a small guarded build-labeled matrix first, then implement a non-default
TacticSpec skeleton from the observed data shape.
```

Health checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Initial results:

```text
pytest: 124 passed in 2.73s
check_env: all OK
new trajectory dirs checked missing before collection:
  data\trajectories\rule_aibuild_hard_terran_army_v1
  data\trajectories\rule_aibuild_hard_terran_strategy_v1
  data\trajectories\coverage_teacher_aibuild_hard_terran_army_v1
  data\trajectories\coverage_teacher_aibuild_hard_terran_strategy_v1
```

Safety:

```text
Hidden-window guard checked before each SC2 batch.
guard pid: 21392
SC2 launched only through scripts\evaluate.py -> scripts\safe_launch.py.
No visible run.py launch.
```

Rule/no-op AIBuild matrix:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Rush Timing Power Macro Air --games-per-combo 1 --run-root runs --run-name eval_rule_aibuild_hard_terran_v1 --policy-name rule_aibuild_hard_terran_v1 --army-policy rule --strategy-policy rule --trajectory-dir data\trajectories\rule_aibuild_hard_terran_army_v1 --strategy-trajectory-dir data\trajectories\rule_aibuild_hard_terran_strategy_v1 --record-decision-interval 16 --game-time-limit 900
```

Rule/no-op results:

```text
run: runs\20260623_145519_eval_rule_aibuild_hard_terran_v1
eval: runs\20260623_145519_eval_rule_aibuild_hard_terran_v1\artifacts\eval.jsonl
strategy trajectory: data\trajectories\rule_aibuild_hard_terran_strategy_v1
return_code=0 for all 5 games
Rush: Result.Victory
Timing: Result.Victory
Power: Result.Defeat
Macro: Result.Defeat
Air: Result.Tie
strategy rows=291, training_rows=286, strategy_v2 only, defaulted=0
strategy actions: STAY_COURSE=286
```

Coverage-teacher AIBuild matrix:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Rush Timing Power Macro Air --games-per-combo 1 --run-root runs --run-name eval_coverage_teacher_aibuild_hard_terran_v1 --policy-name coverage_teacher_aibuild_hard_terran_v1 --army-policy rule --strategy-policy coverage-teacher --trajectory-dir data\trajectories\coverage_teacher_aibuild_hard_terran_army_v1 --strategy-trajectory-dir data\trajectories\coverage_teacher_aibuild_hard_terran_strategy_v1 --record-decision-interval 16 --game-time-limit 900
```

Coverage-teacher results:

```text
run: runs\20260623_150115_eval_coverage_teacher_aibuild_hard_terran_v1
eval: runs\20260623_150115_eval_coverage_teacher_aibuild_hard_terran_v1\artifacts\eval.jsonl
strategy trajectory: data\trajectories\coverage_teacher_aibuild_hard_terran_strategy_v1
strategy timing: runs\20260623_150115_eval_coverage_teacher_aibuild_hard_terran_v1\artifacts\strategy_timing_diagnostics.json
return_code=0 for all 5 games
Rush: Result.Victory
Timing: Result.Victory
Power: Result.Defeat
Macro: Result.Victory
Air: Result.Tie
strategy rows=278, training_rows=273, strategy_v2 only, defaulted=0
action_coverage=100%
actions:
  STAY_COURSE=116
  EXPAND=1
  ADD_GATEWAYS=19
  TECH_ROBO=65
  FORGE_UPGRADES=4
  BUILD_STATIC_DEFENSE=8
  PRODUCE_ARMY=48
  BOOST_WORKERS=12
```

Per-build strategy action counts:

```text
rule/no-op:
  Rush: STAY_COURSE=41
  Timing: STAY_COURSE=42
  Power: STAY_COURSE=66
  Macro: STAY_COURSE=58
  Air: STAY_COURSE=79

coverage-teacher:
  Rush: STAY_COURSE=16, ADD_GATEWAYS=2, TECH_ROBO=17, FORGE_UPGRADES=1, PRODUCE_ARMY=1, BOOST_WORKERS=2
  Timing: STAY_COURSE=31, ADD_GATEWAYS=2, TECH_ROBO=1, FORGE_UPGRADES=1, PRODUCE_ARMY=8, BOOST_WORKERS=1
  Power: STAY_COURSE=14, ADD_GATEWAYS=3, TECH_ROBO=38, BUILD_STATIC_DEFENSE=6, PRODUCE_ARMY=6, BOOST_WORKERS=3
  Macro: STAY_COURSE=26, ADD_GATEWAYS=2, TECH_ROBO=3, BUILD_STATIC_DEFENSE=1, PRODUCE_ARMY=7, BOOST_WORKERS=2
  Air: STAY_COURSE=29, EXPAND=1, ADD_GATEWAYS=10, TECH_ROBO=6, FORGE_UPGRADES=2, BUILD_STATIC_DEFENSE=1, PRODUCE_ARMY=26, BOOST_WORKERS=4
```

Interpretation:

```text
Macro is the first clear build-labeled difference: rule/no-op lost, while
coverage-teacher won with a light mix of early ADD_GATEWAYS, PRODUCE_ARMY,
3 TECH_ROBO actions, and one static-defense action.

Power is a shared failure. coverage-teacher emitted TECH_ROBO 38 times, then
BUILD_STATIC_DEFENSE 6 times under threat. This supports using TacticSpec as a
conservative filter/cooldown layer before online tactic-aware rollout.

Air remains Tie for both and is a useful anti-air tactic smoke target later.
```

TacticSpec code changes:

```text
rl\tactics.py
  Added TacticId:
    SAFE_MACRO
    ANTI_RUSH_DEFENSE
    GATEWAY_PRESSURE
    ROBO_TIMING
    TECH_POWER
    ANTI_AIR_RESPONSE
    RECOVERY
  Added TacticPhase, TacticSpec, TacticState.
  Added DEFAULT_TACTIC_POOL with allowed/preferred/avoid actions, AIBuild hints,
  resource reserves, pending caps, and transition/abort metadata.
  Added filter_strategy_action() to conservatively replace disallowed or repeated
  strategy actions.

bot\managers\tactic_selector.py
  Added RuleTacticSelector.
  Selects a tactic from strategy observation metadata plus opponent_ai_build.
  Has minimum tactic duration cooldown.
  Allows emergency RECOVERY / ANTI_RUSH_DEFENSE / ANTI_AIR_RESPONSE overrides.

bot\managers\__init__.py
  Exports RuleTacticSelector.

tests\test_tactics.py
  Covers tactic pool shape, AIBuild hints, action filtering, selector decisions,
  cooldown behavior, and emergency override behavior.
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Results:

```text
test_tactics: 5 passed
full pytest: 129 passed in 2.87s
```

Runtime/default behavior:

```text
TacticSpec is not wired into the default runtime.
No --strategy-tactic-mode exists yet.
No observation schema was changed.
No strategy action space was changed.
Default --strategy-policy rule remains no-op.
PPO was not implemented or touched.
```

Recommended next step:

```text
Add an explicit opt-in tactic-aware strategy policy or mode that wraps
CoverageStrategyPolicy:

  CoverageStrategyPolicy proposes StrategyAction
  RuleTacticSelector chooses TacticId from observation + opponent_ai_build
  filter_strategy_action() applies TacticSpec constraints
  StrategyExecutor executes the filtered action

Start with focused unit tests for Hard Terran Macro/Power states, then run a
guarded Macro/Power smoke before expanding the AIBuild matrix.
```

## 2026-06-23 Opt-in Tactic-Aware Strategy Policy + Diagnostics

Scope:

```text
Continue strategy-layer development after the AIBuild and TacticSpec skeleton.
Do not implement PPO.
Do not change the PPO/RL mainline.
Keep default rule/no-op baseline unchanged.
Wire tactic-aware behavior only through an explicit opt-in mode.
Add tactic metadata/filter-change diagnostics before larger collections.
```

Runtime changes:

```text
run.py
  Added --strategy-tactic-mode off|rule, default off.
  Non-off mode currently requires --strategy-policy coverage-teacher.
  --strategy-policy coverage-teacher --strategy-tactic-mode rule wraps
  CoverageStrategyPolicy with TacticAwareStrategyPolicy.

scripts/evaluate.py
  Added --strategy-tactic-mode off|rule, default off.
  Records strategy_tactic_mode in EvalRecord and experiment config.
  Forwards --strategy-tactic-mode to run.py only when the mode is not off.

bot\managers\tactic_strategy_policy.py
  Added TacticAwareStrategyPolicy.
  Base policy proposes StrategyAction.
  RuleTacticSelector chooses TacticState from strategy observation +
  opponent_ai_build.
  filter_strategy_action() applies TacticSpec constraints.
  Bot metadata records tactic and before/after action filter state.

rl\trajectory_recorder.py / bot\protoss_rule_bot.py
  Strategy trajectory rows now record tactic metadata and before/after
  strategy action filter fields.
  Army trajectory metadata remains limited to opponent_ai_build.
  Observation schemas were not changed.
```

New diagnostics:

```text
rl\tactic_diagnostics.py
scripts\diagnose_tactics.py
tests\test_tactic_diagnostics.py
```

The tactic diagnostic reports:

```text
rows / training_rows
rows_with_tactic_metadata
rows_with_filter_metadata
filter_change_rows
training_rows_with_tactic_metadata
training_rows_with_filter_metadata
training_filter_change_rows
opponent_ai_build_counts
tactic_counts
tactic_phase_counts
tactic_source_counts
filter_changes by opponent_ai_build / tactic_id / before / after
per-file tactic timelines
```

Focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tactic_strategy_policy.py tests\test_strategy_trajectory_recording.py tests\test_rl_trajectory_modules.py tests\test_evaluate.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_tactic_diagnostics.py tests\test_strategy_timing_diagnostics.py tests\test_tactic_strategy_policy.py -q
```

Results:

```text
first focused set: 26 passed
diagnostic focused set: 11 passed
```

Full tests before tactic diagnostic addition:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
133 passed in 2.90s
```

Final full tests after tactic diagnostic addition:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
136 passed in 3.02s
```

Final environment/process check:

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Result:

```text
check_env: all OK
guard pid: 21392
no SC2_x64.exe / SC2.exe / scripts\evaluate.py / scripts\safe_launch.py /
run.py residual process after excluding the query process itself
```

Environment check:

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

Result:

```text
all OK
```

Safety:

```text
Hidden-window guard was confirmed before the SC2 smoke.
guard pid: 21392
SC2 was launched only through scripts\evaluate.py -> scripts\safe_launch.py.
No naked visible run.py launch.
No SC2 launch was needed for the offline tactic diagnostic script.
```

Tactic-aware guarded smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Macro Power --games-per-combo 1 --run-root runs --run-name eval_tactic_coverage_aibuild_hard_terran_v1 --policy-name tactic_coverage_aibuild_hard_terran_v1 --army-policy rule --strategy-policy coverage-teacher --strategy-tactic-mode rule --trajectory-dir data\trajectories\tactic_coverage_aibuild_hard_terran_army_v1 --strategy-trajectory-dir data\trajectories\tactic_coverage_aibuild_hard_terran_strategy_v1 --record-decision-interval 16 --game-time-limit 900
```

Smoke results:

```text
run: runs\20260623_152644_eval_tactic_coverage_aibuild_hard_terran_v1
eval: runs\20260623_152644_eval_tactic_coverage_aibuild_hard_terran_v1\artifacts\eval.jsonl
strategy trajectory: data\trajectories\tactic_coverage_aibuild_hard_terran_strategy_v1
return_code=0 for both games
Macro: Result.Victory
Power: Result.Victory
```

Strategy trajectory diagnostics:

```text
strategy rows=88
training_rows=86
strategy_v2 only
rows_defaulted_observation_fields=0
results: Victory=2
pending_repeat_counts: <none>
hard_defeat_files: <none>
```

Tactic diagnostic command:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_tactics.py data\trajectories\tactic_coverage_aibuild_hard_terran_strategy_v1 --show-files --json-output runs\20260623_152644_eval_tactic_coverage_aibuild_hard_terran_v1\artifacts\tactic_diagnostics.json
```

Tactic diagnostic results:

```text
files=2
rows=88
training_rows=86
rows_with_tactic_metadata=88
rows_with_filter_metadata=88
filter_change_rows=29
training_rows_with_tactic_metadata=86
training_rows_with_filter_metadata=86
training_filter_change_rows=28
opponent_ai_build_counts: Macro=46, Power=40
tactic_counts:
  SAFE_MACRO=39
  TECH_POWER=19
  ANTI_AIR_RESPONSE=11
  GATEWAY_PRESSURE=9
  ANTI_RUSH_DEFENSE=8
filter_changes:
  Power, TECH_POWER, TECH_ROBO -> PRODUCE_ARMY: 18 training rows
  Macro, ANTI_RUSH_DEFENSE, TECH_ROBO -> PRODUCE_ARMY: 3
  Macro, GATEWAY_PRESSURE, TECH_ROBO -> PRODUCE_ARMY: 2
  Macro, SAFE_MACRO, ADD_GATEWAYS -> BOOST_WORKERS: 2
  Macro, ANTI_AIR_RESPONSE, TECH_ROBO -> PRODUCE_ARMY: 1
  Macro, GATEWAY_PRESSURE, FORGE_UPGRADES -> ADD_GATEWAYS: 1
  Power, SAFE_MACRO, ADD_GATEWAYS -> BOOST_WORKERS: 1
terminal-inclusive Power TECH_POWER TECH_ROBO -> PRODUCE_ARMY count: 19
```

Behavior boundaries:

```text
Default run.py remains --strategy-policy rule --strategy-tactic-mode off.
Default scripts/evaluate.py remains --ai-builds RandomBuild and
--strategy-tactic-mode off.
Old evaluate commands remain compatible.
Tactic-aware behavior is explicit opt-in only:
  --strategy-policy coverage-teacher --strategy-tactic-mode rule
No observation schema change.
No new strategy action space.
No PPO was implemented or touched.
```

Recommended next step:

```text
Use scripts\diagnose_tactics.py for every tactic-aware collection, then expand
guarded paired comparisons across more AIBuild/race combos:

1. rule/no-op
2. coverage-teacher without tactic filter
3. coverage-teacher with --strategy-tactic-mode rule

Use fresh run and trajectory directories for every new collection.
```

## 2026-06-23 Tactic Filter Follow-up Matrix and Power Recheck

Scope:

```text
Continue strategy/tactic layer work after the initial tactic-aware smoke.
Do not implement PPO.
Do not alter the default rule/no-op path.
Use guarded SC2 launches only through scripts\evaluate.py / scripts\safe_launch.py.
Use fresh trajectory directories for each collection.
```

Code changes:

```text
rl\tactics.py
  Safe EXPAND proposals from CoverageStrategyPolicy now pass through tactic
  filters when base_under_threat is false.
  RECOVERY fallback now prioritizes PRODUCE_ARMY before BOOST_WORKERS.
  ROBO_TIMING / ANTI_AIR_RESPONSE can return STAY_COURSE to save minerals for
  the first Robotics Facility instead of spending low minerals on army fallback.
  TECH_POWER keeps the old PRODUCE_ARMY fallback for low-mineral TECH_ROBO.

bot\managers\tactic_strategy_policy.py
  TacticAwareStrategyPolicy now records tactic metadata for all builds but only
  filters actions for configured opponent builds.
  Default filter_opponent_ai_builds is ("Power",), making the online filter
  Power-targeted and leaving non-Power coverage-teacher actions unchanged.

tests\test_tactics.py / tests\test_tactic_strategy_policy.py
  Added focused coverage for safe expand pass-through, initial Robo mineral
  saving, RECOVERY fallback priority, TECH_POWER fallback, and non-Power
  metadata-only behavior.
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py tests\test_strategy_trajectory_recording.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Results:

```text
focused: 17 passed
full pytest after Power-targeted policy: 141 passed in 2.89s
```

Safety:

```text
Guard checked before each SC2 batch.
guard pid: 21392
All SC2 runs used scripts\evaluate.py -> scripts\safe_launch.py.
No naked visible run.py launch.
Fresh trajectory dirs were checked missing before new collections.
```

Tactic-aware all-build v1, broad filter:

```text
run: runs\20260623_154821_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v1
strategy trajectory: data\trajectories\tactic_coverage_aibuild_hard_terran_allbuilds_strategy_v1
results:
  Rush: Result.Defeat
  Timing: Result.Defeat
  Power: Result.Victory
  Macro: Result.Defeat
  Air: Result.Defeat
return_code=0 for all 5 games
tactic diagnostics:
  rows=298
  training_rows=293
  rows_with_tactic_metadata=298
  filter_change_rows=116
  training_filter_change_rows=114
```

Tactic-aware all-build v2, safer TacticSpec filter:

```text
run: runs\20260623_160120_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v2
strategy trajectory: data\trajectories\tactic_coverage_aibuild_hard_terran_allbuilds_strategy_v2
results:
  Rush: Result.Defeat
  Timing: Result.Victory
  Power: Result.Defeat
  Macro: Result.Tie
  Air: Result.Tie
return_code=0 for all 5 games
tactic diagnostics:
  rows=301
  training_rows=296
  rows_with_tactic_metadata=301
  filter_change_rows=101
  training_filter_change_rows=100
```

Tactic-aware all-build v3, Power-targeted filter:

```text
run: runs\20260623_160924_eval_tactic_coverage_aibuild_hard_terran_allbuilds_v3
strategy trajectory: data\trajectories\tactic_coverage_aibuild_hard_terran_allbuilds_strategy_v3
results:
  Rush: Result.Victory
  Timing: Result.Defeat
  Power: Result.Defeat
  Macro: Result.Victory
  Air: Result.Defeat
return_code=0 for all 5 games
tactic diagnostics:
  rows=280
  training_rows=275
  rows_with_tactic_metadata=280
  filter_change_rows=12
  training_filter_change_rows=12
  filter changes were all on Power rows
```

Power-only no-filter coverage-teacher recheck:

```text
run: runs\20260623_161606_eval_coverage_teacher_aibuild_hard_terran_power_recheck_v1
strategy trajectory: data\trajectories\coverage_teacher_aibuild_hard_terran_power_recheck_strategy_v1
games: 3
results:
  Result.Tie
  Result.Victory
  Result.Defeat
summary: 1W / 1T / 1L
return_code=0 for all 3 games
```

Power-only Power-targeted tactic recheck:

```text
run: runs\20260623_162021_eval_tactic_power_targeted_hard_terran_power_recheck_v1
strategy trajectory: data\trajectories\tactic_power_targeted_hard_terran_power_recheck_strategy_v1
games: 3
results:
  Result.Tie
  Result.Tie
  Result.Defeat
summary: 0W / 2T / 1L
return_code=0 for all 3 games
tactic diagnostics:
  rows=226
  training_rows=223
  rows_with_tactic_metadata=226
  filter_change_rows=46
  training_filter_change_rows=46
  filter changes were all Power rows
```

Interpretation:

```text
The initial two-game Macro/Power tactic-aware smoke was not representative.
Broad filtering hurts non-Power builds and must not be used for collection.
Power-targeted filtering is safer because it leaves non-Power actions unchanged,
but it did not beat no-filter coverage-teacher in the 3-game Power recheck.

Do not promote tactic-aware mode.
Do not collect tactic-aware coverage-teacher training data yet.
Next work should inspect Power defeats/ties offline and refine TECH_POWER before
another guarded Power-only paired comparison.
```

Final checks:

```text
pytest: 141 passed in 2.82s
check_env: all OK
guard pid: 21392
no SC2_x64.exe / SC2.exe / scripts\evaluate.py / scripts\safe_launch.py /
run.py residual process after excluding the query process itself
```

## 2026-06-23 Offline Power Tactic Failure Diagnostics

Scope:

```text
Continue strategy/tactic layer development without launching SC2.
Do not implement PPO.
Do not train.
Do not collect tactic-aware training data.
Do not change PPO/RL mainline.
Do not change default rule/no-op behavior.
Keep --strategy-policy rule / --strategy-tactic-mode off as default.
Do not add tactic metadata to observation schema.
```

Code changes:

```text
rl\power_tactic_diagnostics.py
  New offline diagnostics module for Power-build strategy trajectory files.
  Computes per-file result, action timing, action timeline, tactic timeline,
  filter changes, filter counterfactual deltas, Robo/Forge/upgrade/static
  defense timings, Observer/Immortal timings, PRODUCE_ARMY timing, base threat
  action counts, army_count thresholds, worker saturation, gateway/idle gateway
  stats, and minerals/vespene bank rows.

scripts\diagnose_power_tactics.py
  New CLI for human-readable report plus optional JSON and text outputs.

tests\test_power_tactic_diagnostics.py
  Focused unit coverage for Power signal timings, filter counterfactuals, tactic
  timelines, threat stats, economy/gateway stats, and report formatting.
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_power_tactic_diagnostics.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results:

```text
focused pytest: 3 passed
full pytest: 144 passed in 2.66s
check_env: all OK
SC2 launched: no
guard launched/checked: no, because this task did not start SC2
PPO/training: no
```

Offline diagnostic run:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_power_tactics.py data\trajectories\coverage_teacher_aibuild_hard_terran_power_recheck_strategy_v1 data\trajectories\tactic_power_targeted_hard_terran_power_recheck_strategy_v1 --show-files --json-output runs\20260623_power_tactic_diagnostics_v1\artifacts\power_tactic_diagnostics.json --text-output runs\20260623_power_tactic_diagnostics_v1\artifacts\power_tactic_diagnostics.txt
```

Artifacts:

```text
runs\20260623_power_tactic_diagnostics_v1\artifacts\power_tactic_diagnostics.txt
runs\20260623_power_tactic_diagnostics_v1\artifacts\power_tactic_diagnostics.json
```

Inputs:

```text
data\trajectories\coverage_teacher_aibuild_hard_terran_power_recheck_strategy_v1
data\trajectories\tactic_power_targeted_hard_terran_power_recheck_strategy_v1
```

Dataset summary:

```text
files=6
rows=437
training_rows=431
results:
  Victory=1
  Tie=3
  Defeat=2
opponent_ai_build:
  Power=431 training rows
tactic metadata:
  rows_with_tactic_metadata=226
  rows_with_filter_metadata=226
  filter_change_rows=46
  training_filter_change_rows=46
```

Aggregate action timing:

```text
STAY_COURSE: count=272 first=0.0 avg=416.3
EXPAND: count=2 first=297.1 avg=257.1
ADD_GATEWAYS: count=19 first=91.4 avg=439.1
TECH_ROBO: count=18 first=240.0 avg=338.4
FORGE_UPGRADES: count=17 first=365.7 avg=478.7
BUILD_STATIC_DEFENSE: count=22 first=262.9 avg=638.4
PRODUCE_ARMY: count=60 first=114.3 avg=414.7
BOOST_WORKERS: count=21 first=57.1 avg=145.9
```

Power-targeted filter counterfactual:

```text
filter_changes:
  TECH_POWER, TECH_ROBO -> PRODUCE_ARMY: 15
  TECH_POWER, ADD_GATEWAYS -> FORGE_UPGRADES: 11
  TECH_POWER, BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 8
  SAFE_MACRO, ADD_GATEWAYS -> BOOST_WORKERS: 5
  ANTI_AIR_RESPONSE, FORGE_UPGRADES -> TECH_ROBO: 2
  TECH_POWER, BUILD_STATIC_DEFENSE -> FORGE_UPGRADES: 2
  ANTI_AIR_RESPONSE, BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 1
  RECOVERY, BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 1
  RECOVERY, TECH_ROBO -> PRODUCE_ARMY: 1

counterfactual_filter_delta:
  ADD_GATEWAYS=-16
  BOOST_WORKERS=+5
  BUILD_STATIC_DEFENSE=-12
  FORGE_UPGRADES=+11
  PRODUCE_ARMY=+26
  TECH_ROBO=-14
```

Per-file Power findings:

```text
No-filter coverage-teacher:
  001 Tie:
    first TECH_ROBO=240.0, pending_robo=320.0, ready_robo=365.7
    no Observer/Immortal
    base_under_threat rows=7, all BUILD_STATIC_DEFENSE
  002 Victory:
    first TECH_ROBO=251.4, pending_robo=285.7, ready_robo=331.4
    no base_under_threat rows
    max army_count=26
  003 Defeat:
    first TECH_ROBO=251.4, pending_robo=285.7, ready_robo=331.4
    upgrade completed at 674.3
    base_under_threat rows=12, all BUILD_STATIC_DEFENSE

Power-targeted tactic filter:
  001 Tie:
    actual TECH_ROBO delayed to 628.6, pending_robo=640.0, ready_robo=685.7
    filter delta: TECH_ROBO=-11, ADD_GATEWAYS=-13, PRODUCE_ARMY=+11,
      FORGE_UPGRADES=+13
    max army_count=69, but no Observer/Immortal
  002 Tie:
    first TECH_ROBO=274.3, ready_robo=331.4
    threat rows=7, all PRODUCE_ARMY after static-defense suppression
    no Forge/upgrade
  003 Defeat:
    first TECH_ROBO=262.9, ready_robo=320.0
    first Observer=411.4, no Immortal
    final recovery segment present, still Defeat
```

Interpretation:

```text
The Power-targeted tactic filter is not ready to promote. It reduces
BUILD_STATIC_DEFENSE spam, but it also suppresses too many TECH_ROBO and
ADD_GATEWAYS labels, and in one tie delays actual Robo far past the no-filter
timing. The 3-game recheck outcome remains worse than no-filter coverage-teacher
(0W/2T/1L vs 1W/1T/1L).

Next step should be offline counterfactual/spec work for TECH_POWER:
  - preserve one timely Robo before repeated PRODUCE_ARMY fallback
  - avoid converting too many ADD_GATEWAYS into FORGE_UPGRADES
  - keep static-defense caps, but validate replacement choice under threat
  - watch no-Immortal/no-Observer outcomes before another guarded Power-only
    paired comparison

Do not collect tactic-aware training data from this filter.
Do not start PPO from this state.
```

## 2026-06-23 Revised TECH_POWER Counterfactual Spec Tests

Scope:

```text
Continue from offline Power failure diagnostics.
Write focused tests before changing TECH_POWER behavior.
Do not launch SC2.
Do not train.
Do not implement PPO.
Do not change default rule/no-op runtime.
Keep tactic-aware behavior explicit opt-in only.
Do not change observation schema.
```

Code changes:

```text
rl\tactics.py
  Revised TECH_POWER fallback order from FORGE_UPGRADES-first to:
    TECH_ROBO -> PRODUCE_ARMY -> ADD_GATEWAYS -> FORGE_UPGRADES
  TECH_POWER now saves low minerals for the first Robo instead of converting the
  first TECH_ROBO proposal to PRODUCE_ARMY when no ready/pending Robo exists.
  TECH_POWER no longer accepts repeated TECH_ROBO once a Robo is already ready.
  Under threat, rejected BUILD_STATIC_DEFENSE now falls back to PRODUCE_ARMY.

tests\test_tactics.py
  Added focused counterfactual tests for:
    - saving for initial TECH_POWER Robo
    - capped ADD_GATEWAYS -> first TECH_ROBO
    - capped ADD_GATEWAYS -> PRODUCE_ARMY after Robo started
    - capped/rejected static defense under threat -> PRODUCE_ARMY
    - ready Robo plus missing Observer/Immortal -> PRODUCE_ARMY, not repeat Robo

tests\test_tactic_strategy_policy.py
  Added wrapper-level opt-in Power tests for:
    - ADD_GATEWAYS proposal filtered to TECH_ROBO when gateways are capped and
      no Robo has started
    - BUILD_STATIC_DEFENSE proposal filtered to PRODUCE_ARMY under threat
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results:

```text
focused pytest: 19 passed
full pytest: 150 passed in 2.79s
check_env: all OK
SC2 launched: no
guard launched/checked: no, because this task did not start SC2
PPO/training: no
```

Current interpretation:

```text
This is only an offline spec/counterfactual adjustment. It directly addresses
the diagnosed over-conversion of TECH_ROBO and ADD_GATEWAYS into PRODUCE_ARMY /
FORGE_UPGRADES, while retaining static-defense repeat protection.

The revised filter is not promoted yet. Before any collection or training, run a
guarded Power-only paired comparison with fresh run and trajectory directories:
  1. no-filter coverage-teacher
  2. coverage-teacher + revised Power-targeted tactic filter

Do not broaden to all builds unless the revised Power comparison beats or at
least clearly matches no-filter coverage-teacher without new failure modes.
```

## 2026-06-23 Guarded Revised Power Tactic A/B

Scope:

```text
Run one guarded fresh-dir Power-only A/B comparison:
  A. coverage-teacher without tactic filter
  B. coverage-teacher with revised Power-targeted tactic filter
Do not train.
Do not implement PPO.
Do not use broad/all-build collection.
Use scripts\evaluate.py only.
Run hidden-window guard before each SC2-launching batch.
```

Fresh dirs checked missing before launch:

```text
runs\20260623_eval_power_ab_no_filter_revised_v1
runs\20260623_eval_power_ab_revised_tactic_v1
data\trajectories\power_ab_no_filter_revised_army_v1
data\trajectories\power_ab_no_filter_revised_strategy_v1
data\trajectories\power_ab_revised_tactic_army_v1
data\trajectories\power_ab_revised_tactic_strategy_v1
```

Guard:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Result:

```text
guard pid: 21392
```

No-filter command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 3 --run-root runs --run-name 20260623_eval_power_ab_no_filter_revised_v1 --policy-name coverage_teacher_power_ab_no_filter_revised_v1 --army-policy rule --strategy-policy coverage-teacher --trajectory-dir data\trajectories\power_ab_no_filter_revised_army_v1 --strategy-trajectory-dir data\trajectories\power_ab_no_filter_revised_strategy_v1 --record-decision-interval 16 --game-time-limit 900
```

No-filter result:

```text
run: runs\20260623_170757_20260623_eval_power_ab_no_filter_revised_v1
eval: runs\20260623_170757_20260623_eval_power_ab_no_filter_revised_v1\artifacts\eval.jsonl
summary: runs\20260623_170757_20260623_eval_power_ab_no_filter_revised_v1\artifacts\summary.json
army trajectory: data\trajectories\power_ab_no_filter_revised_army_v1
strategy trajectory: data\trajectories\power_ab_no_filter_revised_strategy_v1
games:
  Result.Defeat, return_code=0
  Result.Defeat, return_code=0
  Result.Victory, return_code=0
summary: 1 Victory / 2 Defeat / 0 Tie
```

Revised tactic command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 3 --run-root runs --run-name 20260623_eval_power_ab_revised_tactic_v1 --policy-name revised_tactic_power_ab_v1 --army-policy rule --strategy-policy coverage-teacher --strategy-tactic-mode rule --trajectory-dir data\trajectories\power_ab_revised_tactic_army_v1 --strategy-trajectory-dir data\trajectories\power_ab_revised_tactic_strategy_v1 --record-decision-interval 16 --game-time-limit 900
```

Revised tactic result:

```text
run: runs\20260623_171134_20260623_eval_power_ab_revised_tactic_v1
eval: runs\20260623_171134_20260623_eval_power_ab_revised_tactic_v1\artifacts\eval.jsonl
summary: runs\20260623_171134_20260623_eval_power_ab_revised_tactic_v1\artifacts\summary.json
army trajectory: data\trajectories\power_ab_revised_tactic_army_v1
strategy trajectory: data\trajectories\power_ab_revised_tactic_strategy_v1
games:
  Result.Defeat, return_code=0
  Result.Defeat, return_code=0
  Result.Tie, return_code=0
summary: 0 Victory / 2 Defeat / 1 Tie
```

Diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_tactics.py data\trajectories\power_ab_revised_tactic_strategy_v1 --show-files --json-output runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_tactic_diagnostics.json
.\.venv\Scripts\python.exe scripts\diagnose_power_tactics.py data\trajectories\power_ab_no_filter_revised_strategy_v1 data\trajectories\power_ab_revised_tactic_strategy_v1 --show-files --json-output runs\20260623_power_ab_revised_diagnostics_v1\artifacts\power_ab_diagnostics.json --text-output runs\20260623_power_ab_revised_diagnostics_v1\artifacts\power_ab_diagnostics.txt
.\.venv\Scripts\python.exe scripts\diagnose_power_tactics.py data\trajectories\power_ab_no_filter_revised_strategy_v1 --json-output runs\20260623_power_ab_revised_diagnostics_v1\artifacts\no_filter_power_diagnostics.json --text-output runs\20260623_power_ab_revised_diagnostics_v1\artifacts\no_filter_power_diagnostics.txt
.\.venv\Scripts\python.exe scripts\diagnose_power_tactics.py data\trajectories\power_ab_revised_tactic_strategy_v1 --json-output runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_power_diagnostics.json --text-output runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_power_diagnostics.txt
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py data\trajectories\power_ab_no_filter_revised_strategy_v1 --kind strategy --show-files --json-output runs\20260623_power_ab_revised_diagnostics_v1\artifacts\no_filter_strategy_trajectory_diagnostics.json
.\.venv\Scripts\python.exe scripts\diagnose_trajectories.py data\trajectories\power_ab_revised_tactic_strategy_v1 --kind strategy --show-files --json-output runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_strategy_trajectory_diagnostics.json
```

Diagnostic artifacts:

```text
runs\20260623_power_ab_revised_diagnostics_v1\artifacts\power_ab_diagnostics.txt
runs\20260623_power_ab_revised_diagnostics_v1\artifacts\power_ab_diagnostics.json
runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_tactic_diagnostics.json
runs\20260623_power_ab_revised_diagnostics_v1\artifacts\no_filter_power_diagnostics.json
runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_power_diagnostics.json
runs\20260623_power_ab_revised_diagnostics_v1\artifacts\no_filter_strategy_trajectory_diagnostics.json
runs\20260623_power_ab_revised_diagnostics_v1\artifacts\revised_strategy_trajectory_diagnostics.json
```

Metadata/schema checks:

```text
No-filter strategy diagnostics:
  files=3
  rows=180
  training_rows=177
  strategy_v2 rows=180
  rows_defaulted_observation_fields=0
  results=1 Victory / 2 Defeat

Revised strategy diagnostics:
  files=3
  rows=213
  training_rows=210
  strategy_v2 rows=213
  rows_defaulted_observation_fields=0
  results=1 Tie / 2 Defeat
  rows_with_tactic_metadata=213
  rows_with_filter_metadata=213
  training_filter_change_rows=42
  opponent_ai_build=Power for all training rows
```

Robo/Gateway diagnosis:

```text
Robo:
  no-filter first TECH_ROBO:
    251.4s / 251.4s / 251.4s
  revised first TECH_ROBO:
    274.3s / 297.1s / 274.3s
  revised ready_robo:
    331.4s / 354.3s / 331.4s

The revised filter fixed the previous catastrophic first-Robo delay
(previous tactic run had a 628.6s first TECH_ROBO in one Tie), but it is still
later than no-filter coverage-teacher by roughly 23-46 game seconds.

Gateway:
  no-filter ADD_GATEWAYS:
    count=6, first=91.4s, min=91.4s, avg=110.5s
  revised ADD_GATEWAYS:
    count=5, source earliest=502.9s, file 002 first=754.3s, avg=685.7s
  revised filter changes still include:
    SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS: 7
    TECH_POWER ADD_GATEWAYS -> PRODUCE_ARMY: 2

Gateway rhythm is not fixed. The over-conversion to FORGE_UPGRADES is gone, but
early ADD_GATEWAYS is still being suppressed, mostly by SAFE_MACRO -> BOOST_WORKERS.
```

Filter/action diagnosis:

```text
Revised tactic filter changes:
  ANTI_AIR_RESPONSE FORGE_UPGRADES -> TECH_ROBO: 9
  RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 8
  SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS: 7
  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 4
  RECOVERY FORGE_UPGRADES -> PRODUCE_ARMY: 4
  ANTI_AIR_RESPONSE ADD_GATEWAYS -> TECH_ROBO: 3
  TECH_POWER BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 3
  TECH_POWER ADD_GATEWAYS -> PRODUCE_ARMY: 2
  RECOVERY TECH_ROBO -> PRODUCE_ARMY: 1
  TECH_POWER TECH_ROBO -> STAY_COURSE: 1

Counterfactual filter delta:
  ADD_GATEWAYS=-12
  BOOST_WORKERS=+7
  BUILD_STATIC_DEFENSE=-15
  FORGE_UPGRADES=-13
  PRODUCE_ARMY=+22
  STAY_COURSE=+1
  TECH_ROBO=+10

Static-defense cap is working directionally:
  no-filter BUILD_STATIC_DEFENSE=15, all under threat
  revised BUILD_STATIC_DEFENSE=9, PRODUCE_ARMY under threat=16
```

Observer/Immortal:

```text
No-filter:
  Observer none in all three files
  Immortal none in all three files

Revised:
  Defeat file 001: no Observer / no Immortal
  Defeat file 002: no Observer / no Immortal
  Tie file 003: first Observer=560.0s, first Immortal=605.7s

This is a partial improvement, but the two Defeat files still miss the Robo unit
payoff entirely.
```

Conclusion:

```text
The revised filter did not outperform no-filter:
  no-filter: 1W / 0T / 2L
  revised:   0W / 1T / 2L

It partially fixed Robo timing and reduced FORGE_UPGRADES/static-defense spam,
but did not fix early Gateway rhythm and did not reliably produce Observer /
Immortal in losing games.

Do not promote revised tactic filter.
Do not collect tactic-aware training data from it.
Next work should stay offline and inspect SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS
and Robo unit production after ready_robo before another guarded comparison.
```

## 2026-06-24 Strategy Outcome Plan

Scope:

```text
Prepare the next development window.
No SC2 launch.
No PPO.
No training.
No tactic-aware data collection.
Do not change default rule/no-op behavior.
Keep --strategy-policy rule / --strategy-tactic-mode off as the default.
Do not add tactic metadata to observation schema.
```

Docs updated:

```text
doc\STRATEGY_OUTCOME_PLAN.md
doc\CODEX.md
doc\README.md
doc\STATE.md
doc\TACTIC_POOL_PLAN.md
```

Strategic conclusion:

```text
The current bottleneck is not model size, PPO, or more training.
The missing loop is action-to-outcome evidence:
  ADD_GATEWAYS -> did Gateway count or pending Gateway increase?
  TECH_ROBO -> did pending/ready Robo appear?
  ready_robo -> did Observer/Immortal appear?
  PRODUCE_ARMY -> did army_count increase?
  BUILD_STATIC_DEFENSE -> did threat clear or persist?
```

Latest Power A/B remains the key evidence:

```text
no-filter coverage-teacher:
  data\trajectories\power_ab_no_filter_revised_strategy_v1
  result: 1 Victory / 2 Defeat / 0 Tie

revised Power tactic filter:
  data\trajectories\power_ab_revised_tactic_strategy_v1
  result: 0 Victory / 2 Defeat / 1 Tie

Conclusion:
  revised filter partially improved Robo delay but still lost Gateway rhythm and
  did not reliably produce Observer/Immortal payoff.
```

Next implementation target:

```text
rl\strategy_outcome_diagnostics.py
scripts\diagnose_strategy_outcomes.py
tests\test_strategy_outcome_diagnostics.py
```

First diagnostic inputs:

```text
data\trajectories\power_ab_no_filter_revised_strategy_v1
data\trajectories\power_ab_revised_tactic_strategy_v1
```

Required output:

```text
Human-readable report.
JSON report.
Per-file result summaries.
Per-action lookahead windows: +30s, +60s, +90s, +120s.
Outcome metrics for Gateway, Robo, Observer/Immortal, Forge/upgrade, static
defense, PRODUCE_ARMY, threat persistence/clearance, worker saturation, idle
Gateway/Robo, and minerals/vespene bank.
```

Recommended design shift:

```text
Make tactic filtering guardrail-first:
  default pass-through
  block only repeated / pending-capped / clearly harmful actions
  avoid broad action rewrites
  preserve early Gateway rhythm
  preserve one timely Robo
  diagnose or explicitly bias Robo unit payoff only behind opt-in behavior
  keep static-defense repeat cap but validate threat fallback with outcome data
```

Training roadmap:

```text
Do not train direct observation -> strategy_action again yet.
Later, prefer an action critic/veto model:
  observation + proposed_action -> likely landed / delayed / harmful / blocked

Only after outcome diagnostics and guarded comparisons are stable should
tactic-aware imitation be considered.
PPO remains much later.
```

## 2026-06-24 Documentation Reorganization

Scope:

```text
Standardize the strategy outcome plan document.
Move project Markdown documentation into a dedicated doc directory.
Do not change runtime code.
Do not launch SC2.
Do not train.
Do not touch PPO/RL mainline behavior.
```

Documentation layout:

```text
doc\CODEX.md
doc\README.md
doc\STATE.md
doc\STRATEGY_OUTCOME_PLAN.md
doc\TACTIC_POOL_PLAN.md
doc\archive\STRATEGY_EXPANSION_PLAN.md
```

Changes:

```text
doc\STRATEGY_OUTCOME_PLAN.md
  Rewritten as a standard development plan with purpose, scope, constraints,
  phases, required metrics, verification, and promotion criteria.

doc\CODEX.md
  Updated read-first paths to doc\...

doc\README.md
  Updated project structure and documentation map to the doc directory layout.

doc\TACTIC_POOL_PLAN.md
  Updated documentation references and next-step pointer to doc\STRATEGY_OUTCOME_PLAN.md.

doc\archive\STRATEGY_EXPANSION_PLAN.md
  Updated current-plan pointer to doc\STRATEGY_OUTCOME_PLAN.md.

doc\STATE.md
  Updated current documentation map and recorded this reorganization.
```

Current next implementation target remains:

```text
rl\strategy_outcome_diagnostics.py
scripts\diagnose_strategy_outcomes.py
tests\test_strategy_outcome_diagnostics.py
```

## 2026-06-24 Documentation Archival Pass

范围：

```text
整理 doc 目录下的文档层级。
不修改运行代码。
不启动 SC2。
不训练。
不触碰 PPO/RL 主线行为。
```

当前文档层级：

```text
doc\CODEX.md
doc\README.md
doc\STATE.md
doc\STRATEGY_OUTCOME_PLAN.md
doc\TACTIC_POOL_PLAN.md
doc\archive\STRATEGY_EXPANSION_PLAN.md
```

整理结果：

```text
doc\STRATEGY_OUTCOME_PLAN.md
  当前主开发计划。

doc\TACTIC_POOL_PLAN.md
  保留为 AIBuild / TacticSpec 背景计划，并在顶部标注当前执行计划以
  doc\STRATEGY_OUTCOME_PLAN.md 为准。

doc\archive\STRATEGY_EXPANSION_PLAN.md
  归档为历史路线文档，不再作为新窗口必读入口。

doc\CODEX.md / doc\README.md / doc\STATE.md
  更新文档地图和读取顺序，当前路径均指向 doc 或 doc\archive。
```

当前主线不变：

```text
StrategyOutcomeDiagnostics 已实现；下一步使用 outcome 证据设计 guardrail-first tactic filter。
默认 --strategy-policy rule / --strategy-tactic-mode off 保持不变。
不采 tactic-aware training data。
不进入 PPO。
```

## 2026-06-24 Strategy Outcome Diagnostics Implementation

范围：

```text
实现离线 StrategyOutcomeDiagnostics。
不启动 SC2。
不训练。
不实现或修改 PPO。
不修改 runtime strategy/tactic 行为。
不修改 observation schema。
默认 --strategy-policy rule / --strategy-tactic-mode off 保持不变。
```

新增文件：

```text
rl\strategy_outcome_diagnostics.py
scripts\diagnose_strategy_outcomes.py
tests\test_strategy_outcome_diagnostics.py
```

实现内容：

```text
diagnose_strategy_outcomes():
  读取 strategy trajectory JSONL。
  支持输入文件或目录。
  对每个非 terminal strategy row 计算 +30/+60/+90/+120s outcome。
  terminal row 只作为后续状态参与 lookahead，不作为 action sample。

输出：
  per-source summary
  per-file summary
  per-action timing and outcome windows
  per-action + lookahead outcome metrics/events
  tactic filter-change outcome summaries
  JSON via dataclasses.asdict
  human-readable report via scripts\diagnose_strategy_outcomes.py
```

覆盖指标：

```text
Gateway:
  ready_gateway_delta
  pending_gateway_seen
  first_pending_gateway_after_action
  first_ready_gateway_delta_time

Robo / Robo payoff:
  pending_robo_seen
  ready_robo_seen
  observer_delta
  immortal_delta
  observer_immortal_delta
  robo_idle_after

Production / threat / economy:
  army_count_delta
  static_defense_delta
  threat_persisted
  threat_cleared
  forge/upgrades
  base/pending Nexus
  worker_delta
  worker_saturation_after
  minerals_after
  vespene_after
  gateway_idle_after
```

Validation：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_outcome_diagnostics.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results：

```text
focused pytest: 4 passed
full pytest: 154 passed in 2.51s
check_env: all OK
SC2 launched: no
hidden-window guard run: no, because no SC2 launch occurred
training/PPO: no
```

Offline diagnosis command：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_strategy_outcomes.py data\trajectories\power_ab_no_filter_revised_strategy_v1 data\trajectories\power_ab_revised_tactic_strategy_v1 --show-files --json-output runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.json --text-output runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.txt
```

Artifacts：

```text
runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.txt
runs\20260624_strategy_outcome_power_ab_v1\artifacts\strategy_outcomes.json
```

Dataset summary：

```text
inputs:
  data\trajectories\power_ab_no_filter_revised_strategy_v1
  data\trajectories\power_ab_revised_tactic_strategy_v1

files=6
rows=393
training_rows=387
results:
  Victory=1
  Tie=1
  Defeat=4
```

Source comparison：

```text
no-filter:
  files=3
  rows=180
  training_rows=177
  results=1 Victory / 2 Defeat
  filter_change_rows=0
  first ADD_GATEWAYS=91.4s
  first TECH_ROBO=251.4s
  action counts:
    STAY_COURSE=100
    ADD_GATEWAYS=6
    TECH_ROBO=27
    FORGE_UPGRADES=2
    BUILD_STATIC_DEFENSE=15
    PRODUCE_ARMY=21
    BOOST_WORKERS=6

revised tactic:
  files=3
  rows=213
  training_rows=210
  results=1 Tie / 2 Defeat
  filter_change_rows=42
  first actual ADD_GATEWAYS=502.9s at source level
  first TECH_ROBO=274.3s
  action counts:
    STAY_COURSE=125
    EXPAND=2
    ADD_GATEWAYS=5
    TECH_ROBO=15
    FORGE_UPGRADES=4
    BUILD_STATIC_DEFENSE=9
    PRODUCE_ARMY=34
    BOOST_WORKERS=16
```

Key outcome findings：

```text
SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS:
  count=7
  early_before_240=7
  +120s average ready_gateway_delta=3.1
  +120s average army_count_delta=5.9
  Interpretation:
    The filter removes early strategy-level ADD_GATEWAYS labels. The rule
    baseline still builds some Gateways underneath, so this is not a total
    production block, but it explains why revised actual ADD_GATEWAYS is late.

TECH_POWER ADD_GATEWAYS -> PRODUCE_ARMY:
  count=2
  +30s army_count_delta=-1.0
  +60s army_count_delta=-1.0
  +120s army_count_delta=9.0
  +120s observer_immortal_delta=2.0
  Interpretation:
    Not an immediate army payoff, but both small samples show later Robo-unit
    payoff. Too small to promote.

TECH_POWER BUILD_STATIC_DEFENSE -> PRODUCE_ARMY:
  count=3
  threat_cleared=3/3 by +30s
  Interpretation:
    This specific suppression looks acceptable in the current sample.

RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY:
  count=8
  +60s threat_cleared=7/8, threat_persisted=1/8
  +120s threat_cleared=8/8
  Interpretation:
    Mixed early, but no persistent threat by +120s in this sample.

ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY:
  count=4
  +60s threat_persisted=4/4
  +120s threat_cleared=4/4
  Interpretation:
    Early fallback looks riskier here; do not generalize static-defense
    suppression without tactic-specific guardrails.
```

Robo payoff finding：

```text
revised Defeat file 001:
  ready_robo=331.4s
  observer=none
  immortal=none
  post-ready rows repeatedly show robo_idle_count=1.

revised Defeat file 002:
  ready_robo=354.3s
  observer=none
  immortal=none
  post-ready rows repeatedly show robo_idle_count=1.

revised Tie file 003:
  ready_robo=331.4s
  observer=560.0s
  immortal=605.7s

Interpretation:
  The revised filter fixed catastrophic first-Robo delay, but it did not ensure
  Robo unit payoff. StrategyExecutor trains Observer/Immortal from Robo inside
  TECH_ROBO execution; after Robo becomes ready, revised Defeat files mostly emit
  STAY_COURSE / PRODUCE_ARMY / FORGE_UPGRADES rather than TECH_ROBO. PRODUCE_ARMY
  delegates to Gateway production only, so Robo stays idle.
```

Current next step：

```text
Do not promote revised tactic filter.
Do not collect tactic-aware training data.
Do not train or start PPO.
Use outcome diagnostics to design a guardrail-first change:
  - preserve early Gateway rhythm without broad ADD_GATEWAYS rewrites
  - preserve one timely Robo
  - add a narrow opt-in way to trigger Observer/Immortal production after ready_robo
  - keep tactic metadata out of observation schema
  - keep all behavior explicit opt-in
```

## 2026-06-24 SAFE_MACRO Guardrail And Ready-Robo Production Tests

Scope：

```text
Offline-only focused guardrail fix.
No SC2 launch.
No training.
No PPO.
No observation schema change.
Default --strategy-policy rule / --strategy-tactic-mode off remains unchanged.
```

Code changes：

```text
rl\tactics.py
  SAFE_MACRO now preserves early ADD_GATEWAYS when game_time < 240s and
  pending_gateways == 1.
  Early SAFE_MACRO ADD_GATEWAYS is still capped at pending_gateways >= 2.
  minerals < 100 still blocks Gateway spending through the existing reserve.

bot\managers\strategy_executor.py
  PRODUCE_ARMY now also checks ready Robotics Facilities:
    - train Observer first when none exists and affordable
    - otherwise train Immortal when affordable and supply allows
  It still calls the existing _train_army gateway production hook.
```

Tests added：

```text
tests\test_tactics.py
  SAFE_MACRO preserves early ADD_GATEWAYS with one pending Gateway.
  SAFE_MACRO caps early ADD_GATEWAYS at two pending Gateways.

tests\test_tactic_strategy_policy.py
  Power early-game tactic-aware wrapper keeps SAFE_MACRO ADD_GATEWAYS instead
  of rewriting it to BOOST_WORKERS.

tests\test_strategy_policy.py
  PRODUCE_ARMY trains Observer from ready Robo.
  PRODUCE_ARMY trains Immortal from ready Robo when Observer already exists.
  Both cases still delegate to gateway _train_army.
```

Validation：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py tests\test_strategy_policy.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results：

```text
focused pytest: 36 passed in 1.20s
full pytest: 159 passed in 9.82s
check_env: all OK
SC2 launched: no
hidden-window guard run: no, because no SC2 launch occurred
training/PPO: no
```

Next：

```text
Do not promote revised tactic filter yet.
Do not collect tactic-aware training data from this change.
Before any game recheck, use fresh dirs and hidden-window guard, then run only
scripts\evaluate.py / scripts\safe_launch.py.
After any guarded A/B, regenerate tactic, power tactic, and strategy outcome
diagnostics to verify early Gateway timing and ready-Robo Observer/Immortal
payoff.
```

## 2026-06-24 Guarded Power-Only A/B After Guardrail

Scope：

```text
Guarded SC2 evaluation via scripts\evaluate.py / scripts\safe_launch.py.
Fresh strategy trajectory dirs.
No army trajectory / imitation data collection.
No training.
No PPO.
No observation schema change.
```

Hidden-window guard：

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Result：

```text
guard pid: 26628
```

No-filter coverage-teacher command：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --run-root runs --run-name 20260624_power_ab_guardrail_no_filter_v1 --tag guarded-power-ab --tag no-filter --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 3 --strategy-policy coverage-teacher --strategy-tactic-mode off --strategy-trajectory-dir data\trajectories\power_ab_guardrail_no_filter_strategy_v1
```

No-filter result：

```text
run: runs\20260624_103759_20260624_power_ab_guardrail_no_filter_v1
strategy trajectory: data\trajectories\power_ab_guardrail_no_filter_strategy_v1
games: 3
results: 1 Victory / 2 Defeat
return_code: 3/3 zero
```

Tactic guardrail command：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --run-root runs --run-name 20260624_power_ab_guardrail_tactic_v1 --tag guarded-power-ab --tag tactic-guardrail --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 3 --strategy-policy coverage-teacher --strategy-tactic-mode rule --strategy-trajectory-dir data\trajectories\power_ab_guardrail_tactic_strategy_v1
```

Tactic guardrail result：

```text
run: runs\20260624_104104_20260624_power_ab_guardrail_tactic_v1
strategy trajectory: data\trajectories\power_ab_guardrail_tactic_strategy_v1
games: 3
results: 1 Victory / 2 Defeat
return_code: 3/3 zero
filter_change_rows: 45
```

Outcome diagnostics command：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_strategy_outcomes.py data\trajectories\power_ab_guardrail_no_filter_strategy_v1 data\trajectories\power_ab_guardrail_tactic_strategy_v1 --show-files --json-output runs\20260624_strategy_outcome_power_ab_guardrail_v1\artifacts\strategy_outcomes.json --text-output runs\20260624_strategy_outcome_power_ab_guardrail_v1\artifacts\strategy_outcomes.txt
```

Outcome artifacts：

```text
runs\20260624_strategy_outcome_power_ab_guardrail_v1\artifacts\strategy_outcomes.txt
runs\20260624_strategy_outcome_power_ab_guardrail_v1\artifacts\strategy_outcomes.json
```

Dataset summary：

```text
files=6
rows=361
training_rows=355
results=2 Victory / 4 Defeat
```

Source comparison：

```text
no-filter:
  files=3
  rows=193
  training_rows=190
  results=1 Victory / 2 Defeat
  first ADD_GATEWAYS=91.4s
  first TECH_ROBO=240.0s
  actions:
    STAY_COURSE=110
    ADD_GATEWAYS=8
    TECH_ROBO=36
    BUILD_STATIC_DEFENSE=18
    PRODUCE_ARMY=13
    BOOST_WORKERS=5

tactic guardrail:
  files=3
  rows=168
  training_rows=165
  results=1 Victory / 2 Defeat
  filter_change_rows=45
  first ADD_GATEWAYS=137.1s
  first TECH_ROBO=240.0s
  actions:
    STAY_COURSE=95
    EXPAND=2
    ADD_GATEWAYS=5
    TECH_ROBO=4
    FORGE_UPGRADES=3
    BUILD_STATIC_DEFENSE=5
    PRODUCE_ARMY=37
    BOOST_WORKERS=14
```

Outcome findings：

```text
Early Gateway:
  improved vs previous revised tactic run.
  Previous revised source-level first ADD_GATEWAYS was 502.9s.
  Guardrail run source-level first ADD_GATEWAYS is 137.1s.
  Still later than no-filter 91.4s, and one tactic Defeat file delayed
  ADD_GATEWAYS to 571.4s.

SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS:
  count=3, all before 240s.
  This is reduced from the previous count=7, but not eliminated.

Robo timing:
  source-level first TECH_ROBO is tied at 240.0s.
  tactic guardrail emits far fewer TECH_ROBO labels: 4 vs no-filter 36.

Robo payoff:
  tactic guardrail produced Observer in all 3 files:
    525.7s / 331.4s / 445.7s.
  no tactic guardrail file produced Immortal.
  ready-Robo production hook improved Observer payoff but did not solve Immortal
  payoff.

PRODUCE_ARMY:
  count=37 in tactic guardrail vs 13 no-filter.
  PRODUCE_ARMY windows show robo_unit_produced=7 by +30s and 13 by +60s.

Static-defense suppression:
  TECH_POWER BUILD_STATIC_DEFENSE -> PRODUCE_ARMY:
    count=7, +60s threat_cleared=6/7 and threat_persisted=1/7.
  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY:
    count=6, +60s threat_cleared=3/6 and threat_persisted=3/6.
  Anti-air/static fallback remains risky.
```

Conclusion：

```text
Do not promote tactic guardrail.
Do not collect tactic-aware training data from this run.
Do not train or start PPO.
The guardrail fixed the worst early-Gateway regression and Observer payoff, but
the A/B result did not beat no-filter and Immortal/static-defense payoff remains
unresolved.
```

## 2026-06-24 Offline Third-Game Timeline And First-Immortal Bias

Scope：

```text
Offline-only diagnosis and focused opt-in tactic filter fix.
No SC2 launch.
No hidden-window guard required because no game was started.
No training.
No PPO.
No observation schema change.
Default --strategy-policy rule / --strategy-tactic-mode off unchanged.
```

Changed diagnostics：

```text
rl\tactic_diagnostics.py
  Added per-file filter_timeline events for rows with tactic filter metadata.
  Fields include line, step, game_time, tactic, original_action,
  selected_action, changed, minerals, vespene, supply_left, pending/ready
  gateways, pending/ready Robo, pending/ready static defense,
  base_under_threat, gateway_idle_count, robo_idle_count.

scripts\diagnose_tactics.py
  Added --text-output and --show-filter-timeline.

rl\strategy_outcome_diagnostics.py
  Added RoboPayoffSummary per file.
  It classifies Observer/Immortal payoff after ready_robo and distinguishes:
    action_not_triggered
    robo_not_idle
    action_not_triggered_while_idle
    resource_or_supply_blocked
    not_produced_after_affordable_action
```

Artifacts：

```text
runs\20260624_guardrail_tactic_timeline_v1\artifacts\tactic_timeline.txt
runs\20260624_guardrail_tactic_timeline_v1\artifacts\tactic_timeline.json
runs\20260624_guardrail_tactic_robo_outcomes_v1\artifacts\strategy_outcomes.txt
runs\20260624_guardrail_tactic_robo_outcomes_v1\artifacts\strategy_outcomes.json
```

Third tactic guardrail game diagnosis：

```text
file:
  data\trajectories\power_ab_guardrail_tactic_strategy_v1\
  20260624_104237_AcropolisLE_Hard_Terran_Power_003.jsonl

The selected ADD_GATEWAYS label first appears at 571.4s, but the filter timeline
shows this is not continuous early Gateway suppression:

  102.9s SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS
    minerals=290
    pending_gateways=2
    ready_gateways=0
    base_under_threat=0
    gateway_idle_count=0

  274.3s ready_gateways=4

  571.4s RECOVERY ADD_GATEWAYS -> ADD_GATEWAYS
    minerals=175
    pending_gateways=0
    ready_gateways=0
    base_under_threat=0

Conclusion:
  The 571.4s label is a late rebuild after production was damaged, not proof
  that SAFE_MACRO kept suppressing early Gateway. The early suppression row
  happened at the pending_gateway cap.
```

Robo payoff diagnosis：

```text
Tactic guardrail file 001:
  ready_robo=502.9
  observer=525.7
  immortal=none
  observer_status=produced_after_ready
  immortal_blocker=resource_or_supply_blocked
  blocker detail: minerals=0 rows, vespene=0 rows, supply=3 rows

Tactic guardrail file 002:
  ready_robo=297.1
  observer=331.4
  immortal=none
  observer_status=produced_after_ready
  immortal_blocker=resource_or_supply_blocked
  blocker detail: minerals=9 rows, vespene=0 rows, supply=1 row

Tactic guardrail file 003:
  ready_robo=365.7
  observer=445.7
  immortal=none
  observer_status=produced_after_ready
  immortal_blocker=resource_or_supply_blocked
  blocker detail: minerals=6 rows, vespene=0 rows, supply=0 rows
```

Runtime fix：

```text
rl\tactics.py
  Added first-Immortal bias inside explicit tactic filtering only.

Condition:
  tactic in ROBO_TIMING / TECH_POWER / ANTI_AIR_RESPONSE
  ready_robo > 0
  observers > 0
  immortals == 0
  base_under_threat == 0
  vespene >= 100

Behavior:
  if minerals >= 275 and supply_left >= 4:
    choose PRODUCE_ARMY
  elif minerals >= 100 and supply_left >= 4 and proposed action could spend
  resources before the first Immortal:
    choose STAY_COURSE
  elif minerals >= 275 but supply_left < 4 and proposed action could spend:
    choose STAY_COURSE

This does not affect default rule/off runtime and does not add tactic metadata
to the observation schema.
```

Tests：

```text
tests\test_tactic_diagnostics.py
  filter timeline resources and formatting.

tests\test_strategy_outcome_diagnostics.py
  Robo payoff blocker classification.

tests\test_tactics.py
  first-Immortal bias, bank behavior, and static-defense-under-threat guard.

tests\test_tactic_strategy_policy.py
  opt-in tactic-aware wrapper routes ready-Robo state to first Immortal bias.
```

Validation：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tactic_diagnostics.py tests\test_strategy_outcome_diagnostics.py tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
```

Results：

```text
focused pytest: 35 passed in 1.22s
full pytest: 165 passed in 2.50s
check_env: all OK
SC2 launched: no
training/PPO: no
```

Next：

```text
Consider a fresh-dir guarded Power-only A/B for the first-Immortal bias.
Before starting SC2, run hidden-window guard.
Use only scripts\evaluate.py / scripts\safe_launch.py.
Do not train or collect tactic-aware large data unless a guarded comparison
beats no-filter coverage-teacher.
```

## 2026-06-24 First-Immortal Bias Guarded Power-Only A/B

Scope：

```text
Guarded SC2 evaluation via scripts\evaluate.py / scripts\safe_launch.py.
Fresh strategy trajectory dirs.
No training.
No PPO.
No tactic-aware large data collection.
No observation schema change.
Default --strategy-policy rule / --strategy-tactic-mode off unchanged.
```

Hidden-window guard：

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Result：

```text
guard pid: 26628
```

No-filter coverage-teacher command：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --run-root runs --run-name 20260624_power_ab_immortal_bias_no_filter_v1 --tag guarded-power-ab --tag immortal-bias --tag no-filter --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 3 --strategy-policy coverage-teacher --strategy-tactic-mode off --strategy-trajectory-dir data\trajectories\power_ab_immortal_bias_no_filter_strategy_v1
```

No-filter result：

```text
run: runs\20260624_111950_20260624_power_ab_immortal_bias_no_filter_v1
strategy trajectory: data\trajectories\power_ab_immortal_bias_no_filter_strategy_v1
games: 3
results: 0 Victory / 3 Defeat
avg_duration_seconds: 45.49
return_code: 3/3 zero
```

Tactic-rule first-Immortal bias command：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --run-root runs --run-name 20260624_power_ab_immortal_bias_tactic_v1 --tag guarded-power-ab --tag immortal-bias --tag tactic-rule --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 3 --strategy-policy coverage-teacher --strategy-tactic-mode rule --strategy-trajectory-dir data\trajectories\power_ab_immortal_bias_tactic_strategy_v1
```

Tactic-rule result：

```text
run: runs\20260624_112225_20260624_power_ab_immortal_bias_tactic_v1
strategy trajectory: data\trajectories\power_ab_immortal_bias_tactic_strategy_v1
games: 3
results: 0 Victory / 3 Defeat
avg_duration_seconds: 59.72
return_code: 3/3 zero
filter_change_rows: 76
```

Diagnostics：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_tactics.py data\trajectories\power_ab_immortal_bias_tactic_strategy_v1 --show-files --show-filter-timeline --json-output runs\20260624_power_ab_immortal_bias_tactic_timeline_v1\artifacts\tactic_timeline.json --text-output runs\20260624_power_ab_immortal_bias_tactic_timeline_v1\artifacts\tactic_timeline.txt

.\.venv\Scripts\python.exe scripts\diagnose_strategy_outcomes.py data\trajectories\power_ab_immortal_bias_no_filter_strategy_v1 data\trajectories\power_ab_immortal_bias_tactic_strategy_v1 --show-files --json-output runs\20260624_strategy_outcome_power_ab_immortal_bias_v1\artifacts\strategy_outcomes.json --text-output runs\20260624_strategy_outcome_power_ab_immortal_bias_v1\artifacts\strategy_outcomes.txt
```

Diagnostic artifacts：

```text
runs\20260624_power_ab_immortal_bias_tactic_timeline_v1\artifacts\tactic_timeline.txt
runs\20260624_power_ab_immortal_bias_tactic_timeline_v1\artifacts\tactic_timeline.json
runs\20260624_strategy_outcome_power_ab_immortal_bias_v1\artifacts\strategy_outcomes.txt
runs\20260624_strategy_outcome_power_ab_immortal_bias_v1\artifacts\strategy_outcomes.json
```

Source comparison：

```text
no-filter:
  files=3
  rows=183
  training_rows=180
  results=3 Defeat
  first ADD_GATEWAYS=91.4s
  first TECH_ROBO=240.0s
  first BUILD_STATIC_DEFENSE=262.9s
  actions:
    STAY_COURSE=88
    ADD_GATEWAYS=5
    TECH_ROBO=43
    BUILD_STATIC_DEFENSE=19
    PRODUCE_ARMY=9
    BOOST_WORKERS=14

tactic-rule first-Immortal bias:
  files=3
  rows=221
  training_rows=218
  results=3 Defeat
  filter_change_rows=76
  first ADD_GATEWAYS=422.9s
  first TECH_ROBO=240.0s
  first BUILD_STATIC_DEFENSE=514.3s
  actions:
    STAY_COURSE=108
    EXPAND=3
    ADD_GATEWAYS=6
    TECH_ROBO=8
    FORGE_UPGRADES=4
    BUILD_STATIC_DEFENSE=11
    PRODUCE_ARMY=59
    BOOST_WORKERS=19
```

Robo payoff：

```text
no-filter:
  file 001: ready_robo=331.4 observer=434.3 immortal=none
            immortal_blocker=resource_or_supply_blocked
  file 002: ready_robo=none observer=none immortal=none
  file 003: ready_robo=365.7 observer=none immortal=none
            observer_blocked; immortal_blocker=action_not_triggered

tactic-rule first-Immortal bias:
  file 001: ready_robo=331.4 observer=434.3 immortal=582.9
            observer_status=produced_after_ready
            immortal_status=produced_after_ready
  file 002: ready_robo=502.9 observer=525.7 immortal=none
            immortal_blocker=not_produced_after_affordable_action
  file 003: ready_robo=297.1 observer=331.4 immortal=none
            immortal_blocker=resource_or_supply_blocked

Conclusion:
  First-Immortal bias improved Immortal payoff in 1/3 tactic files and did not
  regress Observer in tactic files. It is not enough to improve match results.
```

Gateway / static-defense findings：

```text
Gateway:
  no-filter source-level first ADD_GATEWAYS=91.4s
  tactic-rule source-level first ADD_GATEWAYS=422.9s
  per-file tactic first ADD_GATEWAYS:
    617.1s / 422.9s / 674.3s
  This is a serious regression and blocks promotion.

Static defense:
  no-filter selected BUILD_STATIC_DEFENSE count=19
  tactic-rule selected BUILD_STATIC_DEFENSE count=11
  tactic-rule filter changes:
    TECH_POWER BUILD_STATIC_DEFENSE -> PRODUCE_ARMY count=13
      +60s threat_persisted=7/13
    ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY count=6
      +60s threat_persisted=5/6
    RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY count=13
      +60s threat_persisted=4/13
  Static-defense suppression remains unsafe.
```

Conclusion：

```text
Do not promote first-Immortal bias tactic filter.
Do not collect tactic-aware training data from this run.
Do not train action-outcome / veto / imitation from this A/B yet.
Do not start PPO.

Evidence supports only the narrow claim that first-Immortal bias can make
Immortal land in at least one file. It also introduced or exposed major Gateway
timing and static-defense regressions. Next offline work should focus on:

  1. Gateway precedence / early-to-midgame ADD_GATEWAYS retention.
  2. Static-defense retention under TECH_POWER / ANTI_AIR_RESPONSE / RECOVERY
     when base_under_threat is active.
  3. A safer Immortal bias that does not rewrite too many ADD_GATEWAYS and
     static-defense labels.
```

## 2026-06-24 Training / PPO / Tactic-Aware Data Policy Clarification

The previous "do not train / do not PPO / do not collect tactic-aware data"
language was a stage-specific guardrail for the unsafe revised tactic-filter
experiments. It is not a permanent project ban.

Current policy：

```text
Allowed when evidence supports it:
  small fresh-dir tactic-aware data collection
  action-outcome / action-critic training
  veto / safer-fallback model training
  tactic-aware strategy imitation experiments

Still gated:
  do not promote tactic-aware runtime by default
  do not weaken the default rule baseline
  keep default --strategy-policy rule / --strategy-tactic-mode off
  keep tactic metadata out of the observation schema unless a future explicit
  model design requires an intentional schema version bump
  only use scripts\evaluate.py / scripts\safe_launch.py for SC2 launches
  run hidden-window guard before any SC2 launch

PPO:
  not forbidden, but should wait until SC2 env, reward design, baseline
  comparisons, safe-launch boundaries, and rollback/eval protocol are clear.
```

Practical next step is a fresh-dir guarded Power-only A/B for the revised
guardrail-first tactic filter. If that evidence is positive, small tactic-aware
data collection or outcome/veto training is an acceptable next branch.

## 2026-06-24 Offline Gateway / Static-Defense Retention

Offline runtime/test fix only; no SC2 launch, no hidden-window guard needed in
this step, no training, no PPO, no tactic-aware data collection.

Changed files:

```text
rl\tactics.py
tests\test_tactics.py
tests\test_tactic_strategy_policy.py
doc\CODEX.md
doc\README.md
doc\STATE.md
doc\STRATEGY_OUTCOME_PLAN.md
doc\TACTIC_POOL_PLAN.md
```

Behavior changes are restricted to explicit opt-in tactic filtering:

```text
Default remains:
  --strategy-policy rule
  --strategy-tactic-mode off

TECH_POWER static-defense retention:
  If original action is BUILD_STATIC_DEFENSE, base_under_threat is active, and
  pending_static_defense is below the tactic cap:
    minerals >= 100 -> keep BUILD_STATIC_DEFENSE
    minerals < 100  -> STAY_COURSE to bank for Shield Battery / static defense
  If pending_static_defense is already capped, keep the previous fallback to
  PRODUCE_ARMY.

First-Immortal bias Gateway precedence:
  If first-Immortal bias is active but original action is ADD_GATEWAYS and
  ready_gateways + pending_gateways < own_bases * 4, do not rewrite it to
  PRODUCE_ARMY. This aligns the tactic filter with StrategyExecutor's
  DEFAULT_GATEWAYS_PER_BASE = 4 target.
```

New tests:

```text
test_tech_power_preserves_static_defense_under_active_threat
test_tech_power_banks_for_static_defense_under_threat_when_minerals_short
test_tech_power_preserves_underbuilt_gateway_before_first_immortal
test_tactic_aware_power_policy_preserves_underbuilt_gateway
test_tactic_aware_power_policy_preserves_static_defense_under_threat
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

Conclusion:

```text
This fixes the offline regressions that blocked the first-Immortal bias:
  Gateway precedence before first Immortal
  Static-defense retention under active threat

Do not promote tactic-rule yet.
Next evidence step: fresh-dir guarded Power-only A/B, then tactic timeline and
enhanced strategy outcome diagnostics. Only consider tactic-aware data or
outcome/veto training if that guarded comparison is positive and reversible.
```

## 2026-06-24 Guardrail Retention Power A/B

SC2 launch rules were followed:

```text
hidden-window guard:
  .\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
  pid=26628

SC2 entrypoint:
  scripts\evaluate.py only

No training.
No PPO.
No tactic-aware data collection.
Fresh strategy trajectory dirs were used.
```

Commands:

```text
no-filter:
  .\.venv\Scripts\python.exe scripts\evaluate.py --run-root runs --run-name 20260624_power_ab_guardrail_retention_no_filter_v1 --tag guarded-power-ab --tag guardrail-retention --tag no-filter --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 3 --strategy-policy coverage-teacher --strategy-tactic-mode off --strategy-trajectory-dir data\trajectories\power_ab_guardrail_retention_no_filter_strategy_v1

tactic-rule:
  .\.venv\Scripts\python.exe scripts\evaluate.py --run-root runs --run-name 20260624_power_ab_guardrail_retention_tactic_v1 --tag guarded-power-ab --tag guardrail-retention --tag tactic-rule --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 3 --strategy-policy coverage-teacher --strategy-tactic-mode rule --strategy-trajectory-dir data\trajectories\power_ab_guardrail_retention_tactic_strategy_v1
```

Results:

```text
no-filter coverage-teacher:
  run: runs\20260624_114236_20260624_power_ab_guardrail_retention_no_filter_v1
  strategy trajectory: data\trajectories\power_ab_guardrail_retention_no_filter_strategy_v1
  result: 2 Victory / 1 Defeat
  return_code=0 for all games

tactic-rule guardrail-retention:
  run: runs\20260624_114514_20260624_power_ab_guardrail_retention_tactic_v1
  strategy trajectory: data\trajectories\power_ab_guardrail_retention_tactic_strategy_v1
  result: 0 Victory / 3 Defeat
  return_code=0 for all games
```

Diagnostics:

```text
tactic timeline:
  runs\20260624_power_ab_guardrail_retention_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_guardrail_retention_tactic_timeline_v1\artifacts\tactic_timeline.json

strategy outcomes:
  runs\20260624_strategy_outcome_power_ab_guardrail_retention_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_guardrail_retention_v1\artifacts\strategy_outcomes.json
```

Key findings:

```text
Source comparison:
  no-filter: files=3 rows=172 training_rows=169 results=2W/1L
    actions: TECH_ROBO=36, ADD_GATEWAYS=15, PRODUCE_ARMY=16
    first TECH_ROBO=251.4s, first ADD_GATEWAYS=91.4s

  tactic-rule: files=3 rows=175 training_rows=172 results=0W/3L
    filter_change_rows=38
    actions: TECH_ROBO=3, ADD_GATEWAYS=4, PRODUCE_ARMY=34, STAY_COURSE=95
    first TECH_ROBO=274.3s, first ADD_GATEWAYS=171.4s

Largest tactic filter changes:
  RECOVERY TECH_ROBO -> PRODUCE_ARMY: 9
  RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 8
  TECH_POWER TECH_ROBO -> STAY_COURSE: 7
  SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS: 4
  TECH_POWER BUILD_STATIC_DEFENSE -> STAY_COURSE: 4

Robo payoff:
  no-filter Victory file 001: ready_robo=331.4 observer=388.6 immortal=none
  no-filter Defeat file 002: no ready Robo
  no-filter Victory file 003: ready_robo=331.4 observer=457.1 immortal=none

  tactic file 001: ready_robo=331.4 observer=514.3 immortal=none
  tactic file 002: ready_robo=331.4 observer=422.9 immortal=none
  tactic file 003: no ready Robo, no Observer, no Immortal

Interpretation:
  Gateway/static-defense retention did not make tactic-rule safe enough.
  The filter still suppresses too much TECH_ROBO and converts too many recovery
  tech/static-defense actions into PRODUCE_ARMY. The third tactic game missed
  the first Robo window entirely: at 251.4s/274.3s it had enough resources and
  no threat but selected STAY_COURSE, then later TECH_ROBO rows were rewritten or
  arrived when minerals were too low.
```

Conclusion:

```text
Do not promote guardrail-retention tactic filter.
Do not collect tactic-aware training data from these trajectories.
Do not train outcome/veto/imitation models from this A/B.
```

## 2026-06-24 Offline Initial-Robo Precedence Fix

Offline runtime/test fix after the negative guardrail-retention A/B. No new SC2
launch after this fix, no training, no PPO, no tactic-aware data collection.

Behavior change is restricted to explicit opt-in tactic filtering:

```text
TECH_POWER initial Robo precedence:
  If original action is STAY_COURSE and:
    has_cybernetics_core > 0
    ready_robo == 0
    pending_robo == 0
    base_under_threat == 0
    TECH_ROBO passes tactic timing/resource/pending checks
  then select TECH_ROBO.

This targets the observed failure where Power tactic states had enough resources
for the first Robo but stayed idle through the timing window.
```

New tests:

```text
test_tech_power_starts_initial_robo_from_affordable_stay_course
test_tech_power_does_not_force_initial_robo_under_threat
test_tactic_aware_power_policy_starts_initial_robo_from_stay_course
```

Validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
34 passed

.\.venv\Scripts\python.exe -m pytest -q
173 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK
```

Next:

```text
The initial-Robo precedence fix is not promoted. It needs a fresh-dir guarded
Power-only A/B against no-filter coverage-teacher, followed by tactic timeline
and enhanced strategy outcome diagnostics.
```

## 2026-06-24 Initial-Robo Power A/B

SC2 launch rules were followed:

```text
hidden-window guard:
  pid=26628

SC2 entrypoint:
  scripts\evaluate.py only

No training.
No PPO.
No tactic-aware data collection.
Fresh strategy trajectory dirs were used.
```

Results:

```text
no-filter coverage-teacher:
  run: runs\20260624_115458_20260624_power_ab_initial_robo_no_filter_v1
  strategy trajectory: data\trajectories\power_ab_initial_robo_no_filter_strategy_v1
  result: 1 Victory / 1 Tie / 1 Defeat
  return_code=0 for all games

tactic-rule initial-Robo:
  run: runs\20260624_115849_20260624_power_ab_initial_robo_tactic_v1
  strategy trajectory: data\trajectories\power_ab_initial_robo_tactic_strategy_v1
  result: 0 Victory / 1 Tie / 2 Defeat
  return_code=0 for all games
```

Diagnostics:

```text
tactic timeline:
  runs\20260624_power_ab_initial_robo_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_initial_robo_tactic_timeline_v1\artifacts\tactic_timeline.json

strategy outcomes:
  runs\20260624_strategy_outcome_power_ab_initial_robo_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_initial_robo_v1\artifacts\strategy_outcomes.json
```

Key findings:

```text
Source comparison:
  no-filter: files=3 rows=250 training_rows=247 results=1W/1T/1L
    actions: TECH_ROBO=58, ADD_GATEWAYS=18, PRODUCE_ARMY=8
    first TECH_ROBO=274.3s, first ADD_GATEWAYS=91.4s

  tactic-rule: files=3 rows=288 training_rows=285 results=0W/1T/2L
    filter_change_rows=44
    actions: TECH_ROBO=12, ADD_GATEWAYS=6, PRODUCE_ARMY=32, STAY_COURSE=201
    first TECH_ROBO=240.0s, first ADD_GATEWAYS=377.1s

Largest tactic filter changes:
  ANTI_AIR_RESPONSE FORGE_UPGRADES -> TECH_ROBO: 9
  RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 7
  TECH_POWER BUILD_STATIC_DEFENSE -> STAY_COURSE: 6
  SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS: 5
  TECH_POWER ADD_GATEWAYS -> PRODUCE_ARMY: 5

Robo payoff:
  no-filter file 001 Victory: no ready Robo
  no-filter file 002 Tie: ready_robo=651.4 observer=731.4 immortal=none
  no-filter file 003 Defeat: no ready Robo

  tactic file 001 Tie: ready_robo=297.1 observer=422.9 immortal=491.4
  tactic file 002 Defeat: ready_robo=297.1 observer=445.7 immortal=none
  tactic file 003 Defeat: ready_robo=354.3 observer=422.9 immortal=502.9

Interpretation:
  Initial-Robo precedence fixed the no-ready-Robo failure and improved Robo
  payoff. It did not improve match results. The remaining regression is
  Gateway rhythm: tactic selected first ADD_GATEWAYS at 377.1s source-level
  versus no-filter 91.4s, and selected only 6 ADD_GATEWAYS rows versus 18.
```

Conclusion:

```text
Do not promote initial-Robo tactic filter.
Do not collect tactic-aware training data from these trajectories.
Do not train outcome/veto/imitation models from this A/B.
```

## 2026-06-24 Offline Midgame Gateway Cap Fix

Offline runtime/test fix after the negative initial-Robo A/B. No new SC2 launch
after this fix, no training, no PPO, no tactic-aware data collection.

Behavior change is restricted to explicit opt-in tactic filtering:

```text
TECH_POWER midgame Gateway cap:
  If:
    base_under_threat == 0
    ready_robo > 0 or pending_robo > 0
    ready_gateways + pending_gateways < own_bases * 4
  then allow up to 2 pending Gateways for ADD_GATEWAYS.

No-Robo first-tech behavior is preserved:
  If no ready/pending Robo exists and ADD_GATEWAYS is capped at one pending
  Gateway, TECH_POWER still falls back to TECH_ROBO.
```

New tests:

```text
test_tech_power_preserves_underbuilt_gateway_with_one_pending_after_robo
test_tactic_aware_power_policy_preserves_pending_underbuilt_gateway
```

Validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
36 passed

.\.venv\Scripts\python.exe -m pytest -q
175 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK
```

Next:

```text
The midgame Gateway cap fix was not promoted from offline tests alone.
Its fresh-dir guarded Power-only A/B has now been run; see the next section.
```

## 2026-06-24 Midgame Gateway Cap Power A/B

SC2 launch rules were followed:

```text
hidden-window guard:
  pid=26628

SC2 entrypoint:
  scripts\evaluate.py only

No training.
No PPO.
No tactic-aware data collection.
Fresh strategy trajectory dirs were used.
```

Results:

```text
no-filter coverage-teacher:
  run: runs\20260624_134126_20260624_power_ab_midgame_gateway_no_filter_v1
  strategy trajectory: data\trajectories\power_ab_midgame_gateway_no_filter_strategy_v1
  result: 1 Victory / 2 Defeat
  return_code=0 for all games

tactic-rule midgame Gateway cap:
  run: runs\20260624_134449_20260624_power_ab_midgame_gateway_tactic_v1
  strategy trajectory: data\trajectories\power_ab_midgame_gateway_tactic_strategy_v1
  result: 1 Victory / 2 Defeat
  return_code=0 for all games
```

Diagnostics:

```text
tactic timeline:
  runs\20260624_power_ab_midgame_gateway_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_midgame_gateway_tactic_timeline_v1\artifacts\tactic_timeline.json

strategy outcomes:
  runs\20260624_strategy_outcome_power_ab_midgame_gateway_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_midgame_gateway_v1\artifacts\strategy_outcomes.json
```

Key findings:

```text
Source comparison:
  no-filter: files=3 rows=211 training_rows=208 results=1W/2L
    actions: TECH_ROBO=54, ADD_GATEWAYS=7, PRODUCE_ARMY=13
    first TECH_ROBO=251.4s, first ADD_GATEWAYS=91.4s

  tactic-rule: files=3 rows=178 training_rows=175 results=1W/2L
    filter_change_rows=71
    actions: TECH_ROBO=6, ADD_GATEWAYS=5, PRODUCE_ARMY=20, STAY_COURSE=118
    first TECH_ROBO=240.0s, first ADD_GATEWAYS=171.4s

Largest tactic filter changes:
  TECH_POWER TECH_ROBO -> STAY_COURSE: 28
  TECH_POWER BUILD_STATIC_DEFENSE -> STAY_COURSE: 13
  ANTI_AIR_RESPONSE TECH_ROBO -> STAY_COURSE: 8
  ANTI_AIR_RESPONSE FORGE_UPGRADES -> TECH_ROBO: 4
  SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS: 4

Robo payoff:
  tactic file 001 Defeat: no ready Robo; repeated TECH_ROBO -> STAY_COURSE
  tactic file 002 Victory: ready_robo=297.1 observer=468.6 immortal=514.3
  tactic file 003 Defeat: ready_robo=297.1 observer=320.0 immortal=none,
    immortal_blocker=not_produced_after_affordable_action

Interpretation:
  Midgame Gateway cap improved source-level first ADD_GATEWAYS versus the
  initial-Robo A/B (171.4s vs 377.1s) and matched no-filter sample result
  count. It still lags no-filter Gateway timing (91.4s), keeps too many
  filter changes, and still has Robo payoff failures.
```

Conclusion:

```text
Do not promote midgame Gateway cap tactic filter.
Do not collect tactic-aware training data from these trajectories.
Do not train outcome/veto/imitation models from this A/B.
```

## 2026-06-24 Offline First-Robo Banking Guard

Offline runtime/test fix after the mixed midgame Gateway A/B. Implementation and
unit validation did not launch SC2; the later guarded A/B is recorded below. No
training, no PPO, no tactic-aware data collection.

Behavior change is restricted to explicit opt-in tactic filtering:

```text
TECH_POWER first-Robo banking guard:
  If:
    has_cybernetics_core > 0
    ready_robo == 0
    pending_robo == 0
    base_under_threat == 0
    proposed action is STAY_COURSE / ADD_GATEWAYS / TECH_ROBO /
      FORGE_UPGRADES / BUILD_STATIC_DEFENSE / PRODUCE_ARMY / BOOST_WORKERS
  then:
    select TECH_ROBO once minerals >= 150 and vespene >= 100
    otherwise hold STAY_COURSE when gas is ready but minerals are short.

Safety boundary:
  EXPAND is intentionally not intercepted by this guard, preserving the
  existing safe-teacher expand contract.

Default runtime:
  --strategy-policy rule / --strategy-tactic-mode off remains unchanged.
```

New tests:

```text
test_tech_power_redirects_affordable_forge_to_initial_robo
test_tech_power_banks_army_for_initial_robo_when_minerals_short
test_tactic_aware_power_policy_redirects_forge_to_initial_robo
```

Validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
39 passed

.\.venv\Scripts\python.exe -m pytest -q
178 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK
```

Next:

```text
The first-Robo banking guard was not promoted from offline tests alone.
Its fresh-dir guarded Power-only A/B has now been run; see the next section.
```

## 2026-06-24 First-Robo Banking Guard Power A/B

SC2 launch rules were followed:

```text
hidden-window guard:
  pid=26628

SC2 entrypoint:
  scripts\evaluate.py only

No training.
No PPO.
No tactic-aware data collection.
Fresh strategy trajectory dirs were used.
```

Results:

```text
no-filter coverage-teacher:
  run: runs\20260624_135850_20260624_power_ab_first_robo_bank_no_filter_v1
  strategy trajectory: data\trajectories\power_ab_first_robo_bank_no_filter_strategy_v1
  result: 1 Victory / 1 Tie / 1 Defeat
  win_rate: 0.333
  return_code=0 for all games

tactic-rule first-Robo banking:
  run: runs\20260624_140148_20260624_power_ab_first_robo_bank_tactic_v1
  strategy trajectory: data\trajectories\power_ab_first_robo_bank_tactic_strategy_v1
  result: 2 Victory / 1 Tie / 0 Defeat
  win_rate: 0.667
  return_code=0 for all games
```

Diagnostics:

```text
tactic timeline:
  runs\20260624_power_ab_first_robo_bank_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_first_robo_bank_tactic_timeline_v1\artifacts\tactic_timeline.json

power tactic diagnostics:
  runs\20260624_power_ab_first_robo_bank_power_tactics_v1\artifacts\power_tactic_diagnostics.txt
  runs\20260624_power_ab_first_robo_bank_power_tactics_v1\artifacts\power_tactic_diagnostics.json

strategy outcomes:
  runs\20260624_strategy_outcome_power_ab_first_robo_bank_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_first_robo_bank_v1\artifacts\strategy_outcomes.json
```

Key findings:

```text
Source comparison:
  no-filter: files=3 rows=204 training_rows=201 results=1W/1T/1L
    actions: TECH_ROBO=47, ADD_GATEWAYS=13, PRODUCE_ARMY=15,
      BUILD_STATIC_DEFENSE=19
    first TECH_ROBO=262.9s, first ADD_GATEWAYS=91.4s

  tactic-rule: files=3 rows=168 training_rows=165 results=2W/1T/0L
    filter_change_rows=23
    actions: TECH_ROBO=7, ADD_GATEWAYS=6, PRODUCE_ARMY=13,
      BUILD_STATIC_DEFENSE=0, STAY_COURSE=125
    first TECH_ROBO=240.0s, first ADD_GATEWAYS=411.4s

Robo payoff:
  no-filter file 001 Tie: no ready Robo
  no-filter file 002 Victory: ready_robo=331.4 observer=422.9 immortal=none
  no-filter file 003 Defeat: ready_robo=457.1 observer=571.4 immortal=none

  tactic file 001 Tie: ready_robo=297.1 observer=457.1 immortal=525.7
  tactic file 002 Victory: ready_robo=297.1 observer=331.4 immortal=none
  tactic file 003 Victory: ready_robo=354.3 observer=377.1 immortal=none

Threat/static defense:
  no-filter base_threat_rows by file: 11 / 0 / 16
  tactic base_threat_rows by file: 0 / 1 / 0
  tactic selected no BUILD_STATIC_DEFENSE rows in this sample.

Filter changes:
  total filter_change_rows=23, down from midgame Gateway A/B's 71.
  SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS: 6, all before 240s
  TECH_POWER TECH_ROBO -> STAY_COURSE: 5
  TECH_POWER PRODUCE_ARMY -> TECH_ROBO: 1
  TECH_POWER STAY_COURSE -> TECH_ROBO: 1

Interpretation:
  This is the first guarded Power-only A/B where tactic-rule beats no-filter
  in the 3-game sample. First-Robo/Observer payoff and threat exposure improved.
  The result is still not enough to train or collect tactic-aware data because
  Gateway rhythm regressed sharply: first ADD_GATEWAYS is 411.4s versus
  no-filter 91.4s, and ADD_GATEWAYS count is 6 versus 13.
```

Conclusion:

```text
Do not promote first-Robo banking tactic filter to default runtime.
Do not collect tactic-aware training data from this A/B yet.
Do not train outcome/veto/imitation models from this A/B yet.
Next evidence step should be a confirmatory fresh-dir A/B or a narrow Gateway
preservation follow-up that keeps the Robo/threat gains without pushing
ADD_GATEWAYS past the early/midgame window.
```

## 2026-06-24 Offline Gateway Preservation Follow-Up

Offline runtime/test fix after the first-Robo banking A/B exposed selected
Gateway regression. No training, no PPO, no tactic-aware data collection.

Behavior change is restricted to explicit opt-in tactic filtering:

```text
SAFE_MACRO pre-Robo-gas Gateway preservation:
  If:
    game_time < 120s
    ready_gateways == 0
    pending_gateways >= 2
    minerals >= 250
    vespene < 100
    base_under_threat == 0
  then allow up to 3 pending Gateways for ADD_GATEWAYS.

Safety boundary:
  Once Robo gas is ready (vespene >= 100), SAFE_MACRO still caps at 2 pending
  Gateways, preserving first-Robo banking.
  Under active base threat, SAFE_MACRO also keeps the 2 pending Gateway cap.

Default runtime:
  --strategy-policy rule / --strategy-tactic-mode off remains unchanged.
```

New tests:

```text
test_safe_macro_preserves_early_gateway_before_robo_gas_with_two_pending
test_safe_macro_keeps_gateway_cap_after_robo_gas_ready
test_safe_macro_keeps_gateway_cap_under_threat
test_tactic_aware_power_policy_preserves_pre_robo_gas_early_gateway
test_tactic_aware_power_policy_stops_extra_gateway_after_robo_gas_ready
```

Validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
44 passed

.\.venv\Scripts\python.exe -m pytest -q
183 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK
```

## 2026-06-24 Gateway Preservation Power A/B

SC2 launch rules were followed:

```text
hidden-window guard:
  pid=26628

SC2 entrypoint:
  scripts\evaluate.py only

No training.
No PPO.
No tactic-aware data collection.
Fresh strategy trajectory dirs were used.
```

Results:

```text
no-filter coverage-teacher:
  run: runs\20260624_141715_20260624_power_ab_gateway_preserve_no_filter_v1
  strategy trajectory: data\trajectories\power_ab_gateway_preserve_no_filter_strategy_v1
  result: 1 Victory / 0 Tie / 2 Defeat
  win_rate: 0.333
  return_code=0 for all games

tactic-rule Gateway preservation:
  run: runs\20260624_141951_20260624_power_ab_gateway_preserve_tactic_v1
  strategy trajectory: data\trajectories\power_ab_gateway_preserve_tactic_strategy_v1
  result: 2 Victory / 1 Tie / 0 Defeat
  win_rate: 0.667
  return_code=0 for all games
```

Diagnostics:

```text
tactic timeline:
  runs\20260624_power_ab_gateway_preserve_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_gateway_preserve_tactic_timeline_v1\artifacts\tactic_timeline.json

power tactic diagnostics:
  runs\20260624_power_ab_gateway_preserve_power_tactics_v1\artifacts\power_tactic_diagnostics.txt
  runs\20260624_power_ab_gateway_preserve_power_tactics_v1\artifacts\power_tactic_diagnostics.json

strategy outcomes:
  runs\20260624_strategy_outcome_power_ab_gateway_preserve_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_gateway_preserve_v1\artifacts\strategy_outcomes.json
```

Key findings:

```text
Source comparison:
  no-filter: files=3 rows=181 training_rows=178 results=1W/2L
    actions: ADD_GATEWAYS=13, TECH_ROBO=25, BUILD_STATIC_DEFENSE=20
    first ADD_GATEWAYS=91.4s, first TECH_ROBO=240.0s

  tactic-rule: files=3 rows=173 training_rows=170 results=2W/1T/0L
    filter_change_rows=26
    actions: ADD_GATEWAYS=11, TECH_ROBO=9, BUILD_STATIC_DEFENSE=4,
      PRODUCE_ARMY=23
    first ADD_GATEWAYS=91.4s, first TECH_ROBO=240.0s

Gateway:
  Previous first-Robo banking A/B tactic first ADD_GATEWAYS was 411.4s.
  Gateway preservation tactic first ADD_GATEWAYS is now 91.4s.
  ADD_GATEWAYS count improved from 6 to 11.

Robo payoff:
  no-filter file 001 Defeat: ready_robo=422.9, no Observer/Immortal,
    blocker=action_not_triggered
  no-filter file 002 Defeat: ready_robo=297.1, no Observer/Immortal,
    blocker=action_not_triggered
  no-filter file 003 Victory: no ready Robo

  tactic file 001 Victory: ready_robo=297.1 observer=331.4 immortal=none
  tactic file 002 Tie: ready_robo=320.0 observer=480.0 immortal=560.0
  tactic file 003 Victory: ready_robo=297.1 observer=none immortal=none,
    blocker=resource_or_supply_blocked

Threat/static defense:
  no-filter base_threat_rows by file: 18 / 12 / 2
  tactic base_threat_rows by file: 0 / 17 / 2
  tactic selected BUILD_STATIC_DEFENSE=4, no-filter selected 20.

Interpretation:
  Gateway preservation fixed the selected Gateway timing regression while
  keeping the 3-game result positive and preserving ready Robo in all tactic
  files. Observer payoff improved versus no-filter, but not fully: one tactic
  Victory still had ready Robo without Observer due resource/supply blocking.
  The tactic Tie still had 17 threat rows, so static-defense/threat handling is
  not solved.
```

Conclusion:

```text
Do not promote tactic-aware mode to default runtime.
Do not collect tactic-aware training data from this A/B yet.
Do not train outcome/veto/imitation models from this A/B yet.
This branch is now worth another confirmatory A/B or a very small
post-ready-Robo Observer/resource follow-up, but default rule/off remains the
baseline.
```

## 2026-06-24 Gateway Preservation Confirmatory Power A/B

Fresh-dir guarded Power-only confirmatory A/B. No training, no PPO, no
tactic-aware data collection. Default runtime remains:

```text
--strategy-policy rule
--strategy-tactic-mode off
```

SC2 launch discipline:

```text
hidden-window guard:
  pid=26628

SC2 entrypoint:
  scripts\evaluate.py only

fresh trajectory dirs:
  data\trajectories\power_ab_gateway_confirm6_no_filter_strategy_v1
  data\trajectories\power_ab_gateway_confirm6_no_filter_topup1_strategy_v1
  data\trajectories\power_ab_gateway_confirm6_tactic_strategy_v1
```

Eval runs:

```text
no-filter coverage-teacher:
  run: runs\20260624_143758_20260624_power_ab_gateway_confirm6_no_filter_v1
  result: 2 Victory / 3 Defeat / 1 NO_RESULT
  note: game 4 returned code=1 after 550.9s and wrote an empty strategy file.

no-filter top-up:
  run: runs\20260624_145141_20260624_power_ab_gateway_confirm6_no_filter_topup1_v1
  result: 1 Tie

no-filter valid aggregate:
  2 Victory / 1 Tie / 3 Defeat

tactic-rule:
  run: runs\20260624_145322_20260624_power_ab_gateway_confirm6_tactic_v1
  result: 2 Victory / 1 Tie / 3 Defeat
  return_code=0 for all 6 games
```

Diagnostics:

```text
tactic timeline:
  runs\20260624_power_ab_gateway_confirm6_tactic_timeline_v1\artifacts\tactic_timeline.txt
  runs\20260624_power_ab_gateway_confirm6_tactic_timeline_v1\artifacts\tactic_timeline.json

power tactic diagnostics:
  runs\20260624_power_ab_gateway_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics.txt
  runs\20260624_power_ab_gateway_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics.json

strategy outcomes:
  runs\20260624_strategy_outcome_power_ab_gateway_confirm6_v1\artifacts\strategy_outcomes.txt
  runs\20260624_strategy_outcome_power_ab_gateway_confirm6_v1\artifacts\strategy_outcomes.json
```

Key aggregate comparison, excluding the no-filter empty NO_RESULT trajectory:

```text
Results:
  no-filter valid: 2W / 1T / 3L
  tactic-rule:     2W / 1T / 3L

Gateway:
  no-filter first ADD_GATEWAYS=91.4s, count=25
  tactic-rule first ADD_GATEWAYS=91.4s, count=23
  SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS count=1, early_before_240=1.

Robo payoff:
  no-filter: ready_robo 4/6, Observer 4/6, Immortal 0/6
  tactic-rule: ready_robo 5/6, Observer 5/6, Immortal 2/6
  tactic observer delay was slower on average: 96.0s after ready Robo vs
  no-filter 62.9s.

Threat/static defense:
  no-filter base_threat_rows=39, BUILD_STATIC_DEFENSE=39
  tactic-rule base_threat_rows=44, BUILD_STATIC_DEFENSE=13
  tactic shifted many threat/static rows into PRODUCE_ARMY or STAY_COURSE.

Filter changes:
  tactic training_filter_change_rows=75
  largest changes:
    TECH_POWER TECH_ROBO -> STAY_COURSE: 26
    RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 13
    ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 9
```

Conclusion:

```text
The confirmatory A/B does not support promotion, tactic-aware data collection,
or training. Gateway timing is preserved and Robo payoff improves, but the
sample result only ties no-filter, static-defense/threat handling regresses,
and filter changes are not sparse. Next runtime work should be narrower:
static-defense/threat retention and/or diagnosing excessive TECH_ROBO ->
STAY_COURSE filtering.
```

## 2026-06-24 Offline Static-Defense Retention Follow-Up

Offline runtime/test fix after the confirmatory A/B showed worse
static-defense/threat handling. No SC2 launch in this step. No training, no
PPO, no tactic-aware data collection. Default runtime remains:

```text
--strategy-policy rule
--strategy-tactic-mode off
```

Behavior change is restricted to explicit opt-in tactic filtering:

```text
Static-defense retention now applies to:
  TECH_POWER
  ANTI_RUSH_DEFENSE
  ROBO_TIMING
  ANTI_AIR_RESPONSE
  RECOVERY

When proposed action is BUILD_STATIC_DEFENSE and base_under_threat > 0:
  pending_static_defense >= cap:
    do not override fallback

  minerals >= 100:
    keep BUILD_STATIC_DEFENSE

  minerals < 100 and no ready static defense:
    STAY_COURSE to bank for static defense

  minerals < 100 and ready static defense already exists:
    do not override fallback, so PRODUCE_ARMY can still fire
```

Confirm6 offline replay with the new filter:

```text
Old largest static-defense suppressions:
  RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 13
  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 9

New replay:
  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> STAY_COURSE: 3
  RECOVERY BUILD_STATIC_DEFENSE -> STAY_COURSE appears on low-mineral/no-static rows.

TECH_POWER TECH_ROBO -> STAY_COURSE remained 27 in replay, because most sampled
rows had minerals < 150 and no ready/pending Robo; treat this as affordability
diagnostic noise before changing Robo behavior again.
```

New tests:

```text
test_anti_air_banks_for_static_defense_under_threat_without_static
test_anti_air_uses_army_when_static_exists_and_minerals_short
test_recovery_preserves_affordable_static_defense_under_threat
test_tactic_aware_anti_air_banks_static_defense_under_threat
```

Validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
48 passed

.\.venv\Scripts\python.exe -m pytest -q
187 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK
```

Conclusion:

```text
This is a narrow opt-in runtime fix. It is not yet promoted by online evidence.
Next guarded Power-only A/B should verify that threat rows/static-defense
recover without losing Gateway timing or Robo payoff.
```

## 2026-06-24 Robo Banking Context Diagnostics And Static-Retention Confirm6 A/B

Follow-up after the offline static-defense retention fix. Defaults remain:

```text
--strategy-policy rule
--strategy-tactic-mode off
```

No training, no PPO, no tactic-aware data collection.

### Code / Diagnostics

`rl\power_tactic_diagnostics.py` now reports
`robo_banking_filter_contexts` for `TECH_ROBO -> STAY_COURSE` filter changes.
Contexts distinguish:

```text
first_robo_affordable
first_robo_mineral_short
first_robo_vespene_short
first_robo_resource_short
pending_robo_cap
ready_robo_already_exists
base_under_threat
no_cybernetics_core
other
```

This is diagnostics-only; it does not affect runtime policy or observation
schema. Text output from `scripts\diagnose_power_tactics.py` now includes a
`robo_banking_filter_context:` section.

Old confirm6 replay with the new diagnostic:

```text
runs\20260624_power_tactic_robo_banking_context_v1\artifacts\power_tactic_diagnostics.txt
runs\20260624_power_tactic_robo_banking_context_v1\artifacts\power_tactic_diagnostics.json

TECH_POWER TECH_ROBO -> STAY_COURSE: 26
robo_banking_filter_context:
  Power, TECH_POWER, first_robo_mineral_short: 26
```

So the largest Robo filter-change bucket is resource banking, not a clear
false veto of an affordable first Robo.

### Validation

```text
.\.venv\Scripts\python.exe -m pytest tests\test_power_tactic_diagnostics.py -q
4 passed

.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py tests\test_power_tactic_diagnostics.py -q
52 passed

.\.venv\Scripts\python.exe -m pytest -q
188 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK
```

### Guarded A/B

Hidden-window guard was run before SC2 launch:

```text
guard pid: 26628
```

All SC2 launches used `scripts\evaluate.py`; no visible `run.py` was launched.
Fresh dirs were used.

No-filter coverage-teacher:

```text
run: runs\20260624_152347_20260624_power_ab_static_retention_confirm6_no_filter_v1
army trajectory: data\trajectories\power_ab_static_retention_confirm6_no_filter_army_v1
strategy trajectory: data\trajectories\power_ab_static_retention_confirm6_no_filter_strategy_v1
result: 3 Victory / 3 Defeat
return_code: 0 for all 6 games
```

Tactic-rule:

```text
run: runs\20260624_152825_20260624_power_ab_static_retention_confirm6_tactic_v1
army trajectory: data\trajectories\power_ab_static_retention_confirm6_tactic_army_v1
strategy trajectory: data\trajectories\power_ab_static_retention_confirm6_tactic_strategy_v1
result: 3 Victory / 1 Tie / 2 Defeat
return_code: 0 for all 6 games
```

Diagnostics:

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

### Key Results

```text
Results:
  no-filter:   3W / 3L
  tactic-rule: 3W / 1T / 2L

Gateway:
  no-filter first ADD_GATEWAYS=91.4s, count=25
  tactic-rule first ADD_GATEWAYS=91.4s, count=15
  tactic SAFE_MACRO ADD_GATEWAYS -> BOOST_WORKERS count=2

Robo payoff:
  no-filter: ready_robo 5/6, Observer 3/6, Immortal 0/6
  tactic-rule: ready_robo 5/6, Observer 4/6, Immortal 2/6
  observer delay after ready Robo:
    no-filter avg 34.3s
    tactic avg 82.9s

Threat/static defense:
  no-filter base_threat_rows=46, BUILD_STATIC_DEFENSE=43
  tactic-rule base_threat_rows=52, BUILD_STATIC_DEFENSE=6
  no-filter threat_action_counts:
    STAY_COURSE=2, FORGE_UPGRADES=1, BUILD_STATIC_DEFENSE=43
  tactic threat_action_counts:
    STAY_COURSE=22, TECH_ROBO=3, BUILD_STATIC_DEFENSE=6, PRODUCE_ARMY=21

Filter changes:
  tactic training_filter_change_rows=82
  largest changes:
    TECH_POWER TECH_ROBO -> STAY_COURSE: 26
    ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 14
    ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> STAY_COURSE: 7
    RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 6
    ROBO_TIMING BUILD_STATIC_DEFENSE -> STAY_COURSE: 5

Robo banking context:
  TECH_POWER TECH_ROBO -> STAY_COURSE:
    first_robo_mineral_short: 26
```

Per-file red flags:

```text
Tactic file 004:
  Victory, but no first ADD_GATEWAYS label and no ready Robo.
  TECH_ROBO first selected at 502.9s.
  21 TECH_ROBO -> STAY_COURSE rows were first_robo_mineral_short.

Tactic file 005:
  Defeat, ready Robo at 308.6s, but no Observer/Immortal.
  Observer blocker: action_not_triggered.

Tactic file 001:
  Defeat despite Observer + Immortal.
  base_threat_rows=23, PRODUCE_ARMY under threat=15.
```

Conclusion:

```text
Do not promote tactic filter.
Do not collect tactic-aware training data from this run.
Do not train action-outcome / veto / imitation models yet.

The latest opt-in tactic rule improves raw result count slightly and preserves
first Gateway timing, and it improves Immortal payoff vs no-filter. However,
static-defense/threat handling is still worse: fewer actual
BUILD_STATIC_DEFENSE rows, more threat rows, more STAY_COURSE/PRODUCE_ARMY
under threat, and 82 tactic filter changes. The Robo banking context diagnostic
also says the biggest TECH_ROBO -> STAY_COURSE bucket is mineral-short banking,
so the next fix should not target first-Robo affordability. It should diagnose
and narrow active-threat static-defense suppression.
```

## 2026-06-24 Pending Static Wait Follow-Up

Offline follow-up; no SC2 launch in this step. No training, no PPO, no
tactic-aware data collection. Defaults remain:

```text
--strategy-policy rule
--strategy-tactic-mode off
```

### Diagnostic Extension

`rl\power_tactic_diagnostics.py` now also reports
`static_defense_filter_contexts` for active-threat
`BUILD_STATIC_DEFENSE -> <other>` tactic-filter changes. Contexts:

```text
no_static_affordable
no_static_mineral_short
pending_static_waiting
pending_static_with_ready
ready_static_low_minerals
ready_static_affordable
other
```

New artifact:

```text
runs\20260624_power_ab_static_retention_confirm6_static_context_v2\artifacts\power_tactic_diagnostics.txt
runs\20260624_power_ab_static_retention_confirm6_static_context_v2\artifacts\power_tactic_diagnostics.json
```

Key context counts on the static-retention confirm6 tactic trajectory:

```text
no_static_affordable:
  0

ready_static_low_minerals -> PRODUCE_ARMY:
  ANTI_AIR_RESPONSE: 11
  RECOVERY: 5

no_static_mineral_short -> STAY_COURSE:
  ANTI_AIR_RESPONSE: 7
  ROBO_TIMING: 5
  TECH_POWER: 4
  RECOVERY: 1

pending_static_waiting:
  ANTI_AIR_RESPONSE -> PRODUCE_ARMY: 2
  ANTI_AIR_RESPONSE -> TECH_ROBO: 2
  RECOVERY -> PRODUCE_ARMY: 1

pending_static_with_ready:
  ANTI_AIR_RESPONSE -> PRODUCE_ARMY: 1
```

Interpretation:

```text
The previous A/B did not show an affordable/no-static false veto. Most static
suppression happened with low minerals, an existing ready static, or a pending
static. The actionable bucket is pending_static_waiting: static defense is
already being built, no static is ready yet, and old fallback still spends the
turn on PRODUCE_ARMY or TECH_ROBO.
```

### Runtime Fix

Minimal opt-in tactic-filter change in `rl\tactics.py`:

```text
When proposed action is BUILD_STATIC_DEFENSE and base_under_threat > 0:
  if pending_static_defense >= cap and ready_static_defense == 0:
    return STAY_COURSE

  if pending_static_defense >= cap and ready_static_defense > 0:
    keep previous fallback behavior
```

Offline replay of the old confirm6 tactic trajectory with current filter:

```text
old_after != new_after rows: 5

ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE:
  PRODUCE_ARMY -> STAY_COURSE: 2
  TECH_ROBO -> STAY_COURSE: 2

RECOVERY BUILD_STATIC_DEFENSE:
  PRODUCE_ARMY -> STAY_COURSE: 1
```

The five changed rows all had:

```text
base_under_threat=1
pending_static_defense=1
ready_static_defense=0
```

### Validation

```text
.\.venv\Scripts\python.exe -m pytest tests\test_power_tactic_diagnostics.py -q
5 passed

.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py tests\test_power_tactic_diagnostics.py -q
56 passed

.\.venv\Scripts\python.exe -m pytest -q
192 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK
```

Conclusion:

```text
This is a narrow opt-in runtime fix with offline evidence only. It should not
be promoted, trained on, or used for tactic-aware data collection yet. Next
online step, if requested, should be a guarded fresh-dir Power-only A/B focused
on threat rows, BUILD_STATIC_DEFENSE rows, TECH_ROBO/PRODUCE_ARMY under threat,
Gateway timing, and Robo payoff.
```

## 2026-06-24 Pending Static Wait Confirmatory Power A/B

Purpose:

```text
Validate the opt-in pending-static wait follow-up online against the same
Power-only guarded comparison pattern:

  no-filter coverage-teacher
  vs
  coverage-teacher + --strategy-tactic-mode rule

No training, no PPO, no tactic-aware data collection, no default baseline change.
```

Safety:

```text
hidden-window guard command was run before SC2 launch.
guard pid: 26628
scripts\check_env.py: OK
SC2 launched only through scripts\evaluate.py -> scripts\safe_launch.py.
No visible run.py launch.
```

Commands:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"

.\.venv\Scripts\python.exe scripts\check_env.py

.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 6 --run-root runs --run-name 20260624_power_ab_pending_static_wait_confirm6_no_filter_v1 --tag strategy-tactic --tag power-ab --tag pending-static-wait --tag no-filter --policy-name power_ab_pending_static_wait_confirm6_no_filter_v1 --army-policy rule --strategy-policy coverage-teacher --strategy-tactic-mode off --trajectory-dir data\trajectories\power_ab_pending_static_wait_confirm6_no_filter_army_v1 --strategy-trajectory-dir data\trajectories\power_ab_pending_static_wait_confirm6_no_filter_strategy_v1 --record-decision-interval 16 --game-time-limit 900

.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"

.\.venv\Scripts\python.exe scripts\evaluate.py --maps AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 6 --run-root runs --run-name 20260624_power_ab_pending_static_wait_confirm6_tactic_v1 --tag strategy-tactic --tag power-ab --tag pending-static-wait --tag tactic-rule --policy-name power_ab_pending_static_wait_confirm6_tactic_v1 --army-policy rule --strategy-policy coverage-teacher --strategy-tactic-mode rule --trajectory-dir data\trajectories\power_ab_pending_static_wait_confirm6_tactic_army_v1 --strategy-trajectory-dir data\trajectories\power_ab_pending_static_wait_confirm6_tactic_strategy_v1 --record-decision-interval 16 --game-time-limit 900
```

Run / trajectory paths:

```text
no-filter:
  run: runs\20260624_155710_20260624_power_ab_pending_static_wait_confirm6_no_filter_v1
  army trajectory: data\trajectories\power_ab_pending_static_wait_confirm6_no_filter_army_v1
  strategy trajectory: data\trajectories\power_ab_pending_static_wait_confirm6_no_filter_strategy_v1
  result: 1W / 2T / 3L

tactic-rule:
  run: runs\20260624_160251_20260624_power_ab_pending_static_wait_confirm6_tactic_v1
  army trajectory: data\trajectories\power_ab_pending_static_wait_confirm6_tactic_army_v1
  strategy trajectory: data\trajectories\power_ab_pending_static_wait_confirm6_tactic_strategy_v1
  result: 1W / 1T / 4L
```

Diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_tactics.py data\trajectories\power_ab_pending_static_wait_confirm6_tactic_strategy_v1 --show-files --show-filter-timeline --json-output runs\20260624_power_ab_pending_static_wait_confirm6_tactic_timeline_v1\artifacts\tactic_timeline.json --text-output runs\20260624_power_ab_pending_static_wait_confirm6_tactic_timeline_v1\artifacts\tactic_timeline.txt

.\.venv\Scripts\python.exe scripts\diagnose_power_tactics.py data\trajectories\power_ab_pending_static_wait_confirm6_tactic_strategy_v1 --show-files --json-output runs\20260624_power_ab_pending_static_wait_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics.json --text-output runs\20260624_power_ab_pending_static_wait_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics.txt

.\.venv\Scripts\python.exe scripts\diagnose_power_tactics.py data\trajectories\power_ab_pending_static_wait_confirm6_no_filter_strategy_v1 --show-files --json-output runs\20260624_power_ab_pending_static_wait_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics_no_filter.json --text-output runs\20260624_power_ab_pending_static_wait_confirm6_power_tactics_v1\artifacts\power_tactic_diagnostics_no_filter.txt

.\.venv\Scripts\python.exe scripts\diagnose_strategy_outcomes.py data\trajectories\power_ab_pending_static_wait_confirm6_no_filter_strategy_v1 data\trajectories\power_ab_pending_static_wait_confirm6_tactic_strategy_v1 --show-files --json-output runs\20260624_strategy_outcome_power_ab_pending_static_wait_confirm6_v1\artifacts\strategy_outcomes.json --text-output runs\20260624_strategy_outcome_power_ab_pending_static_wait_confirm6_v1\artifacts\strategy_outcomes.txt
```

Diagnostic artifact paths:

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

Key results:

```text
Result:
  no-filter: 1W / 2T / 3L
  tactic-rule: 1W / 1T / 4L

Gateway:
  no-filter first ADD_GATEWAYS: first=91.4s avg=97.1s per_file=91.4,91.4,91.4,102.9,102.9,102.9
  tactic-rule first ADD_GATEWAYS: first=91.4s avg=121.9s per_file=91.4,102.9,102.9,91.4,171.4,171.4
  no-filter ADD_GATEWAYS count: 28
  tactic-rule ADD_GATEWAYS count: 15

Robo:
  no-filter ready/Observer/Immortal: 3/6, 3/6, 0/6
  tactic-rule ready/Observer/Immortal: 5/6, 5/6, 1/6
  no-filter Observer delay avg after ready Robo: 64.8s
  tactic-rule Observer delay avg after ready Robo: 73.1s
  tactic-rule Immortal blockers: no_ready_robo=1, none=1, resource_or_supply_blocked=4

Threat/static:
  no-filter base_threat_rows: 33
  tactic-rule base_threat_rows: 72
  no-filter BUILD_STATIC_DEFENSE count: 33
  tactic-rule BUILD_STATIC_DEFENSE count: 15
  no-filter threat actions: BUILD_STATIC_DEFENSE=33
  tactic-rule threat actions: STAY_COURSE=32, PRODUCE_ARMY=24, BUILD_STATIC_DEFENSE=15, BOOST_WORKERS=1

Filter:
  tactic-rule training_filter_change_rows: 85
  top changes:
    TECH_POWER TECH_ROBO -> STAY_COURSE: 20
    ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 12
    RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 11
    ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> STAY_COURSE: 7
    TECH_POWER BUILD_STATIC_DEFENSE -> STAY_COURSE: 6
  robo_banking_filter_context:
    TECH_POWER first_robo_mineral_short: 20
  static_defense_filter_context:
    ready_static_low_minerals -> PRODUCE_ARMY: 19
    no_static_mineral_short -> STAY_COURSE: 13
    pending_static_waiting -> STAY_COURSE: 4
    pending_static_with_ready -> PRODUCE_ARMY: 5
  filter delta:
    ADD_GATEWAYS=-10
    BUILD_STATIC_DEFENSE=-41
    PRODUCE_ARMY=+30
    STAY_COURSE=+37
    TECH_ROBO=-14
```

Post-A/B validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py tests\test_power_tactic_diagnostics.py -q
56 passed

.\.venv\Scripts\python.exe -m pytest -q
192 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK

Get-Process SC2,SC2_x64 -ErrorAction SilentlyContinue
no output
```

Decision:

```text
The pending-static wait follow-up did not validate online.

It improved Robo construction and Observer/Immortal payoff versus no-filter,
but it worsened the match result, Gateway volume/timing, and active-threat
defense. The active-threat surface remains the main blocker: the tactic filter
still converts too many BUILD_STATIC_DEFENSE rows into PRODUCE_ARMY/STAY_COURSE,
and base_threat_rows rose sharply.

Do not promote tactic-rule.
Do not collect tactic-aware data from this run.
Do not train action-outcome / veto / imitation models from this run.
Do not run PPO.
Keep default runtime unchanged:
  --strategy-policy rule
  --strategy-tactic-mode off
```

Recommended next offline step:

```text
Do not add another online patch immediately. First isolate active-threat rows
where ready static exists but threat persists:

  ready_static_low_minerals -> PRODUCE_ARMY
  pending_static_with_ready -> PRODUCE_ARMY
  BUILD_STATIC_DEFENSE -> STAY_COURSE under no_static_mineral_short

Compare whether these rows clear threat by +30/+60/+90/+120s and whether
PRODUCE_ARMY actually increases army count enough to justify suppressing static.
Only then consider one more minimal opt-in static-defense escalation rule.
```

## 2026-06-24 Development Route Correction And Active-Threat Outcome Slice

Route correction:

```text
Freeze the current tactic-rule runtime. Do not continue the loop of:

  small hand-written tactic patch
  small Power-only A/B
  another patch to fix the previous side effect

This pattern improved local Robo metrics but repeatedly regressed Gateway
volume and active-threat defense. Future runtime changes need a narrower
offline outcome proof first, plus an explicit rollback path and no default
baseline change.
```

New offline diagnostics:

```text
rl\active_threat_outcome_diagnostics.py
scripts\diagnose_active_threat_outcomes.py
tests\test_active_threat_outcome_diagnostics.py
```

Purpose:

```text
For active-threat rows where tactic metadata changes:

  BUILD_STATIC_DEFENSE -> <other>

group by:

  opponent_ai_build
  tactic_id
  before_action
  after_action
  static_defense_filter_context

and report +30/+60/+90/+120s outcome:

  threat_cleared / threat_persisted
  army_count_delta
  static_defense_delta
  pending_static_defense_delta
  base_under_threat_after
  minerals_after
  gateway_idle_after
```

Focused test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_active_threat_outcome_diagnostics.py -q
```

Result:

```text
2 passed
```

Related focused diagnostics validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_active_threat_outcome_diagnostics.py tests\test_power_tactic_diagnostics.py tests\test_strategy_outcome_diagnostics.py -q
```

Result:

```text
12 passed
```

Offline run:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_active_threat_outcomes.py data\trajectories\power_ab_pending_static_wait_confirm6_tactic_strategy_v1 --show-files --json-output runs\20260624_active_threat_outcome_pending_static_wait_v1\artifacts\active_threat_outcomes.json --text-output runs\20260624_active_threat_outcome_pending_static_wait_v1\artifacts\active_threat_outcomes.txt
```

Artifacts:

```text
runs\20260624_active_threat_outcome_pending_static_wait_v1\artifacts\active_threat_outcomes.txt
runs\20260624_active_threat_outcome_pending_static_wait_v1\artifacts\active_threat_outcomes.json
```

Key findings:

```text
active_threat_filter_rows: 41

ANTI_AIR_RESPONSE, BUILD_STATIC_DEFENSE -> PRODUCE_ARMY, ready_static_low_minerals:
  count=11
  +30s: threat_persisted=11/11, army_count_delta=-3.6
  +60s: threat_persisted=11/11, army_count_delta=-7.0
  +90s: threat_persisted=10/11, threat_cleared=1/11
  +120s: threat_persisted=5/11, threat_cleared=6/11

RECOVERY, BUILD_STATIC_DEFENSE -> PRODUCE_ARMY, ready_static_low_minerals:
  count=8
  +60s: threat_cleared=6/8, threat_persisted=2/8
  +120s: threat_cleared=8/8

ANTI_AIR_RESPONSE, BUILD_STATIC_DEFENSE -> STAY_COURSE, pending_static_waiting:
  count=4
  static_defense_increased=4/4 by +30s
  threat_persisted=4/4 through +120s

TECH_POWER, BUILD_STATIC_DEFENSE -> STAY_COURSE, no_static_mineral_short:
  count=6
  +60s: threat_cleared=6/6

RECOVERY, BUILD_STATIC_DEFENSE -> STAY_COURSE, no_static_mineral_short:
  count=4
  +60s: threat_cleared=4/4

ANTI_AIR_RESPONSE, BUILD_STATIC_DEFENSE -> STAY_COURSE, no_static_mineral_short:
  count=3
  +60s: threat_cleared=2/3, threat_persisted=1/3
```

Interpretation:

```text
Do not make a broad static-defense retention rule.

The harmful bucket is tactic-specific:

  ANTI_AIR_RESPONSE + ready_static_low_minerals -> PRODUCE_ARMY

It does not produce a useful army payoff and leaves threat active for too long.
The same ready_static_low_minerals context under RECOVERY is much healthier by
+120s, so a global ready-static fallback change would overcorrect.

pending_static_waiting -> STAY_COURSE makes static finish, but did not clear
threat in these ANTI_AIR rows. The issue may be anti-air defense quality /
positioning / insufficient army, not just static completion.
```

Exact bad-bucket row inspection:

```text
Bucket:
  ANTI_AIR_RESPONSE
  BUILD_STATIC_DEFENSE -> PRODUCE_ARMY
  ready_static_low_minerals

Rows: 11
Files:
  20260624_160322_AcropolisLE_Hard_Terran_Power_002.jsonl: 2 rows, t=662.9-674.3
  20260624_160425_AcropolisLE_Hard_Terran_Power_003.jsonl: 2 rows, t=468.6-491.4
  20260624_160510_AcropolisLE_Hard_Terran_Power_004.jsonl: 7 rows, t=502.9-571.4

Shared shape:
  minerals: 30-75
  vespene: 1497-2253
  supply_left: 13-59
  ready_static_defense: 1-3
  base_under_air_threat: 1 for all rows
  gateway_idle_count: grows as high as 4 in the worst file
```

Implication:

```text
The bad bucket is not supply blocked and not vespene blocked. It is mostly a
low-mineral anti-air emergency with existing static defense and large gas bank.
Switching to PRODUCE_ARMY does not create enough payoff. The next fix should
not be a generic static-defense retention rule; it should inspect whether
ANTI_AIR_RESPONSE needs a mineral-bank / anti-air production reserve or a
different fallback when minerals are too low to build more static immediately.
```

Next implementation target:

```text
If we do another runtime change, keep it opt-in and narrow:

  only ANTI_AIR_RESPONSE
  only active base threat
  only BUILD_STATIC_DEFENSE proposed
  only ready_static_low_minerals or pending_static_waiting contexts

Before changing runtime, inspect the exact rows/timeline to decide whether the
better fallback should be STAY_COURSE, BUILD_STATIC_DEFENSE retention, or a
different anti-air response bias. Do not touch default rule/off behavior.
```

Final validation for this pass:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_env.py
Get-Process SC2,SC2_x64 -ErrorAction SilentlyContinue
```

Result:

```text
194 passed
check_env: OK
SC2 process check: no output
SC2 launched: no
training/PPO: no
```

## 2026-06-24 Anti-Air Ready-Static Banking Follow-Up

Offline runtime follow-up only. No SC2 launch, no hidden-window guard needed,
no training, no PPO, no tactic-aware data collection. Defaults remain:

```text
--strategy-policy rule
--strategy-tactic-mode off
```

Motivation:

```text
Active-threat outcome diagnostics showed one clearly harmful bucket:

  ANTI_AIR_RESPONSE
  BUILD_STATIC_DEFENSE -> PRODUCE_ARMY
  ready_static_low_minerals

count=11
+60s: threat_persisted=11/11, army_count_delta=-7.0
+120s: threat_persisted=5/11, threat_cleared=6/11

The same ready_static_low_minerals context under RECOVERY cleared by +120s,
so the fix must not be global.
```

Implemented in `rl\tactics.py`:

```text
Only inside explicit opt-in tactic filtering, when proposed action is
BUILD_STATIC_DEFENSE under active base threat:

  tactic_id == ANTI_AIR_RESPONSE
  base_under_air_threat > 0
  ready_static_defense > 0
  pending_static_defense <= 0
  minerals < 100

return STAY_COURSE instead of allowing fallback to PRODUCE_ARMY.

Ground-only threat with ready static still falls back to PRODUCE_ARMY.
RECOVERY behavior is unchanged.
```

Tests added/updated:

```text
tests\test_tactics.py:
  test_anti_air_banks_when_air_threat_static_exists_and_minerals_short
  test_anti_air_uses_army_for_ground_threat_when_static_exists_and_minerals_short

tests\test_tactic_strategy_policy.py:
  test_tactic_aware_anti_air_banks_when_ready_static_low_minerals
```

Offline replay against
`data\trajectories\power_ab_pending_static_wait_confirm6_tactic_strategy_v1`:

```text
changed_rows: 11

Power / ANTI_AIR_RESPONSE / BUILD_STATIC_DEFENSE:
  recorded PRODUCE_ARMY -> current STAY_COURSE: 11

Files:
  20260624_160322_AcropolisLE_Hard_Terran_Power_002.jsonl: 2 rows
  20260624_160425_AcropolisLE_Hard_Terran_Power_003.jsonl: 2 rows
  20260624_160510_AcropolisLE_Hard_Terran_Power_004.jsonl: 7 rows

Shared shape:
  minerals: 30-75
  vespene: 1497-2253
  supply_left: 13-59
  ready_static_defense: 1-3
  pending_static_defense: 0
  base_under_air_threat: 1
  ready_robo: 1
```

Validation:

```text
initial focused test before runtime fix:
  2 failed, 51 passed

.\.venv\Scripts\python.exe -m pytest tests\test_tactics.py tests\test_tactic_strategy_policy.py -q
  53 passed

.\.venv\Scripts\python.exe -m pytest tests\test_active_threat_outcome_diagnostics.py tests\test_power_tactic_diagnostics.py tests\test_strategy_outcome_diagnostics.py -q
  12 passed

.\.venv\Scripts\python.exe -m pytest -q
  196 passed

.\.venv\Scripts\python.exe scripts\check_env.py
  OK
```

Decision:

```text
This is a narrow, opt-in runtime repair for a diagnosis-proven bad bucket.
It is not promoted to default runtime. Do not collect tactic-aware training
data or train from this change yet.

Next evidence gate, if requested:
  guarded fresh-dir Power-only A/B
  hidden-window guard first
  scripts\evaluate.py only
  tactic timeline + power tactic + strategy outcome diagnostics
```

## 2026-06-24 Anti-Air Ready-Static Guarded Power A/B

Fresh-dir guarded Power-only A/B for the anti-air ready-static banking follow-up.

Safety:

```text
hidden-window guard command was run before each SC2 batch.
guard pid: 26628
SC2 launched only through scripts\evaluate.py -> scripts\safe_launch.py.
No visible run.py launch.
No training, no PPO, no tactic-aware data collection.
Default rule/off path unchanged.
```

Commands used the same scenario:

```text
AcropolisLE / Hard / Terran / Power
games_per_combo=6
army-policy=rule
strategy-policy=coverage-teacher
record_decision_interval=16
game_time_limit=900
```

Run / trajectory paths:

```text
no-filter:
  run: runs\20260624_164350_20260624_power_ab_anti_air_ready_static_no_filter_v1
  army trajectory: data\trajectories\power_ab_anti_air_ready_static_no_filter_army_v1
  strategy trajectory: data\trajectories\power_ab_anti_air_ready_static_no_filter_strategy_v1
  result: 2W / 2T / 2L

tactic-rule:
  run: runs\20260624_164917_20260624_power_ab_anti_air_ready_static_tactic_v1
  army trajectory: data\trajectories\power_ab_anti_air_ready_static_tactic_army_v1
  strategy trajectory: data\trajectories\power_ab_anti_air_ready_static_tactic_strategy_v1
  result: 0W / 0T / 6L
```

Diagnostic artifacts:

```text
runs\20260624_power_ab_anti_air_ready_static_tactic_timeline_v1\artifacts\tactic_timeline.txt
runs\20260624_power_ab_anti_air_ready_static_tactic_timeline_v1\artifacts\tactic_timeline.json

runs\20260624_power_ab_anti_air_ready_static_power_tactics_v1\artifacts\power_tactic_diagnostics.txt
runs\20260624_power_ab_anti_air_ready_static_power_tactics_v1\artifacts\power_tactic_diagnostics.json
runs\20260624_power_ab_anti_air_ready_static_power_tactics_v1\artifacts\power_tactic_diagnostics_no_filter.txt
runs\20260624_power_ab_anti_air_ready_static_power_tactics_v1\artifacts\power_tactic_diagnostics_no_filter.json

runs\20260624_strategy_outcome_power_ab_anti_air_ready_static_v1\artifacts\strategy_outcomes.txt
runs\20260624_strategy_outcome_power_ab_anti_air_ready_static_v1\artifacts\strategy_outcomes.json

runs\20260624_active_threat_outcome_anti_air_ready_static_v1\artifacts\active_threat_outcomes.txt
runs\20260624_active_threat_outcome_anti_air_ready_static_v1\artifacts\active_threat_outcomes.json
```

Key results:

```text
Match result:
  no-filter:   2W / 2T / 2L
  tactic-rule: 0W / 0T / 6L

Gateway:
  first ADD_GATEWAYS stayed at 91.4s for both sides.
  no-filter ADD_GATEWAYS count: 33
  tactic-rule ADD_GATEWAYS count: 19

Robo:
  no-filter ready/Observer/Immortal: 4/6, 4/6, 0/6
  tactic-rule ready/Observer/Immortal: 6/6, 6/6, 0/6
  Tactic-rule improved Robo/Observer coverage but still failed every game and
  did not produce an Immortal.

Threat/static:
  no-filter threat actions:
    BUILD_STATIC_DEFENSE=26, STAY_COURSE=7, PRODUCE_ARMY=1
  tactic-rule threat actions:
    STAY_COURSE=35, PRODUCE_ARMY=27, BUILD_STATIC_DEFENSE=19, BOOST_WORKERS=1

Action totals:
  no-filter:
    ADD_GATEWAYS=33, TECH_ROBO=34, BUILD_STATIC_DEFENSE=26, PRODUCE_ARMY=25
  tactic-rule:
    ADD_GATEWAYS=19, TECH_ROBO=6, BUILD_STATIC_DEFENSE=19, PRODUCE_ARMY=47

Filter:
  tactic training_filter_change_rows=66
  filter delta:
    BUILD_STATIC_DEFENSE=-46
    PRODUCE_ARMY=+26
    STAY_COURSE=+26
    ADD_GATEWAYS=-3
    TECH_ROBO=-3

Largest filter changes:
  RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY: 25
  RECOVERY BUILD_STATIC_DEFENSE -> STAY_COURSE: 10
  TECH_POWER BUILD_STATIC_DEFENSE -> STAY_COURSE: 7
  RECOVERY TECH_ROBO -> PRODUCE_ARMY: 4
  TECH_POWER TECH_ROBO -> STAY_COURSE: 4
```

Active-threat outcome follow-up:

```text
The narrow anti-air repair itself triggered only two STAY_COURSE rows:

  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> STAY_COURSE
  ready_static_low_minerals
  count=2
  +60s: threat_persisted=2/2
  +120s: threat_cleared=2/2

The remaining old anti-air fallback rows were still bad:

  ANTI_AIR_RESPONSE BUILD_STATIC_DEFENSE -> PRODUCE_ARMY
  ready_static_low_minerals
  count=2
  +60s: threat_persisted=2/2
  +120s: threat_persisted=2/2

However, the overall tactic-rule branch still failed badly. The larger blocker
is broader active-threat filtering, especially:

  RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY
  ready_static_low_minerals
  count=21

  RECOVERY / TECH_POWER pending_static_waiting -> STAY_COURSE
  still often leaves threat active at +60/+120s.
```

Decision:

```text
Do not promote tactic-rule.
Do not collect tactic-aware data.
Do not train action-outcome / veto / imitation models from this run.
Do not run PPO.

The anti-air narrow fix is not the main regression source, but the branch as a
whole is not viable online. Freeze runtime again. Next work should be offline
diagnostics of RECOVERY / TECH_POWER active-threat static-defense and TECH_ROBO
suppression before any new runtime change.
```

## 2026-06-25 Active-Threat Suppression Diagnostics

Implemented the requested offline suppression slice after the anti-air
ready-static A/B failed 0W/6L for tactic-rule.

Safety / scope:

```text
No SC2 launch.
No hidden-window guard run; it was not needed because no SC2 process was started.
No training / PPO.
No tactic-aware data collection.
Default --strategy-policy rule / --strategy-tactic-mode off unchanged.
No tactic metadata added to observation schema.
```

New files:

```text
rl\active_threat_suppression_diagnostics.py
scripts\diagnose_active_threat_suppression.py
tests\test_active_threat_suppression_diagnostics.py
```

Diagnostic command:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_active_threat_suppression.py data\trajectories\power_ab_anti_air_ready_static_no_filter_strategy_v1 data\trajectories\power_ab_anti_air_ready_static_tactic_strategy_v1 --show-files --show-timeline --json-output runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\active_threat_suppression.json --text-output runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\active_threat_suppression.txt
```

Artifacts:

```text
runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\active_threat_suppression.txt
runs\20260625_active_threat_suppression_anti_air_ready_static_v1\artifacts\active_threat_suppression.json
```

Inputs:

```text
data\trajectories\power_ab_anti_air_ready_static_no_filter_strategy_v1
data\trajectories\power_ab_anti_air_ready_static_tactic_strategy_v1
```

Source summary:

```text
no-filter:
  files=6
  result=2W / 2T / 2L
  target_suppression_rows=0
  filter_change_rows=0
  actions:
    ADD_GATEWAYS=33
    TECH_ROBO=34
    BUILD_STATIC_DEFENSE=26
    PRODUCE_ARMY=25
  threat_actions:
    BUILD_STATIC_DEFENSE=26
    PRODUCE_ARMY=1
    STAY_COURSE=7

tactic-rule:
  files=6
  result=0W / 0T / 6L
  target_suppression_rows=50
  filter_change_rows=66
  actions:
    ADD_GATEWAYS=19
    TECH_ROBO=6
    BUILD_STATIC_DEFENSE=19
    PRODUCE_ARMY=47
  threat_actions:
    BUILD_STATIC_DEFENSE=19
    PRODUCE_ARMY=27
    STAY_COURSE=35
```

Replay-only candidate impact:

```text
candidate: pass_through_before_action
affected_rows=50
immediate_candidate_executable_rows=6
action_delta:
  BUILD_STATIC_DEFENSE +42
  TECH_ROBO +8
  PRODUCE_ARMY -29
  STAY_COURSE -21
```

Main context outcomes:

```text
RECOVERY BUILD_STATIC_DEFENSE -> PRODUCE_ARMY
ready_static_low_minerals / ground_threat
count=18, candidate_executable=0/18
  +30s: threat_persisted=15/18, threat_cleared=3/18
  +60s: threat_persisted=10/18, threat_cleared=8/18
  +90s: threat_persisted=5/18,  threat_cleared=13/18
  +120s: threat_persisted=2/18, threat_cleared=16/18
  +120s averages: army_count_delta=-1.1, static_defense_delta=-0.8,
    ready_robo_delta=-0.7, observer_delta=-0.3, immortal_delta=0.0

TECH_POWER BUILD_STATIC_DEFENSE -> STAY_COURSE
pending_static_waiting / ground_threat
count=5, candidate_executable=3/5
  +30s: static_defense_increased=5/5, threat_persisted=5/5
  +60s: static_defense_increased=5/5, threat_persisted=3/5
  +90s: static_defense_increased=5/5, threat_persisted=3/5
  +120s: static_defense_increased=5/5, threat_persisted=3/5

RECOVERY BUILD_STATIC_DEFENSE -> STAY_COURSE
pending_static_waiting / ground_threat
count=4, candidate_executable=2/4
  +30s/+60s/+90s: threat_persisted=4/4
  +120s: threat_persisted=2/4, threat_cleared=2/4

RECOVERY TECH_ROBO -> PRODUCE_ARMY
first_robo_mineral_short / no_threat
count=4, candidate_executable=0/4
  no pending_robo_seen / ready_robo_seen / Observer / Immortal in +120s window

TECH_POWER TECH_ROBO -> STAY_COURSE
first_robo_mineral_short / no_threat
count=4, candidate_executable=0/4
  +60s: pending_robo_seen=4/4
  +120s: ready_robo_seen=4/4
  +120s: Observer=0/4, Immortal=0/4
```

Per-file target suppression distribution:

```text
001: 4 rows, first=491.4s
002: 10 rows, first=400.0s
003: 11 rows, first=262.9s
004: 10 rows, first=240.0s
005: 6 rows, first=308.6s
006: 9 rows, first=400.0s
```

Interpretation:

```text
The tactic-rule failure is still broad, not one clean runtime bug.

RECOVERY ready_static_low_minerals -> PRODUCE_ARMY is the largest bucket and is
bad early, but most rows clear by +120s. A global rule to always retain static
or always STAY_COURSE there is not justified.

pending_static_waiting -> STAY_COURSE does produce static defense, but threat
often persists through +120s. Waiting for pending static is not sufficient by
itself.

TECH_ROBO suppression rows are no-threat and mineral-short at the selected row.
TECH_POWER STAY_COURSE later reaches pending/ready Robo, but still lacks
Observer/Immortal payoff. RECOVERY TECH_ROBO -> PRODUCE_ARMY looks more
suspicious because it never reaches pending/ready Robo in the window, but the
rows are late, supply/resource constrained, and not immediately executable.
```

Decision:

```text
Do not patch runtime from this slice yet.
Do not promote tactic-rule.
Do not collect tactic-aware data.
Do not train action-outcome / veto / imitation models.
Do not run PPO.

Keep the branch frozen until a smaller replay-only candidate has a clearer
outcome gap and fewer side effects.
```

Validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_active_threat_suppression_diagnostics.py -q
  2 passed

.\.venv\Scripts\python.exe -m pytest tests\test_active_threat_suppression_diagnostics.py tests\test_active_threat_outcome_diagnostics.py tests\test_power_tactic_diagnostics.py tests\test_strategy_outcome_diagnostics.py -q
  14 passed

.\.venv\Scripts\python.exe -m pytest -q
  198 passed

.\.venv\Scripts\python.exe scripts\check_env.py
  OK

Get-Process SC2,SC2_x64 -ErrorAction SilentlyContinue
  no process output
```

## 2026-06-25 Strategy Replay Candidate Diagnostics

Scope:

```text
Continue P3 from doc\DEVELOPMENT_PLAN.md.
Offline only.
No SC2 launch.
No training / PPO.
No tactic-aware data collection.
Default rule/off runtime unchanged.
```

Implemented:

```text
rl\strategy_replay_candidate.py
scripts\diagnose_strategy_replay_candidate.py
tests\test_strategy_replay_candidate.py
```

Behavior:

```text
candidate_source=before_filter
  candidate_action = strategy_action_before_tactic_filter
  recorded_action = selected strategy_action / strategy_action_after_tactic_filter

For changed rows, the diagnostic reports:
  machine-readable gate_decision
  action delta
  immediate candidate executable rows
  candidate blocker when not executable
  grouped context/threat state
  +30/+60/+90/+120s recorded outcome slice
  per-file changed-row timeline
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_strategy_replay_candidate.py <strategy-trajectory-dir> --show-files --show-timeline --json-output <run-artifacts-dir>\strategy_replay_candidate.json --text-output <run-artifacts-dir>\strategy_replay_candidate.txt
```

Decision:

```text
This is a pre-runtime replay gate. It does not simulate the alternate future.
Use it to decide whether a proposed before-filter pass-through surface is:
  narrow
  mostly executable
  context-specific
  associated with a clear recorded +30/+60/+90/+120s outcome gap

Do not patch tactic-rule runtime unless this gate is favorable.
```

Validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_replay_candidate.py -q
  4 passed

.\.venv\Scripts\python.exe -m pytest tests\test_strategy_candidate_audit.py tests\test_strategy_replay_candidate.py -q
  8 passed

.\.venv\Scripts\python.exe -m pytest -q
  209 passed

.\.venv\Scripts\python.exe scripts\check_env.py
  OK

Get-Process SC2,SC2_x64 -ErrorAction SilentlyContinue
  no process output
```

## 2026-06-25 Anti-Air Ready-Static Replay Gate

Offline-only replay and promotion gate refresh for the frozen anti-air
ready-static A/B. No SC2 launch, no training, no PPO, no tactic-aware data
collection, and default rule/off runtime remains unchanged.

Inputs:

```text
baseline:
  data\trajectories\power_ab_anti_air_ready_static_no_filter_strategy_v1

candidate / replay:
  data\trajectories\power_ab_anti_air_ready_static_tactic_strategy_v1
```

Artifacts:

```text
runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\promotion_gate.json
runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\strategy_candidate_audit.txt
runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\strategy_replay_candidate.json
runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\strategy_replay_candidate.txt
```

Candidate audit:

```text
promotable: false
blocking_reasons:
  candidate_result_worse_than_baseline
  base_threat_rows_regressed
  add_gateways_count_regressed
  tech_robo_count_regressed
  build_static_defense_count_regressed

warnings:
  candidate_has_filter_changes

baseline result score: 0.333
candidate result score: 0.000
base_threat_rows: baseline=34 candidate=82 delta=48
filter_change_rows: baseline=0 candidate=66
```

Replay gate:

```text
gate_decision: hold_runtime_patch
runtime_patch_candidate: false
blocking_reasons:
  candidate_surface_too_broad
  candidate_executability_low
  largest_group_surface_too_broad
  largest_group_executability_low

changed_rows: 66
candidate_executable: 15/66
candidate_executable_ratio: 0.23
largest_group_count: 18
largest_group_executable_ratio: 0.00
```

Largest changed-row group:

```text
RECOVERY PRODUCE_ARMY -> BUILD_STATIC_DEFENSE
ready_static_low_minerals / ground_threat
count=18
candidate_executable=0/18
```

Decision:

```text
Do not patch tactic-rule runtime from this replay.
Do not promote tactic-rule.
Do not collect tactic-aware data.
Do not train action-outcome / veto / imitation models from this failed branch.
Do not run PPO.

Next work should use the replay gate to search for smaller, mostly executable
candidate groups, not broaden another hand-written tactic rule.
```

## 2026-06-25 Execution Observability And Candidate Audit

Scope:

```text
Continue from the route correction:
  freeze tactic-rule runtime
  keep default rule/off unchanged
  do not collect tactic-aware data
  do not train or start PPO
  implement observability and promotion gates offline first
```

Implemented P1 execution observability:

```text
bot\managers\strategy_executor.py
  StrategyExecutor.execute() now returns StrategyExecutionResult metadata.

bot\protoss_rule_bot.py
  Strategy trajectory rows now carry strategy_execution_* metadata.

rl\trajectory_recorder.py
  Strategy rows accept execution attempted/effect/blocker/unit/target fields.

rl\strategy_outcome_diagnostics.py
scripts\diagnose_strategy_outcomes.py
  Outcome diagnostics summarize execution effect/blocker counts.
```

Implemented P2 candidate audit:

```text
rl\strategy_candidate_audit.py
scripts\audit_strategy_candidate.py
tests\test_strategy_candidate_audit.py
```

Promotion gate behavior:

```text
Blocks when:
  candidate result score regresses
  base_threat_rows increases
  ADD_GATEWAYS count regresses
  TECH_ROBO count regresses
  BUILD_STATIC_DEFENSE count regresses
  strategy_execution_blocker counts increase

Warns when:
  candidate_has_filter_changes
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\audit_strategy_candidate.py <baseline-strategy-dir> <candidate-strategy-dir> --json-output <run-artifacts-dir>\promotion_gate.json --text-output <run-artifacts-dir>\strategy_candidate_audit.txt
```

Decision:

```text
Do not promote tactic-rule.
Do not collect tactic-aware data.
Do not train action-outcome / veto / imitation models yet.
Do not run PPO.

Next development step is P3 replay-only candidate:
  recorded trajectory -> candidate action -> changed rows -> executable rows
  -> +30/+60/+90/+120s outcome slice
```

Validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_candidate_audit.py -q
  3 passed

.\.venv\Scripts\python.exe -m pytest -q
  204 passed

.\.venv\Scripts\python.exe scripts\check_env.py
  OK

Get-Process SC2,SC2_x64 -ErrorAction SilentlyContinue
  no process output
```
