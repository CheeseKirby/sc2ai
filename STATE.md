# Project State

Compact current state and experiment ledger.

## Current Snapshot

- Project root: `D:\opus\data\raw\alpaca-gpt4\sc2\sc2-ai-bot`
- Not a git repository.
- Current tests: `.\.venv\Scripts\python.exe -m pytest -q` -> `64 passed`.
- Environment check: `.\.venv\Scripts\python.exe scripts\check_env.py` -> OK.
- SC2 launches must use hidden-window guard.
- PPO is not implemented.
- Current learned-policy boundary: high-level army decisions only.

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

Useful next options:

1. Continue larger guarded evals for `imitation_v3_candidate`.
2. Improve retreat data/teacher if online `RETREAT_HOME` remains token-count.
3. Expand strategy capability using `STRATEGY_EXPANSION_PLAN.md` if the project goal is richer tech/army composition decisions.
4. Only consider PPO after stable learned baselines, reward design, and environment boundaries exist.

## Documentation

- `CODEX.md`: concise agent handoff.
- `README.md`: project overview and commands.
- `STATE.md`: this compact state ledger.
- `STRATEGY_EXPANSION_PLAN.md`: future strategy expansion plan.

