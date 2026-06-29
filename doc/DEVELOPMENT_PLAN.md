# Development Plan

This is the current execution plan for the next development stage. It supersedes
the loop of small tactic-rule patches followed by small Power-only A/B runs.

## 2026-06-29 Update: Next Development Operating Mode

Use the combined plan below for the next window/agent:

```text
time-boxed training loop
+ hard training trigger
+ candidate checkpoint discipline
```

This update is meant to prevent the project from drifting into endless gates and
diagnostics while avoiding the opposite mistake of training and promoting from
bad data.

### Current State

The engineering path is viable, but the old strategy data is not trainable.
The latest data-readiness result for:

```text
data\trajectories\power_ab_anti_air_ready_static_tactic_strategy_v1
```

is:

```text
recommendation: hold
training_ready: false
trajectory_detail_gate: hold
policy_explanation_gate: hold
observation_detail_gate: hold
```

Known old-data blockers:

```text
strategy_observation_details: missing on 330/330 non-terminal strategy rows
strategy_policy_source:      missing on 330/330 non-terminal strategy rows
strategy_policy_reason:      missing on 330/330 non-terminal strategy rows
```

Do not train from this old dataset. It is useful for diagnostics only.

### Chosen Operating Rules

1. Do not add more gates or analysis tools by default.
2. A new gate is allowed only when a real collection, training run, audit, or
   evaluation exposes a concrete failure that existing gates cannot catch.
3. Two consecutive development cycles must not both be "gate/diagnostic only".
4. Every cycle should produce at least one of:

```text
new trajectories
candidate training run
checkpoint audit
small SC2 evaluation
```

5. When readiness is green and the minimum row threshold is met, start a small
   candidate training run instead of delaying for more infrastructure.
6. Training output is always a candidate checkpoint first. It must not replace
   the default runtime automatically.

### Minimum Training Trigger

Collect a small but real new strategy dataset first. A reasonable first target:

```text
100-200 non-terminal strategy rows
```

Start a small strategy imitation run when all are true:

```text
rows >= 100
trajectory_detail_gate: ready
policy_explanation_gate: ready
observation_detail_gate: ready
training_readiness: train
```

If `training_ready=false`, fix only the reported blocker and collect again. Do
not expand the gate system speculatively.

### Required Pre-Training Check

Run the one-shot readiness pipeline on any newly collected strategy data:

```powershell
.\.venv\Scripts\python.exe scripts\run_strategy_data_readiness_pipeline.py <new-strategy-trajectory-dir> --output-dir runs --prefix strategy_data_readiness_latest --promotion-gate runs\strategy_promotion_gate_latest.json --text-output runs\strategy_data_readiness_latest.txt --fail-on-hold
```

Expected behavior:

```text
exit 0 only when training_ready=true
exit 1 when any gate still holds
```

The pipeline now includes:

```text
raw trajectory detail gate
policy explanation gate
emergency action analysis
observation detail gate
training readiness gate
```

### First Candidate Training Run

When the readiness pipeline is green, run a small candidate imitation training
job:

```powershell
.\.venv\Scripts\python.exe scripts\train_strategy_imitation.py <new-strategy-trajectory-dir> --run-root runs --run-name strategy_imitation_candidate_v1 --epochs 5 --batch-size 64 --class-weighting balanced --signal-filter trainable --observation-detail-gate runs\strategy_data_readiness_latest_observation_detail_gate.json
```

The result remains a candidate. Do not change the default runtime:

```text
--strategy-policy rule
--strategy-tactic-mode off
```

until checkpoint audits and small SC2 evaluations justify it.

### Candidate Checkpoint Discipline

After training, audit before any promotion. At minimum inspect:

```text
action distribution
signal audit
non-executable prediction rows
veto-negative matches
drop-non-executable matches
action critic fallback rows
baseline comparison
small SC2 evaluation
```

Promotion must be explicit and gate-backed. A candidate that trains successfully
is not automatically a better bot.

### Next Concrete Step

The next agent should prefer this order:

```text
1. Collect a small new strategy trajectory batch with current metadata.
2. Run the readiness pipeline.
3. If training_ready=true, run the small candidate imitation training.
4. If training_ready=false, fix only the reported blocker.
5. Audit the candidate before any runtime promotion.
```

This is the current priority unless the user explicitly asks for a different
task.

## Current Decision

Freeze the current tactic-rule runtime.

Do not promote `--strategy-tactic-mode rule`, do not collect tactic-aware
training data, do not train a tactic-aware imitation policy, and do not start PPO
from the current tactic branch.

The default runtime remains:

```text
--strategy-policy rule
--strategy-tactic-mode off
```

The next stage is about making strategy execution observable and auditable, not
about adding more hand-written tactic guardrails.

## Why

Recent guarded Power-only evidence shows that tactic-rule is not a promotable
branch:

```text
no-filter coverage-teacher: 2W / 2T / 2L
tactic-rule:                0W / 0T / 6L
```

Diagnostics show broad side effects rather than one clean bug:

```text
ADD_GATEWAYS suppressed
TECH_ROBO suppressed
BUILD_STATIC_DEFENSE suppressed
PRODUCE_ARMY and STAY_COURSE over-selected under threat
Robo/Observer coverage improved in places, but Immortal payoff and wins did not
```

The project should stop treating every bad bucket as a reason to patch
`rl/tactics.py`. A strategy action must first be traceable from intent, to
executor result, to +30/+60/+90/+120s outcome.

## Development Order

### P0: Stabilize

- Keep tactic-rule frozen.
- Keep default rule/off behavior unchanged.
- Preserve the passing test suite and environment check.
- Avoid destructive git operations while the current worktree contains many
  existing uncommitted changes.

### P1: Execution Observability

Add executor-level metadata for every strategy action:

```text
strategy_execution_attempted
strategy_execution_effect
strategy_execution_blocker
strategy_execution_unit_type
strategy_execution_target
```

Examples:

```json
{
  "strategy_action_name": "TECH_ROBO",
  "strategy_execution_attempted": true,
  "strategy_execution_effect": "build_structure",
  "strategy_execution_unit_type": "ROBOTICSFACILITY",
  "strategy_execution_target": "power_field"
}
```

```json
{
  "strategy_action_name": "PRODUCE_ARMY",
  "strategy_execution_attempted": true,
  "strategy_execution_effect": "delegate_train_army",
  "strategy_execution_blocker": "no_ready_robo"
}
```

This lets diagnostics separate these cases:

```text
action was not executable
action executed but payoff was delayed
action executed but downstream resources/supply blocked payoff
action was filtered before execution
```

### P2: Candidate Audit

Build a unified candidate audit script after execution metadata is available.

Current status: implemented for the first promotion gate. The first version is
intentionally narrow and offline-only: it compares baseline and candidate
strategy trajectory directories through `StrategyOutcomeDiagnostics`, then
checks result score, base-threat rows, key macro action counts, tactic filter
change warnings, and executor blocker deltas.

Script:

```text
scripts\audit_strategy_candidate.py
```

Inputs:

```text
baseline strategy trajectory dir
candidate strategy trajectory dir
```

Outputs:

```text
human-readable text report
machine-readable JSON audit
```

Example:

```powershell
.\.venv\Scripts\python.exe scripts\audit_strategy_candidate.py <baseline-strategy-dir> <candidate-strategy-dir> --json-output <run-artifacts-dir>\promotion_gate.json --text-output <run-artifacts-dir>\strategy_candidate_audit.txt
```

The result is machine-readable:

```json
{
  "promotable": false,
  "blocking_reasons": [
    "candidate_result_worse_than_baseline",
    "base_threat_rows_regressed",
    "tech_robo_count_regressed",
    "build_static_defense_count_regressed",
    "execution_blockers_increased"
  ]
}
```

Later versions can fold in tactic diagnostics, Power tactic diagnostics,
active-threat outcome diagnostics, and active-threat suppression diagnostics as
additional non-SC2 promotion-gate evidence. The first version deliberately keeps
the gate simple enough to run on any pair of strategy trajectory directories.

### P3: Replay Before Runtime

Before any new tactic runtime change, run a replay-only candidate:

```text
recorded trajectory -> candidate action -> changed rows -> executable rows -> outcome slice
```

Current status: first offline pass implemented for before-filter pass-through
replay candidates.

Script:

```text
scripts\diagnose_strategy_replay_candidate.py
```

Example:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_strategy_replay_candidate.py <strategy-trajectory-dir> --show-files --show-timeline --json-output <run-artifacts-dir>\strategy_replay_candidate.json --text-output <run-artifacts-dir>\strategy_replay_candidate.txt
```

The first mode reads `strategy_action_before_tactic_filter` as the candidate
action and compares it with the recorded selected action. It reports:

```text
machine-readable gate_decision
changed rows
candidate action delta
immediate candidate executable rows
per action/context lookahead outcomes
per-file changed-row timeline
```

Only consider a runtime patch when:

```text
changed rows are narrow
candidate rows are mostly executable
outcome gap is clear
side effects are bounded
default rule/off behavior stays unchanged
```

Current frozen anti-air ready-static replay result:

```text
trajectory:
  data\trajectories\power_ab_anti_air_ready_static_tactic_strategy_v1

artifacts:
  runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\strategy_replay_candidate.txt
  runs\20260625_strategy_replay_anti_air_ready_static_v1\artifacts\strategy_replay_candidate.json

gate_decision: hold_runtime_patch
runtime_patch_candidate: false
blocking_reasons:
  candidate_surface_too_broad
  candidate_executability_low
  largest_group_surface_too_broad
  largest_group_executability_low

changed_rows: 66
candidate_executable: 15/66
largest_group:
  RECOVERY PRODUCE_ARMY -> BUILD_STATIC_DEFENSE
  ready_static_low_minerals / ground_threat
  count=18
  candidate_executable=0/18
```

### P4: Learn Narrow Models Before Controllers

If learning is reintroduced, start with auxiliary models:

```text
action-outcome predictor
veto model
```

Do not train a new full strategy controller from failed tactic-aware labels.

### P5: PPO Later

PPO remains downstream of:

```text
Gymnasium SC2 environment
reward design
episode reset and launch isolation
baseline opponent matrix
safe launch harness
checkpoint adapter
```

## Promotion Gate

A candidate strategy/tactic change is not promotable unless it satisfies all of:

```text
result not worse than baseline
base_threat_rows not worse than baseline
first ADD_GATEWAYS not materially delayed
ADD_GATEWAYS count not materially lower
TECH_ROBO timing not materially delayed
Observer / Immortal payoff not lower
BUILD_STATIC_DEFENSE not broadly suppressed under threat
filter_change_rows are narrow and explainable
execution blockers do not increase in critical paths
```

## Current Implementation Step

P1 execution observability, P2 candidate audit, and P3 replay-only candidate
diagnostics with a machine-readable runtime-patch gate are implemented.

Existing strategy trajectories remain readable. New strategy trajectories carry
executor metadata that diagnostics and the candidate audit can summarize. The
frozen anti-air ready-static trajectory has now failed both the promotion audit
and the replay runtime-patch gate, so there is no runtime patch to make from
this surface.

Next development step:

```text
Keep tactic-rule frozen.
Use replay diagnostics to search for smaller candidate groups only.
Prefer action-outcome / veto diagnostics over another hand-written tactic rule.
Do not collect tactic-aware data or train from the failed candidate.
```

Current validation:

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
