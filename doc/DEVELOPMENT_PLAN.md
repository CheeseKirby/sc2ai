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

## 2026-07-02 Update: Pre-Collapse Recovery Teacher Profile

The latest offline analysis showed that online-smoke-v2 failures are dominated
by missed pre-collapse recovery windows, especially:

```text
ThunderbirdLE / Hard Terran Power:
  late no-threat windows with high vespene, no Robo, TECH_ROBO executable

AcropolisLE / Hard Terran Power:
  late no-threat windows with low static defense, BUILD_STATIC_DEFENSE executable
```

An opt-in teacher profile now exists for targeted collection only:

```text
--strategy-policy coverage-teacher
--strategy-teacher-profile pre-collapse-recovery
```

It keeps the default `standard` teacher profile unchanged, and does not change
the default runtime:

```text
--strategy-policy rule
--strategy-tactic-mode off
```

Use it only for the next recovery-window data batch. Before launching SC2, keep
using the hidden-window guard:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Targeted collection command used:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --run-root runs --run-name strategy_pre_collapse_recovery_teacher_batch_v1 --maps ThunderbirdLE AcropolisLE --difficulties Hard --opponents Terran --ai-builds Power --games-per-combo 2 --strategy-policy coverage-teacher --strategy-teacher-profile pre-collapse-recovery --strategy-trajectory-dir data\trajectories\strategy_pre_collapse_recovery_teacher_batch_strategy_v1 --record-decision-interval 16 --game-time-limit 900
```

The collected data passed the existing readiness and pre-collapse recovery
gates:

```text
new targeted data:
  training_ready: true
  pre-collapse recovery: ready, missed=0/15

combined fresh + threat + anti-air + pre-collapse:
  training_ready: true
  pre-collapse recovery: ready, missed=0/36
```

Two small imitation candidates trained from the combined data are still
hold-only:

```text
runs\20260702_105359_20260702_strategy_imitation_pre_collapse_recovery_teacher_cap2_v1
runs\20260702_105447_20260702_strategy_imitation_pre_collapse_recovery_teacher_trainable_cap2_v1
```

Both failed offline executable-mask signal audits on the online-smoke-v2
regression surface:

```text
accept_positive_match: 0/6
veto_negative_match: 4/4
action_space_exhausted_match: 2/2
```

Do not promote either checkpoint. Do not run an online smoke from these
candidates. The next iteration should add a checkpoint-level recovery-window
audit/slice or adjust the supervised objective so the model can preserve
online-smoke-v2 accept-positive recovery labels while still avoiding
veto/action-space bad labels.

## 2026-07-02 Update: Recovery Accept-Positive Loss Weight

The checkpoint-level recovery audit now exists and is the preferred offline
gate before any online checkpoint smoke:

```text
rl\strategy_checkpoint_pre_collapse_recovery_audit.py
scripts\audit_strategy_checkpoint_pre_collapse_recovery.py
tests\test_strategy_checkpoint_pre_collapse_recovery_audit.py
```

It showed that the newest checkpoints are not missing every pre-collapse
recovery window. The sharper blocker is recovery `accept_positive` preservation,
especially `BUILD_STATIC_DEFENSE` positives.

A new opt-in supervised objective knob was added:

```text
--recovery-accept-positive-loss-weight <float>
```

Contract:

```text
default: 1.0, no behavior change
requires a signal filter
weights only observed accept_positive rows for:
  TECH_ROBO
  PRODUCE_ARMY
  BUILD_STATIC_DEFENSE
does not synthesize labels
does not change default runtime
```

Small grid:

```text
w2:
  run: runs\20260702_151609_20260702_strategy_imitation_recovery_accept_positive_w2_cap2_v1
  train_accuracy: 0.176
  validation_accuracy: 0.206
  online-smoke-v2 signal audit:
    accept_positive_match: 1/6
    veto_negative_match: 4/4
    action_space_exhausted_match: 2/2
  online-smoke-v2 recovery audit:
    accept_positive_recovery_match: 1/3
  combined recovery audit:
    accept_positive_recovery_match: 19/54

w3:
  run: runs\20260702_151658_20260702_strategy_imitation_recovery_accept_positive_w3_cap2_v1
  train_accuracy: 0.140
  validation_accuracy: 0.191
  online-smoke-v2 signal audit:
    accept_positive_match: 1/6
    veto_negative_match: 0/4
    action_space_exhausted_match: 2/2
  online-smoke-v2 recovery audit:
    accept_positive_recovery_match: 1/3
  combined recovery audit:
    accept_positive_recovery_match: 20/54

w4:
  run: runs\20260702_151509_20260702_strategy_imitation_recovery_accept_positive_w4_cap2_v1
  train_accuracy: 0.136
  validation_accuracy: 0.176
  online-smoke-v2 signal audit:
    accept_positive_match: 1/6
    veto_negative_match: 0/4
    action_space_exhausted_match: 2/2
  online-smoke-v2 recovery audit:
    accept_positive_recovery_match: 1/3
  combined recovery audit:
    accept_positive_recovery_match: 20/54
```

Decision:

```text
Do not promote w2, w3, or w4.
Do not run online smoke from these candidates.
Do not change default runtime.
Do not run PPO.

w3/w4 are useful evidence because they clear online-smoke-v2 veto_negative
matches under executable-mask prediction. They are still hold-only because
action_space_exhausted_match remains 2/2 and recovery accept-positive match
remains 1/3 on online-smoke-v2.
```

Next concrete experiment:

```text
Do not keep increasing the global recovery-positive scalar weight.

Build a more focused recovery-positive preservation slice/objective:
  preserve BUILD_STATIC_DEFENSE accept_positive rows
  preserve TECH_ROBO rows only when first-robo/high-vespene/no-robo context fits
  keep veto_negative/action_space bad-label avoidance strict
  audit with checkpoint_pre_collapse_recovery before any online smoke
```

## 2026-07-02 Update: Action-Specific Recovery Accept-Positive Weights

The focused follow-up added per-action overrides for observed recovery
accept-positive rows:

```text
--recovery-accept-positive-action-loss-weight ACTION=WEIGHT
```

Contract:

```text
default: no action-specific overrides
requires a signal filter
valid actions:
  TECH_ROBO
  PRODUCE_ARMY
  BUILD_STATIC_DEFENSE
weights must be >= 1.0
action overrides replace the global recovery accept-positive weight
weights only observed accept_positive recovery rows
does not synthesize labels
does not change default runtime
```

Three candidates were trained from the combined fresh + threat + anti-air +
pre-collapse data with:

```text
--signal-filter trainable-recovery-safe
--max-drop-ambiguous-per-positive 2.0
--class-weighting balanced
```

Results:

```text
static-only:
  run: runs\20260702_152639_20260702_strategy_imitation_static_recovery_accept_w4_cap2_v1
  weights:
    BUILD_STATIC_DEFENSE: 4.0
  train_accuracy: 0.357
  validation_accuracy: 0.265
  online-smoke-v2 signal:
    accept_positive_match: 4/6
    veto_negative_match: 4/4
    action_space_exhausted_match: 2/2
  online-smoke-v2 recovery:
    pre-collapse executable recovery: 2/6
    accept_positive_recovery_match: 2/3

static + tech w3:
  run: runs\20260702_152738_20260702_strategy_imitation_static_w4_tech_w3_recovery_accept_cap2_v1
  weights:
    BUILD_STATIC_DEFENSE: 4.0
    TECH_ROBO: 3.0
  train_accuracy: 0.125
  validation_accuracy: 0.162
  online-smoke-v2 signal:
    accept_positive_match: 1/6
    veto_negative_match: 3/4
    action_space_exhausted_match: 2/2
  online-smoke-v2 recovery:
    pre-collapse executable recovery: 6/6
    accept_positive_recovery_match: 1/3

static + tech w4:
  run: runs\20260702_152837_20260702_strategy_imitation_static_w4_tech_w4_recovery_accept_cap2_v1
  weights:
    BUILD_STATIC_DEFENSE: 4.0
    TECH_ROBO: 4.0
  train_accuracy: 0.331
  validation_accuracy: 0.309
  online-smoke-v2 signal:
    accept_positive_match: 1/6
    veto_negative_match: 4/4
    action_space_exhausted_match: 2/2
  online-smoke-v2 recovery:
    pre-collapse executable recovery: 6/6
    accept_positive_recovery_match: 1/3
  combined signal:
    accept_positive_match: 20/112
    veto_negative_match: 7/13
    drop_non_executable_match: 0/452
    action_space_exhausted_match: 23/23
  combined recovery:
    accept_positive_recovery_match: 19/54
```

Decision:

```text
Do not promote any of these checkpoints.
Do not run online smoke from these candidates.
Do not change default runtime.
Do not run PPO.

Action-specific weighting is useful as a diagnostic knob but is not sufficient.
Static-only weighting improves BUILD_STATIC_DEFENSE positive preservation, but
loses Thunderbird TECH_ROBO pre-collapse recovery. Static+tech weighting restores
TECH_ROBO recovery windows, but loses the static-defense gain and still fails
veto/action-space gates.
```

Next concrete experiment:

```text
Stop increasing scalar weights as the main line.

Build a context-aware recovery preservation objective or sampler:
  static-defense positives only in no-static/low-static, affordable,
  threat/static-floor contexts
  TECH_ROBO positives only in first-robo/high-vespene/no-robo contexts
  do not reward TECH_ROBO labels as static-defense predictions, or the reverse
  keep strict gates on veto_negative, action_space_exhausted, and
  drop_non_executable rows

Required offline success before any online smoke:
  improve online-smoke-v2 recovery accept-positive preservation
  preserve all pre-collapse executable recovery windows
  improve or at least not regress veto/action-space bad-label gates
```

## 2026-07-02 Update: Context-Aware Recovery Preservation

The next follow-up made the recovery-positive objective context-aware instead
of simply increasing scalar weights.

New opt-in knobs:

```text
--recovery-accept-positive-context-filter pre-collapse-recovery
--recovery-accept-positive-context-oversample-factor <int>
```

Contract:

```text
default context filter: off
default context oversample factor: 1
requires a signal filter
weights and oversamples only observed accept_positive recovery rows
does not synthesize labels
does not change default runtime
```

The `pre-collapse-recovery` context filter matches the bounded recovery-teacher
semantics:

```text
TECH_ROBO:
  late high-vespene, no Robo, no immediate base threat

BUILD_STATIC_DEFENSE:
  late low-static-defense, affordable static defense, no immediate base threat

PRODUCE_ARMY:
  supply/resource available with idle production or underbuilt army
```

Three candidates were trained with `BUILD_STATIC_DEFENSE=4.0` and
`TECH_ROBO=4.0` action weights:

```text
context-only:
  run: runs\20260702_153843_20260702_strategy_imitation_context_recovery_static_w4_tech_w4_cap2_v1
  context_oversample_factor: 1
  examples: 340
  train_accuracy: 0.588
  validation_accuracy: 0.485
  online-smoke-v2 signal:
    accept_positive_match: 1/6
    veto_negative_match: 4/4
    action_space_exhausted_match: 2/2
  online-smoke-v2 recovery:
    missed_checkpoint_pre_collapse_recovery_rows: 0/6
    accept_positive_recovery_match: 1/3
  combined recovery:
    missed_checkpoint_pre_collapse_recovery_rows: 0/36
    accept_positive_recovery_match: 16/54

context x3:
  run: runs\20260702_154310_20260702_strategy_imitation_context_recovery_x3_static_w4_tech_w4_cap2_v1
  context_oversample_factor: 3
  examples: 350
  train_accuracy: 0.361
  validation_accuracy: 0.457
  online-smoke-v2 signal:
    accept_positive_match: 0/6
    veto_negative_match: 1/4
    action_space_exhausted_match: 2/2
  online-smoke-v2 recovery:
    missed_checkpoint_pre_collapse_recovery_rows: 0/6
    accept_positive_recovery_match: 0/3
  combined recovery:
    missed_checkpoint_pre_collapse_recovery_rows: 0/36
    accept_positive_recovery_match: 19/54

context x10:
  run: runs\20260702_154218_20260702_strategy_imitation_context_recovery_x10_static_w4_tech_w4_cap2_v1
  context_oversample_factor: 10
  examples: 385
  train_accuracy: 0.614
  validation_accuracy: 0.688
  online-smoke-v2 signal:
    accept_positive_match: 2/6
    veto_negative_match: 4/4
    action_space_exhausted_match: 2/2
  online-smoke-v2 recovery:
    missed_checkpoint_pre_collapse_recovery_rows: 0/6
    accept_positive_recovery_match: 2/3
  combined recovery:
    missed_checkpoint_pre_collapse_recovery_rows: 6/36
    accept_positive_recovery_match: 12/54
```

Decision:

```text
Do not promote context-only, context x3, or context x10.
Do not run online smoke from these candidates.
Do not change default runtime.
Do not run PPO.

Context-aware weighting/sampling is now implemented and tested, but current
data is too sparse for this knob to solve the policy. The context-only run has
only 5 matched recovery accept-positive rows, and only 3 receive the
action-specific weight. Oversampling changes the error shape instead of clearing
the gate: x3 improves online-smoke-v2 veto matches but loses recovery positives,
while x10 improves online recovery positives but regresses the combined
pre-collapse gate and over-shifts toward BUILD_STATIC_DEFENSE.
```

Next concrete experiment:

```text
Prefer more targeted evidence over another scalar/sampler sweep.

Collect or construct matched context-positive rows for both:
  TECH_ROBO in late high-vespene/no-robo/no-threat windows
  BUILD_STATIC_DEFENSE in late low-static/no-threat/affordable windows

Then evaluate by explicit slices:
  TECH_ROBO context preservation
  BUILD_STATIC_DEFENSE context preservation
  recovery action confusion matrix
  veto_negative/action_space/drop_non_executable bad-label matches

Do not run online smoke until the offline online-smoke-v2 surface improves
without regressing the combined pre-collapse gate.
```

## 2026-07-02 Update: Recovery Context Slice Audit

The context-aware training runs showed a new failure mode: changing the
recovery-positive objective can preserve one recovery action while stealing
another action's positives. A dedicated audit now makes that visible:

```text
rl\strategy_recovery_context_audit.py
scripts\audit_strategy_recovery_context.py
tests\test_strategy_recovery_context_audit.py
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\audit_strategy_recovery_context.py <inputs> --checkpoint <policy.pt> --prediction-mode executable-mask --json-output <artifact>.json --text-output <artifact>.txt
```

The audit only inspects observed `accept_positive` recovery rows, then splits
them by whether the recorded action matches the `pre-collapse-recovery` context
filter. It reports:

```text
overall recovery accept-positive match
context-matched recovery accept-positive match
context-matched misses
context-matched cross-action confusion
per-action confusion:
  TECH_ROBO
  PRODUCE_ARMY
  BUILD_STATIC_DEFENSE
```

Combined fresh + threat + anti-air + pre-collapse results:

```text
context-only:
  run: runs\20260702_153843_20260702_strategy_imitation_context_recovery_static_w4_tech_w4_cap2_v1
  overall recovery accept-positive: 16/54
  context-matched recovery accept-positive: 2/5
  context-matched cross-action confusion: 2/5
  key confusion:
    TECH_ROBO -> BUILD_STATIC_DEFENSE: 1
    PRODUCE_ARMY -> BUILD_STATIC_DEFENSE: 1

context x3:
  run: runs\20260702_154310_20260702_strategy_imitation_context_recovery_x3_static_w4_tech_w4_cap2_v1
  overall recovery accept-positive: 19/54
  context-matched recovery accept-positive: 2/5
  context-matched cross-action confusion: 3/5
  key confusion:
    TECH_ROBO -> BUILD_STATIC_DEFENSE: 1
    PRODUCE_ARMY -> BUILD_STATIC_DEFENSE: 2

context x10:
  run: runs\20260702_154218_20260702_strategy_imitation_context_recovery_x10_static_w4_tech_w4_cap2_v1
  overall recovery accept-positive: 12/54
  context-matched recovery accept-positive: 4/5
  context-matched cross-action confusion: 1/5
  key confusion:
    PRODUCE_ARMY -> BUILD_STATIC_DEFENSE: 1
```

Online-smoke-v2 recovery accept-positive rows do not match the
`pre-collapse-recovery` context filter:

```text
context_matched_accept_positive_recovery_rows: 0
warning: no_context_matched_accept_positive_recovery_rows
```

That regression surface remains important for veto/action-space failures, but
it cannot prove the matched pre-collapse context objective.

Decision:

```text
Do not promote any context-aware checkpoint.
Do not run online smoke from these candidates.
Do not change default runtime.
Do not run PPO.
```

Updated next concrete experiment:

```text
Stop sampler-only sweeps until more context evidence exists.

Before the next checkpoint can be considered for online smoke, require:
  combined pre-collapse recovery gate does not regress
  online-smoke-v2 veto/action-space bad-label matches improve
  context-matched recovery accept-positive misses are 0
  context-matched cross-action confusion is 0

Preferred data/training work:
  add matched TECH_ROBO rows for late high-vespene/no-robo/no-threat contexts
  add matched BUILD_STATIC_DEFENSE rows for late low-static/no-threat contexts
  keep PRODUCE_ARMY recovery rows separate so they are not absorbed by static
  defense pressure
```

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

2026-06-30 progress:

```text
A small fresh metadata batch was collected with current strategy row metadata:
  data\trajectories\strategy_fresh_metadata_batch_strategy_v1
  runs\20260630_095925_strategy_fresh_metadata_batch_v1

The readiness pipeline passed:
  recommendation: train
  training_ready: true
  trajectory_detail_gate: ready
  policy_explanation_gate: ready
  observation_detail_gate: ready

Small candidate training was run:
  trainable-filter imitation:
    runs\20260630_100315_strategy_imitation_fresh_metadata_candidate_v1
    examples=87, train_accuracy=0.500, validation_accuracy=0.353

  strict-positive imitation:
    runs\20260630_100437_strategy_imitation_fresh_metadata_strict_positive_v1
    examples=11, train_accuracy=1.000, validation_accuracy=0.000

Fresh action critics were trained:
  trainable critic:
    runs\20260630_100545_strategy_action_critic_fresh_metadata_trainable_v1
    validation accuracy/precision/recall: 0.919 / 0.909 / 0.952

  conservative critic:
    runs\20260630_100545_strategy_action_critic_fresh_metadata_conservative_v1
    validation accuracy/precision/recall: 1.000 / 1.000 / 1.000

A larger v2 fresh metadata batch was then collected:
  data\trajectories\strategy_fresh_metadata_batch_strategy_v2
  runs\20260630_143709_strategy_fresh_metadata_batch_v2

Coverage:
  maps: AcropolisLE, ThunderbirdLE
  opponent: Hard Terran
  ai_builds: Rush, Timing, Power, Macro, Air
  games: 10
  results: 4 Victory / 2 Defeat / 4 Tie
  note: Power lost on both maps

The v2 readiness pipeline passed:
  recommendation: train
  training_ready: true
  trajectory_detail_gate: ready
  policy_explanation_gate: ready
  observation_detail_gate: ready

V2 candidate training was run:
  trainable-filter imitation:
    runs\20260630_144517_strategy_imitation_fresh_metadata_batch_v2_trainable
    examples=285, train_accuracy=0.996, validation_accuracy=0.982

  strict-positive imitation:
    runs\20260630_144517_strategy_imitation_fresh_metadata_batch_v2_strict_positive
    examples=2, train_accuracy=1.000, validation_accuracy=1.000

  trainable critic:
    runs\20260630_144517_strategy_action_critic_fresh_metadata_batch_v2_trainable
    examples=514, validation accuracy/precision/recall: 0.990 / 0.977 / 1.000

  conservative critic:
    runs\20260630_144517_strategy_action_critic_fresh_metadata_batch_v2_conservative
    examples=231, validation accuracy/precision/recall: 0.957 / 1.000 / 0.957
```

Current learning decision:

```text
Do not promote any fresh imitation checkpoint.
Do not run an online checkpoint eval yet.
Do not change default runtime.

The training path is now unblocked, but the first fresh batch is too small and
too imbalanced:
  trainable-filter imitation is dominated by ambiguous STAY_COURSE rows.
  strict-positive imitation has only 11 examples and overfits.
  action-critic masks remove non-executable predictions, but fallback too often
  and still collapse heavily toward STAY_COURSE.

The v2 batch confirms the data/label problem rather than solving it:
  trainable-filter imitation has high validation accuracy but predicts
  STAY_COURSE for all 514 audited rows.
  strict-positive imitation has only 2 examples and predicts PRODUCE_ARMY for
  all rows before masking, causing 467/514 non-executable raw predictions.
  trainable action critic is strong as a safety detector, but threshold sweep
  still recommends hold because executable candidates are fully vetoed in some
  rows and accept-positive coverage is too low.

The ambiguous-cap iteration makes the training failure more explicit:
  --max-drop-ambiguous-per-positive was added to train_strategy_imitation.py.
  It caps drop_ambiguous rows to a deterministic multiple of accept_positive
  rows and records the balance in the training summary.

Combined v1+v2 cap experiments:
  cap2:
    runs\20260630_145606_strategy_imitation_fresh_metadata_combined_cap2_v1
    examples=39
    kept_by_training_use: accept_positive=13, drop_ambiguous=26
    raw audit: accept_positive_match=11/13 but predicted_non_executable=214/699
    action-critic sweep selected threshold=0.800, fallback=lowest-risk
    accept_positive_match=11/13, predicted_non_executable=0,
    drop_non_executable_match=1/324, fallback_rows=38

  cap0:
    runs\20260630_145606_strategy_imitation_fresh_metadata_combined_cap0_v1
    examples=13
    kept_by_training_use: accept_positive=13
    action-critic sweep selected threshold=0.800, fallback=lowest-risk
    accept_positive_match=12/13, predicted_non_executable=0,
    drop_non_executable_match=1/324, fallback_rows=38

  combined trainable critic:
    runs\20260630_145624_strategy_action_critic_fresh_metadata_combined_trainable_v1
    validation accuracy/precision/recall: 0.986 / 0.983 / 0.983

  combined trainable v2-schema critic:
    runs\20260630_145934_strategy_action_critic_fresh_metadata_combined_trainable_v2_schema
    validation accuracy/precision/recall: 0.986 / 0.983 / 0.983
    worse fallback than v1 schema in the cap2 audit

Executor-aligned ADD_GATEWAYS replay logic now uses army_observation.gateways
when estimating the executor gateway cap. That removed the remaining bad
ADD_GATEWAYS / target_gateways_reached drop match:
  predicted_non_executable=0/699
  bad_recorded_match=0/327
  veto_negative_match=0/3
  drop_non_executable_match=0/324

The safe-fallback gate now separates total fallback rows from unsafe fallback
rows. With executor-aligned ADD_GATEWAYS and FORGE_UPGRADES checks, cap2 +
trainable v1 action critic now passes the offline gate:
  selected: threshold=0.950 fallback=first-executable
  accept_positive_match=11/13
  predicted_non_executable=0
  veto/drop matches=0
  fallback_rows=32
  unsafe_fallback_rows=0
  recommendation=promotion_candidate

Runtime action-critic masking is implemented for explicit checkpoint eval via:
  --strategy-action-critic-checkpoint
  --strategy-action-critic-threshold
  --strategy-action-critic-fallback-policy

Two small online Hard Terran Power smokes ran with the masked checkpoint. Both
returned cleanly but tied:
  runs\20260630_152526_strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_v1
  runs\20260630_153145_strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_v2

The v1 smoke exposed FORGE_UPGRADES executor mismatch
noop / no_affordable_upgrade. The runtime-aligned v2 smoke fixed that class:
  BUILD_STATIC_DEFENSE build_structure=3
  FORGE_UPGRADES build_structure=2, research_upgrade=1
  no repeated FORGE_UPGRADES noop

The v2 smoke is still not promotable:
  results: 2 Tie
  actions: STAY_COURSE=118, FORGE_UPGRADES=3, BUILD_STATIC_DEFENSE=3
  online self-audit signal_healthy=false
  veto_negative_match=4/4
  action_space_exhausted_match=2/2
  unsafe_fallback_rows=40
```

Next development step:

```text
Keep tactic-rule frozen.
Keep default rule/off behavior unchanged.

Do not promote the checkpoint or change default runtime. The offline gate now
passes, but the online smoke gate holds.

Keep the ambiguous cap in the training path. It prevents STAY_COURSE-heavy
drop_ambiguous rows from silently dominating trainable imitation, but it does
not by itself make a promotable checkpoint.

Use online smoke v2 as the next regression surface. Reduce unsafe fallback and
STAY_COURSE under active threat without reintroducing executor noops.

Collect or construct data that specifically increases accept_positive coverage.
Prioritize:
  TECH_ROBO
  ADD_GATEWAYS
  BUILD_STATIC_DEFENSE
  PRODUCE_ARMY
  EXPAND
  BOOST_WORKERS
  Hard Terran Power losses

Then retrain or re-sweep:
  strategy imitation with the ambiguous cap enabled
  action critic with threshold sweep

Only run broader online checkpoint evaluation after:
  online self-audit has no veto/action_space matches
  unsafe_fallback_rows is near zero
  non-STAY actions execute with useful payoff
  Hard Terran Power improves beyond Tie
```

Current validation:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_strategy_candidate_audit.py -q
4 passed

.\.venv\Scripts\python.exe -m pytest tests\test_strategy_replay_candidate.py -q
5 passed

.\.venv\Scripts\python.exe -m pytest tests\test_strategy_replay_candidate.py tests\test_strategy_checkpoint_signal_audit.py tests\test_strategy_action_critic_threshold_sweep.py -q
22 passed

.\.venv\Scripts\python.exe -m pytest tests\test_strategy_replay_candidate.py tests\test_strategy_checkpoint_signal_audit.py tests\test_strategy_action_critic_threshold_sweep.py tests\test_rl_strategy_policy.py tests\test_evaluate.py -q
39 passed

.\.venv\Scripts\python.exe -m pytest -q
323 passed

.\.venv\Scripts\python.exe scripts\check_env.py
OK

Get-Process SC2,SC2_x64 -ErrorAction SilentlyContinue
no process output
```

## 2026-06-30 Online-Smoke Critic Refresh Follow-Up

The next regression surface was exercised without changing the default runtime:

```text
data\trajectories\strategy_fresh_metadata_batch_strategy_v1
data\trajectories\strategy_fresh_metadata_batch_strategy_v2
data\trajectories\strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_strategy_v2
```

Readiness was rerun for the exact three-input training set:

```text
runs\strategy_data_readiness_combined_plus_online_smoke_v2_summary.json
runs\strategy_data_readiness_combined_plus_online_smoke_v2_observation_detail_gate.json
recommendation: train
training_ready: true
trajectory_detail_gate: ready
policy_explanation_gate: ready
observation_detail_gate: ready
```

Two refreshed action critics were trained from the original fresh metadata plus
the online smoke v2 rows:

```text
v1 feature schema:
  run: runs\20260630_154211_strategy_action_critic_combined_plus_online_smoke_v2_trainable_v1
  examples: 821
  unsafe examples: 332
  validation accuracy/precision/recall: 0.988 / 0.972 / 1.000

v2 threat/action-interaction feature schema:
  run: runs\20260630_154556_strategy_action_critic_combined_plus_online_smoke_v2_trainable_schema_v2_v1
  examples: 821
  unsafe examples: 332
  validation accuracy/precision/recall: 0.988 / 0.972 / 1.000
```

Sweep results against the original combined fresh metadata surface:

```text
v1-schema refreshed critic:
  artifact:
    runs\20260630_145606_strategy_imitation_fresh_metadata_combined_cap2_v1\artifacts\action_critic_threshold_sweep_combined_plus_online_smoke_v2_critic_on_original_training.json
  recommendation: promotion_candidate
  selected: threshold=0.400 fallback=first-executable
  accept_positive_match=5/13
  veto_negative_match=0/3
  drop_non_executable_match=0/324
  fallback_rows=38
  unsafe_fallback_rows=0

v2-schema refreshed critic:
  artifact:
    runs\20260630_145606_strategy_imitation_fresh_metadata_combined_cap2_v1\artifacts\action_critic_threshold_sweep_combined_plus_online_smoke_v2_schema_v2_critic_on_original_training.json
  recommendation: promotion_candidate
  selected: threshold=0.800 fallback=first-executable
  accept_positive_match=7/13
  veto_negative_match=0/3
  drop_non_executable_match=0/324
  fallback_rows=1
  unsafe_fallback_rows=0
```

Sweep results against the online smoke v2 regression surface:

```text
v1-schema refreshed critic:
  artifact:
    runs\20260630_153145_strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_v2\artifacts\action_critic_threshold_sweep_combined_plus_online_smoke_v2_critic_on_online_smoke.json
  recommendation: hold
  selected: threshold=0.400 fallback=rule-safe
  blocking_reasons:
    predicted_matches_action_space_exhausted_labels
    action_critic_all_executable_candidates_vetoed
  accept_positive_match=3/5
  veto_negative_match=0/4
  drop_non_executable_match=0/1
  fallback_rows=8
  unsafe_fallback_rows=8

v2-schema refreshed critic:
  artifact:
    runs\20260630_153145_strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_v2\artifacts\action_critic_threshold_sweep_combined_plus_online_smoke_v2_schema_v2_critic_on_online_smoke.json
  recommendation: hold
  selected: threshold=0.200 fallback=rule-safe
  blocking_reasons:
    predicted_matches_action_space_exhausted_labels
    action_critic_all_executable_candidates_vetoed
  accept_positive_match=3/5
  veto_negative_match=0/4
  drop_non_executable_match=0/1
  fallback_rows=8
  unsafe_fallback_rows=8
```

Detailed online audit for the v1 refreshed critic:

```text
artifact:
  runs\20260630_153145_strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_v2\artifacts\checkpoint_signal_audit_combined_plus_online_smoke_v2_critic_t04_rule_safe_online_smoke.json

raw_predicted_non_executable=55/124
masked predicted_non_executable=0/124
action_critic_fallback_rows=8
action_critic_safe_fallback_rows=0
action_critic_unsafe_fallback_rows=8
action_space_exhausted_match=2/2
```

The remaining failures are concentrated late in Hard Terran Power smoke games
around 650-700 seconds under air-and-ground threat. Several rows have only
`STAY_COURSE` or `STAY_COURSE` plus `BOOST_WORKERS` executable; in Thunderbird,
`rule-safe` falls back to `TECH_ROBO` after all executable candidates are vetoed.
This is no longer primarily a threshold-selection problem.

Decision:

```text
Do not promote any refreshed critic.
Do not promote the cap2 strategy checkpoint.
Do not change default runtime.
Do not run PPO.

The refreshed critics improve the online smoke symptom by reducing unsafe
fallback from 40 rows to 8 rows, and the v2-schema critic is the stronger
offline safety candidate. They still cannot pass the online smoke gate because
the strategy action space is exhausted under late active threat and fallback is
forced to pick a vetoed action.
```

Next concrete work:

```text
Keep online smoke v2 as a regression surface.
Treat action_space_exhausted as an action-space / executable-response gap.
Collect or construct positive threat-state rows where TECH_ROBO, PRODUCE_ARMY,
BUILD_STATIC_DEFENSE, BOOST_WORKERS, EXPAND, or ADD_GATEWAYS are executable and
produce payoff before the bot reaches the late all-candidates-vetoed state.
Consider a narrowly scoped emergency-response action or executor path only after
offline replay shows that it is executable and bounded.
Do not spend another cycle only tuning action-critic thresholds unless new data
or a new executable action surface changes the candidate set.
```

## 2026-06-30 Threat-Positive Teacher Batch Follow-Up

A focused teacher batch was collected to add positive threat-state examples
rather than tuning the existing critic again.

Collection:

```text
run:
  runs\20260630_155608_strategy_threat_positive_teacher_batch_v1
strategy trajectories:
  data\trajectories\strategy_threat_positive_teacher_batch_strategy_v1
army trajectories:
  data\trajectories\strategy_threat_positive_teacher_batch_army_v1

grid:
  maps: AcropolisLE, ThunderbirdLE
  difficulty: Hard
  opponent: Terran
  ai_builds: Power, Air
  games_per_combo: 1
  game_time_limit: 700
  army_policy: coverage-teacher
  strategy_policy: coverage-teacher
  strategy_tactic_mode: off

results:
  4 Tie
  all return_code=0
```

Readiness and signal summary:

```text
readiness:
  artifact:
    runs\20260630_155608_strategy_threat_positive_teacher_batch_v1\artifacts\strategy_data_readiness_threat_positive_teacher_batch_v1_summary.json
  recommendation: train
  training_ready: true
  trajectory_detail_gate: ready
  policy_explanation_gate: ready
  observation_detail_gate: ready

recorded signal dataset:
  artifact:
    runs\20260630_155608_strategy_threat_positive_teacher_batch_v1\artifacts\strategy_signal_dataset_recorded_only_threat_positive_teacher_batch_v1.json
  rows: 248
  accept_positive: 16
  drop_ambiguous: 150
  drop_non_executable: 81
  veto_negative: 1
  accept_positive actions:
    ADD_GATEWAYS=4
    BUILD_STATIC_DEFENSE=4
    FORGE_UPGRADES=3
    PRODUCE_ARMY=3
    TECH_ROBO=2
```

Action-space and emergency-response findings:

```text
action-space artifact:
  runs\20260630_155608_strategy_threat_positive_teacher_batch_v1\artifacts\strategy_action_space_threat_positive_teacher_batch_v1.json

emergency artifact:
  runs\20260630_155608_strategy_threat_positive_teacher_batch_v1\artifacts\strategy_emergency_actions_threat_positive_teacher_batch_v1.json

only_stay_course: 96/248
only_stay_course_under_threat: 7/248
action_space_exhausted: 0/7 threatened only-STAY rows
addressable_threatened_only_stay_course by EMERGENCY_DEFEND: 4/7
unaddressed threatened only-STAY rows: 3/7
unaddressed reason:
  no_observed_anti_air_assets=3

threat-state accept_positive:
  BUILD_STATIC_DEFENSE / ground_threat = 3
  BUILD_STATIC_DEFENSE / air_threat = 1
```

The new data improves positive coverage, especially for BUILD_STATIC_DEFENSE,
but it does not cover the late online-smoke air-and-ground failure mode where
anti-air assets are already gone.

New imitation candidates:

```text
cap2 trainable:
  run: runs\20260630_160130_strategy_imitation_fresh_plus_threat_teacher_cap2_v1
  checkpoint:
    runs\20260630_160130_strategy_imitation_fresh_plus_threat_teacher_cap2_v1\checkpoints\policy.pt
  examples: 87
  kept_by_training_use:
    accept_positive=29
    drop_ambiguous=58
  train_accuracy: 0.471
  validation_accuracy: 0.353

strict-positive:
  run: runs\20260630_160253_strategy_imitation_fresh_plus_threat_teacher_strict_positive_v1
  checkpoint:
    runs\20260630_160253_strategy_imitation_fresh_plus_threat_teacher_strict_positive_v1\checkpoints\policy.pt
  examples: 29
  kept_by_training_use:
    accept_positive=29
  train_accuracy: 0.870
  validation_accuracy: 0.500
```

Sweeps used the current v2-schema action critic:

```text
runs\20260630_154556_strategy_action_critic_combined_plus_online_smoke_v2_trainable_schema_v2_v1\checkpoints\critic.pt
```

Combined fresh+teacher training surface:

```text
cap2:
  artifact:
    runs\20260630_160130_strategy_imitation_fresh_plus_threat_teacher_cap2_v1\artifacts\action_critic_threshold_sweep_v2_critic_on_fresh_plus_threat_teacher.json
  recommendation: promotion_candidate
  selected: threshold=0.800 fallback=first-executable
  accept_positive_match=17/29
  veto_negative_match=0/4
  drop_non_executable_match=0/405
  fallback_rows=1
  unsafe_fallback_rows=0

strict-positive:
  artifact:
    runs\20260630_160253_strategy_imitation_fresh_plus_threat_teacher_strict_positive_v1\artifacts\action_critic_threshold_sweep_v2_critic_on_fresh_plus_threat_teacher.json
  recommendation: promotion_candidate
  selected: threshold=0.800 fallback=first-executable
  accept_positive_match=16/29
  veto_negative_match=0/4
  drop_non_executable_match=0/405
  fallback_rows=1
  unsafe_fallback_rows=0
```

Online smoke v2 regression surface:

```text
cap2:
  artifact:
    runs\20260630_160130_strategy_imitation_fresh_plus_threat_teacher_cap2_v1\artifacts\action_critic_threshold_sweep_v2_critic_on_online_smoke_v2.json
  recommendation: hold
  selected: threshold=0.200 fallback=first-executable
  blocking_reasons:
    predicted_matches_action_space_exhausted_labels
    action_critic_all_executable_candidates_vetoed
  accept_positive_match=2/5
  fallback_rows=8
  unsafe_fallback_rows=8

strict-positive:
  artifact:
    runs\20260630_160253_strategy_imitation_fresh_plus_threat_teacher_strict_positive_v1\artifacts\action_critic_threshold_sweep_v2_critic_on_online_smoke_v2.json
  recommendation: hold
  selected: threshold=0.200 fallback=rule-safe
  blocking_reasons:
    predicted_matches_action_space_exhausted_labels
    action_critic_all_executable_candidates_vetoed
  accept_positive_match=3/5
  fallback_rows=8
  unsafe_fallback_rows=8
  note: threshold=0.400 can reach accept_positive_match=4/5, but still holds
        on the same action-space/fallback blockers.
```

Decision:

```text
Do not promote either new imitation checkpoint.
Do not promote the refreshed critic.
Do not change default runtime.
Do not run PPO.

The focused teacher batch is useful training data, but it did not solve the
online smoke v2 late air-and-ground threat blocker. The project now has more
positive BUILD_STATIC_DEFENSE coverage, yet still lacks positive, executable
anti-air recovery coverage before all candidates become vetoed or unaffordable.
```

Next concrete work:

```text
Treat the next data target as anti-air recovery, not generic threat response.
Collect or construct examples where the bot preserves or rebuilds anti-air
assets before late air-and-ground threat:
  stalker/sentry/cannon present while threat starts
  PRODUCE_ARMY or BUILD_STATIC_DEFENSE executable with payoff
  no transition into only-STAY_COURSE action-space exhaustion

Do not run another imitation training job from the same data until that missing
anti-air recovery surface exists.
```

## 2026-06-30 Anti-Air Recovery Diagnostic Follow-Up

An offline anti-air recovery diagnostic was added to turn the late air-threat
failure into a concrete data target:

```text
rl\strategy_anti_air_recovery_analysis.py
scripts\analyze_strategy_anti_air_recovery.py
tests\test_strategy_anti_air_recovery_analysis.py
```

Artifacts:

```text
online smoke v2:
  runs\20260630_153145_strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_v2\artifacts\strategy_anti_air_recovery_online_smoke_v2.json
  runs\20260630_153145_strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_v2\artifacts\strategy_anti_air_recovery_online_smoke_v2.txt

threat-positive teacher batch:
  runs\20260630_155608_strategy_threat_positive_teacher_batch_v1\artifacts\strategy_anti_air_recovery_threat_positive_teacher_batch_v1.json
  runs\20260630_155608_strategy_threat_positive_teacher_batch_v1\artifacts\strategy_anti_air_recovery_threat_positive_teacher_batch_v1.txt
```

Findings:

```text
Online smoke v2:
  anti_air_gap_files: 2/2
  files_with_pre_gap_recovery_window: 2/2
  files_with_pre_gap_executable_recovery_selected: 1/2
  missed_recovery_windows: 1/2

Threat-positive teacher batch:
  anti_air_gap_files: 2/4
  files_with_pre_gap_recovery_window: 2/2
  files_with_pre_gap_executable_recovery_selected: 2/2
  missed_recovery_windows: 0/2
```

Interpretation:

```text
The teacher batch is not simply missing all recovery behavior; its gap files did
select executable recovery before the gap. The online smoke regression still has
one true missed-window file:
  20260630_153145_AcropolisLE_Hard_Terran_Power_001.jsonl

In that file, first air/no-anti-air gap is at 651.4s, last observed anti-air
asset was at 537.1s, and pre-gap PRODUCE_ARMY / BUILD_STATIC_DEFENSE /
TECH_ROBO windows existed without an executable recovery selection.
```

Next concrete work:

```text
Do not train again from the same fresh+teacher data.
Do not tune thresholds again without changing the data/action surface.

Collect or construct a small anti-air recovery batch that matches the missed
online-smoke shape:
  AcropolisLE / Hard Terran Power
  anti-air assets present before late pressure
  PRODUCE_ARMY, BUILD_STATIC_DEFENSE, or TECH_ROBO executable before the gap
  an executable recovery action selected before anti-air disappears
  no late transition into all-candidates-vetoed fallback

After collection:
  run readiness
  run anti-air recovery diagnostic
  only then rerun capped imitation + action-critic sweep
```

## 2026-06-30 Anti-Air Recovery Teacher Affordability Follow-Up

The anti-air recovery batch requested by the previous plan has now been run,
and it changed the problem shape.

Implemented:

```text
StrategyExecutor:
  BUILD_STATIC_DEFENSE prefers PHOTONCANNON when Forge is ready.
  Shield Battery remains a fallback when cannon is unavailable and Cybernetics
  Core is ready.

CoverageStrategyPolicy:
  TECH_ROBO requires immediate affordability: minerals >= 150, vespene >= 100.
  BUILD_STATIC_DEFENSE under threat requires minerals >= 100.
  BOOST_WORKERS is suppressed while the base is under threat.
  midgame_static_defense_floor selects BUILD_STATIC_DEFENSE when a ready Forge
  exists, static defense is below base count, game_time >= 360, and minerals >=
  150.

Signal/action-space diagnostics:
  successful recorded execution now overrides immediate executability for the
  recorded action, so post-spend rows are not mislabeled as non-executable.

Anti-air diagnostic:
  BUILD_STATIC_DEFENSE counts as anti-air recovery only when execution built
  PHOTONCANNON. Shield Battery is not anti-air recovery.
```

Focused anti-air recovery teacher batch:

```text
run:
  runs\20260630_163009_strategy_anti_air_recovery_teacher_affordability_batch_v1
strategy trajectories:
  data\trajectories\strategy_anti_air_recovery_teacher_affordability_batch_strategy_v1

grid:
  AcropolisLE, ThunderbirdLE
  Hard Terran Power/Air
  1 game per combo

results:
  1 Victory / 1 Defeat / 2 Tie

readiness:
  recommendation: train
  training_ready: true

anti-air recovery:
  anti_air_gap_files: 2/4
  files_with_pre_gap_recovery_window: 2/2
  files_with_pre_gap_executable_recovery_selected: 2/2
  missed_recovery_windows: 0/2
```

The single preceding smoke also proved the new static floor can execute a real
Photon Cannon:

```text
run:
  runs\20260630_162804_strategy_anti_air_recovery_teacher_affordability_smoke_v1

evidence:
  game_time: 525.7s
  action: BUILD_STATIC_DEFENSE
  reason: midgame_static_defense_floor
  effect: build_structure
  unit: PHOTONCANNON
```

Combined training refresh:

```text
inputs:
  data\trajectories\strategy_fresh_metadata_batch_strategy_v1
  data\trajectories\strategy_fresh_metadata_batch_strategy_v2
  data\trajectories\strategy_threat_positive_teacher_batch_strategy_v1
  data\trajectories\strategy_anti_air_recovery_teacher_affordability_batch_strategy_v1

signal dataset:
  runs\strategy_signal_dataset_fresh_plus_threat_plus_anti_air_affordability_v1.json
  training_rows: 1172
  accept_positive: 88
  action_space_exhausted: 11
  drop_ambiguous: 656
  drop_non_executable: 404
  veto_negative: 10

imitation:
  runs\20260630_163807_strategy_imitation_fresh_plus_threat_plus_anti_air_affordability_cap2_v1
  examples: 267
  train_accuracy: 0.607
  validation_accuracy: 0.698

critic:
  runs\20260630_163834_strategy_action_critic_fresh_plus_threat_plus_anti_air_affordability_schema_v2_v1
  examples: 1161
  unsafe_examples: 414
  validation accuracy/precision/recall: 0.948 / 0.865 / 1.000
```

Gate decision:

```text
Do not promote the new imitation checkpoint.
Do not promote the new action critic.
Do not change default runtime.
Do not run PPO.
```

Why:

```text
The anti-air data target is healthier:
  Photon Cannon execution exists.
  The focused anti-air batch has no missed pre-gap recovery windows.
  Successful execution metadata removed false non-executable post-spend labels.

The model is still not promotable:
  combined training sweep recommendation: hold
  online smoke v2 regression sweep recommendation: hold
  blockers:
    predicted_matches_veto_negative_labels
    predicted_matches_action_space_exhausted_labels
    action_critic_all_executable_candidates_vetoed
  online-smoke-v2 unsafe_fallback_rows: 24
```

Next concrete work:

```text
1. Keep default runtime unchanged:
     --strategy-policy rule
     --strategy-tactic-mode off

2. Keep the executor/teacher/data-contract fixes.

3. Do not collect more generic threat-positive data yet.

4. Inspect the failing rows from:
     runs\20260630_163807_strategy_imitation_fresh_plus_threat_plus_anti_air_affordability_cap2_v1\artifacts\action_critic_threshold_sweep_on_online_smoke_v2_regression.json
     runs\20260630_163807_strategy_imitation_fresh_plus_threat_plus_anti_air_affordability_cap2_v1\artifacts\action_critic_threshold_sweep_on_fresh_plus_threat_plus_anti_air_affordability.json

5. Turn those rows into a targeted experiment:
     either an eval slice / gate that reports veto_negative and
     action_space_exhausted prediction matches directly,
     or an objective/weighting change that makes matching those labels more
     expensive than missing a weak accept_positive.

6. Only rerun guarded online checkpoint smoke after offline sweeps show:
     veto_negative_match = 0
     action_space_exhausted_match = 0
     unsafe_fallback_rows near zero
     no all-executable-candidates-vetoed blocker
```

## 2026-06-30 Action-Space-Aware Critic Follow-Up

A targeted critic experiment was run because the previous blocker included
`predicted_matches_action_space_exhausted_labels`.

Implemented for observability and controlled experimentation:

```text
rl\strategy_checkpoint_signal_audit.py
scripts\audit_strategy_checkpoint_signals.py
  decision rows now include action_critic_candidate_unsafe_probabilities,
  aligned with action_critic_candidate_actions.

rl\strategy_action_critic.py
  added experimental label policy:
    trainable-action-space

  Existing trainable behavior is unchanged. The new policy explicitly labels
  action_space_exhausted rows as unsafe for action critic experiments.
```

Training:

```text
run:
  runs\20260630_165359_strategy_action_critic_fresh_plus_threat_plus_anti_air_affordability_schema_v2_action_space_v1

label_policy: trainable-action-space
feature_schema: strategy_action_critic_v2
examples: 1172
unsafe_examples: 425
validation accuracy/precision/recall: 0.932 / 0.853 / 0.989
```

Gate results:

```text
combined fresh+threat+anti-air surface:
  recommendation: hold
  selected: threshold=0.700 fallback=threat-risk
  veto_negative_match: 0/10
  action_critic_fallback_rows: 107
  unsafe_fallback_rows: 71
  blockers:
    predicted_matches_action_space_exhausted_labels
    action_critic_all_executable_candidates_vetoed

online smoke v2 regression surface:
  recommendation: hold
  selected: threshold=0.900 fallback=rule-safe
  veto_negative_match: 0/4
  action_critic_fallback_rows: 34
  unsafe_fallback_rows: 34
  blockers:
    predicted_matches_action_space_exhausted_labels
    action_critic_all_executable_candidates_vetoed
```

Decision:

```text
Do not promote this critic.
Do not promote the imitation checkpoint.
Do not change default runtime.
Do not run PPO.

The experiment is useful as evidence, not as a candidate. It reduces
veto_negative matches, but it increases fallback pressure and cannot solve rows
where the executable action set has already collapsed to STAY_COURSE.
```

Updated next concrete work:

```text
Stop spending cycles on threshold-only fixes for this blocker.

Use the new candidate-probability audit to split failures into:
  avoidable rows:
    more than one executable candidate exists
    a non-STAY recovery action is present

  unavoidable action-space-exhausted rows:
    executable candidates only include STAY_COURSE

For the unavoidable rows, move the intervention earlier:
  find the last pre-collapse row where TECH_ROBO, PRODUCE_ARMY, or
  BUILD_STATIC_DEFENSE was executable
  check whether the policy selected recovery there
  add data or a bounded executor surface at that earlier point

Only run another guarded online checkpoint smoke after the offline regression
surface shows:
  avoidable veto matches eliminated
  unavoidable action-space rows explained by an earlier recovery hypothesis
  unsafe fallback no longer expanding when action_space labels are included
```

## 2026-06-30 Pre-Collapse Recovery Diagnostic Follow-Up

The earlier-recovery hypothesis was tested with a new offline diagnostic, and
the diagnostic now doubles as a strict pre-collapse recovery gate:

```text
rl\strategy_pre_collapse_recovery_analysis.py
scripts\analyze_strategy_pre_collapse_recovery.py
tests\test_strategy_pre_collapse_recovery_analysis.py
```

The diagnostic looks at `veto_negative` and `action_space_exhausted` target rows,
classifies whether the target row itself was avoidable, and scans the previous
240 seconds for executable recovery actions:

```text
TECH_ROBO
PRODUCE_ARMY
BUILD_STATIC_DEFENSE
```

Gate behavior:

```text
default thresholds:
  max_missed_pre_collapse_recovery_rows: 0
  max_missed_pre_collapse_recovery_rate: 0.000

machine-readable fields:
  recommendation: ready | hold
  blocking_reasons
  missed_pre_collapse_recovery_rate

CLI:
  --fail-on-hold
```

### Result: Online Smoke V2 Regression

Input:

```text
data\trajectories\strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_strategy_v2
```

Artifacts:

```text
runs\20260630_153145_strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_v2\artifacts\strategy_pre_collapse_recovery_online_smoke_v2.json
runs\20260630_153145_strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_v2\artifacts\strategy_pre_collapse_recovery_online_smoke_v2.txt
```

Summary:

```text
recommendation: hold
blocking_reasons:
  missed_pre_collapse_recovery_rows
  missed_pre_collapse_recovery_rate
target_rows: 6
missed_pre_collapse_recovery_rate: 0.833
avoidability_counts:
  avoidable_recovery_available: 4
  unavoidable_only_stay_course: 2
target_training_use_counts:
  action_space_exhausted: 2
  veto_negative: 4
target_threat_state_counts:
  air_and_ground_threat: 6

rows_with_pre_collapse_recovery_window: 6/6
rows_with_pre_collapse_selected_recovery: 1/6
rows_with_pre_collapse_selected_executable_recovery: 1/6
missed_pre_collapse_recovery_rows: 5/6
```

Interpretation:

```text
ThunderbirdLE / Hard Terran Power:
  4/4 target rows were avoidable_recovery_available.
  TECH_ROBO was repeatedly executable before collapse.
  The policy recorded STAY_COURSE through those windows.

AcropolisLE / Hard Terran Power:
  2/2 target rows were unavoidable_only_stay_course at the target row.
  One had earlier selected executable BUILD_STATIC_DEFENSE.
  One missed a BUILD_STATIC_DEFENSE window around 480.0s.
```

This means the online regression is not just a critic-threshold failure. Some
rows still have an executable TECH_ROBO response, but the larger pattern is that
the policy must recover before the executable action space collapses.

### Result: Combined Teacher Surface

Inputs:

```text
data\trajectories\strategy_fresh_metadata_batch_strategy_v1
data\trajectories\strategy_fresh_metadata_batch_strategy_v2
data\trajectories\strategy_threat_positive_teacher_batch_strategy_v1
data\trajectories\strategy_anti_air_recovery_teacher_affordability_batch_strategy_v1
```

Artifacts:

```text
runs\strategy_pre_collapse_recovery_fresh_plus_threat_plus_anti_air_affordability_v1.json
runs\strategy_pre_collapse_recovery_fresh_plus_threat_plus_anti_air_affordability_v1.txt
```

Summary:

```text
recommendation: ready
blocking_reasons: <none>
target_rows: 21
missed_pre_collapse_recovery_rate: 0.000
avoidability_counts:
  avoidable_non_recovery_available: 3
  avoidable_recovery_available: 7
  unavoidable_only_stay_course: 11
target_training_use_counts:
  action_space_exhausted: 11
  veto_negative: 10

rows_with_pre_collapse_recovery_window: 21/21
rows_with_pre_collapse_selected_recovery: 21/21
rows_with_pre_collapse_selected_executable_recovery: 21/21
missed_pre_collapse_recovery_rows: 0/21
```

The teacher/combined surface still has bad target rows, but it does not show the
same missed-recovery pattern. That contrast is the useful signal: the next
experiment should make the candidate reproduce the teacher's pre-collapse
recovery behavior on the online-smoke-v2 slice.

### Decision

```text
Do not promote the imitation checkpoint.
Do not promote the trainable critic.
Do not promote the trainable-action-space critic.
Do not change default runtime.
Do not run PPO.
Do not collect generic threat-positive data as the next move.
Do not run another online smoke until the offline pre-collapse slice improves.
```

### Next Experiment

Use the offline gate before more SC2 runs:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_strategy_pre_collapse_recovery.py <candidate-strategy-trajectory-dir> --lookback-seconds 240 --fail-on-hold --json-output <run-artifacts-dir>\strategy_pre_collapse_recovery.json --text-output <run-artifacts-dir>\strategy_pre_collapse_recovery.txt
```

Gate target:

```text
pre_collapse_recovery gate:
  target rows:
    veto_negative
    action_space_exhausted
  lookback:
    240 seconds
  required report fields:
    recommendation
    blocking_reasons
    missed_pre_collapse_recovery_rows
    missed_pre_collapse_recovery_rate
    missed TECH_ROBO windows
    missed PRODUCE_ARMY windows
    missed BUILD_STATIC_DEFENSE windows
    per-map/opponent/build split
```

Then run one targeted training/data iteration against the two observed failure
patterns:

```text
Thunderbird Hard Terran Power:
  no_threat rows around 594-651s
  TECH_ROBO executable
  recorded STAY_COURSE
  later air_and_ground_threat / veto_negative

Acropolis Hard Terran Power:
  row around 480s
  BUILD_STATIC_DEFENSE executable
  recorded STAY_COURSE
  later only-STAY_COURSE action_space_exhausted
```

Candidate promotion remains blocked until:

```text
pre_collapse_recovery gate is ready on the relevant offline regression surface
missed_pre_collapse_recovery_rows improves on online-smoke-v2
veto_negative_match remains zero or explained by a recovery-safe alternative
action_space_exhausted matches are explained by earlier selected recovery
unsafe_fallback_rows do not expand
small guarded online smoke supports the offline slice
```

## 2026-07-02 Recovery-Safe Filter Candidate

An opt-in filter experiment tested whether removing ambiguous no-op labels at
recovery-capable rows would be enough to improve the offline regression surface.

Implemented:

```text
rl\strategy_filtered_datasets.py
  signal filter preset:
    trainable-recovery-safe

scripts\train_strategy_imitation.py
  --signal-filter trainable-recovery-safe

tests\test_strategy_filtered_datasets.py
  recovery-opportunity ambiguous STAY_COURSE coverage
```

Filter contract:

```text
Base behavior:
  same allowed training uses as trainable:
    accept_positive
    drop_ambiguous
    weak_context

Extra removal:
  drop a drop_ambiguous row when recorded action is STAY_COURSE and one of
  these actions is executable from the observation:
    TECH_ROBO
    PRODUCE_ARMY
    BUILD_STATIC_DEFENSE

Purpose:
  prevent behavior cloning from reinforcing no-op labels when recovery is
  immediately available, without creating future-leaking synthetic positives.
```

Dataset effect on the fresh+threat+anti-air surface with cap2:

```text
trainable:
  kept_examples: 267
  kept_by_training_use:
    accept_positive: 88
    drop_ambiguous: 176
    weak_context: 3

trainable-recovery-safe:
  kept_examples: 267
  kept_by_training_use:
    accept_positive: 88
    drop_ambiguous: 176
    weak_context: 3
  recovery_opportunity_ambiguous_examples_removed: 68
  recovery_opportunity_removed_actions_by_name:
    STAY_COURSE: 68
```

Candidate:

```text
run:
  runs\20260702_081918_20260702_strategy_imitation_recovery_safe_filter_cap2_v1

examples: 267
train_accuracy: 0.640
validation_accuracy: 0.660
checkpoint:
  runs\20260702_081918_20260702_strategy_imitation_recovery_safe_filter_cap2_v1\checkpoints\policy.pt
```

Executable-mask audit:

```text
combined fresh+threat+anti-air:
  signal_healthy: false
  accept_positive_match: 23/88
  veto_negative_match: 5/10
  action_space_exhausted_match: 11/11

online-smoke-v2 regression:
  signal_healthy: false
  accept_positive_match: 0/6
  veto_negative_match: 4/4
  action_space_exhausted_match: 2/2
```

Baseline cap2 checkpoint under the same executable-mask audit:

```text
combined fresh+threat+anti-air:
  accept_positive_match: 32/88
  veto_negative_match: 5/10
  action_space_exhausted_match: 11/11

online-smoke-v2 regression:
  accept_positive_match: 1/6
  veto_negative_match: 1/4
  action_space_exhausted_match: 2/2
```

Decision:

```text
Do not promote the recovery-safe-filter checkpoint.
Do not change default runtime.
Do not run PPO.
Do not run another online smoke from this candidate.

The filter is useful as a controlled data knob, but filtering alone is not a
solution. It removed 68 recovery-opportunity STAY_COURSE rows from the
ambiguous pool, but the trained candidate regressed the online-smoke-v2
executable-mask audit.
```

Updated next experiment:

```text
Keep trainable-recovery-safe for controlled ablations.

Do not spend the next cycle on filtering alone. Add positive recovery pressure:
  collect targeted coverage-teacher examples for Thunderbird/Acropolis Power
  oversample observed positive recovery windows
  or add a bounded teacher objective for:
    late high-vespene no-robo TECH_ROBO
    missed BUILD_STATIC_DEFENSE before only-STAY collapse

Offline success before any online smoke:
  online-smoke-v2 veto_negative_match improves below the baseline 1/4
  action_space_exhausted rows are explained by earlier selected recovery
  pre_collapse_recovery gate is ready on the candidate trajectory surface
```

## 2026-07-02 Observed Recovery-Positive Oversampling Candidate

The next ablation added positive pressure without synthetic labels.

Implemented:

```text
rl\strategy_filtered_datasets.py
  recovery_positive_oversample_factor

rl\strategy_imitation.py
scripts\train_strategy_imitation.py
  --recovery-positive-oversample-factor

tests\test_strategy_filtered_datasets.py
  observed positive recovery oversampling coverage
```

Contract:

```text
factor=1:
  default, no behavior change

factor>1:
  duplicate only observed accept_positive rows for:
    TECH_ROBO
    PRODUCE_ARMY
    BUILD_STATIC_DEFENSE

Do not synthesize labels from future failure windows.
```

Candidate:

```text
run:
  runs\20260702_082711_20260702_strategy_imitation_recovery_positive_x3_cap2_v1

filter:
  trainable-recovery-safe
max_drop_ambiguous_per_positive: 2.0
recovery_positive_oversample_factor: 3

examples: 351
train_accuracy: 0.623
validation_accuracy: 0.571
```

Dataset shape:

```text
kept_by_training_use:
  accept_positive: 172
  drop_ambiguous: 176
  weak_context: 3

kept_action_counts_by_name:
  STAY_COURSE: 176
  TECH_ROBO: 51
  BUILD_STATIC_DEFENSE: 51
  PRODUCE_ARMY: 24
  ADD_GATEWAYS: 22
  FORGE_UPGRADES: 17
  EXPAND: 10

recovery_positive_examples_before_oversample: 42
recovery_positive_examples_added_by_oversample: 84
```

Executable-mask audit:

```text
combined fresh+threat+anti-air:
  signal_healthy: false
  accept_positive_match: 30/88
  veto_negative_match: 7/10
  action_space_exhausted_match: 11/11

online-smoke-v2 regression:
  signal_healthy: false
  accept_positive_match: 2/6
  veto_negative_match: 4/4
  action_space_exhausted_match: 2/2
```

Decision:

```text
Do not promote the recovery-positive x3 checkpoint.
Do not change default runtime.
Do not run PPO.
Do not run another online smoke from this candidate.

Observed-positive oversampling is useful as an ablation knob, but it does not
solve the current failure mode. It increased recovery-positive pressure but
still matched all online-smoke-v2 veto/action-space failure rows.
```

Updated next experiment:

```text
Stop spending cycles on resampling the same positive examples.

Collect or teach the actual missed recovery windows:
  Thunderbird Hard Terran Power:
    TECH_ROBO executable around 594-651s
    recorded STAY_COURSE
    later air_and_ground_threat / veto_negative

  Acropolis Hard Terran Power:
    BUILD_STATIC_DEFENSE executable around 480s
    recorded STAY_COURSE
    later only-STAY_COURSE action-space exhaustion

Preferred next move:
  targeted coverage-teacher collection or a bounded opt-in teacher rule for
  those pre-collapse recovery states, followed by:
    readiness gate
    pre_collapse_recovery gate
    executable-mask checkpoint audit
    action-critic sweep only if the raw candidate improves

No online smoke until the offline online-smoke-v2 regression surface improves.
```
