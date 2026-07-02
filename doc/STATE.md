# Project State

This is the short handoff entry point. Full historical ledgers are archived in
`doc/archive/`.

## Current Snapshot

- Project root: `D:\opus\data\raw\alpaca-gpt4\sc2\sc2-ai-bot`
- Current branch: `main`
- Latest full code snapshot before doc slimming:
  `85e752d Add strategy recovery training audits`
- Latest roadmap snapshot before doc slimming:
  `b3a4bb3 Update strategy roadmap and PPO readiness`
- Latest recorded test run: `.\.venv\Scripts\python.exe -m pytest -q`
  -> `370 passed`
- Latest recorded environment check:
  `.\.venv\Scripts\python.exe scripts\check_env.py` -> OK
- SC2 client: NetEase/China `5.0.15.96999`, `Base96999`
- Default runtime remains:

```text
--strategy-policy rule
--strategy-tactic-mode off
```

## Hard Boundaries

Do not:

```text
promote any current strategy checkpoint
promote any current action critic
run PPO
change the default runtime
launch SC2 directly through visible run.py
```

For any SC2 launch, use the hidden-window guard first:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Then use `scripts\evaluate.py` or `scripts\safe_launch.py`.

## Current Capability

The project now has a usable offline strategy-learning loop:

```text
strategy trajectory metadata and readiness gates
strategy execution effect/blocker observability
AIBuild metadata in runtime, eval summaries, and trajectory rows
opt-in pre-collapse recovery teacher profile
strategy imitation training with recovery-focused knobs
action critic masking and signal audits
pre-collapse recovery, anti-air recovery, and recovery-context audits
```

The opt-in targeted teacher profile is:

```text
--strategy-policy coverage-teacher
--strategy-teacher-profile pre-collapse-recovery
```

It is for data collection only. The default teacher profile remains `standard`.

## Latest Evidence

Targeted recovery work succeeded as instrumentation and data plumbing:

```text
anti-air recovery teacher data produced useful Photon Cannon evidence
pre-collapse recovery teacher data collected Thunderbird/Acropolis Hard Terran
  Power recovery-window examples
checkpoint-level pre-collapse and recovery-context audits are implemented
```

Current trained candidates are still hold-only:

```text
global recovery positive weights: hold
action-specific recovery weights: hold
context-aware recovery candidates: hold
current action critics: hold
```

The latest recovery-context audit found the real bottleneck:

```text
combined matched pre-collapse recovery positive rows: only 5
context-only candidate: context match 2/5
context x3 candidate: context match 2/5
context x10 candidate: context match 4/5, but regressed combined
  pre-collapse recovery with 6/36 missed rows
online-smoke-v2 recovery context rows: 0 matched rows, so it is not proof of
  context readiness
```

## Current Blocker

This is no longer mainly a scalar-weight problem.

The blocker is sparse and entangled recovery evidence:

```text
too few matched context-positive recovery rows
TECH_ROBO / BUILD_STATIC_DEFENSE / PRODUCE_ARMY cross-action confusion
STAY_COURSE collapse or unsafe fallback under pressure
veto/action-space/drop_non_executable gates not clean enough for promotion
```

## Active Objective

Build `Recovery-Aware Strategy Candidate v1`.

The candidate should:

```text
recognize pre-collapse recovery windows
choose the correct recovery action
execute useful non-STAY actions
preserve an explainable strategy reason trail
```

Target actions:

```text
TECH_ROBO:
  late high-vespene / no Robo / no immediate base threat

BUILD_STATIC_DEFENSE:
  late low-static-defense / affordable static defense / no immediate base threat

PRODUCE_ARMY:
  production and army-pressure contexts that call for units without being
  absorbed into static-defense pressure
```

## Next Move

Prefer data and training progress over more blind sweeps:

```text
1. Add or collect more matched context-positive recovery rows.
2. Run readiness and recovery analysis gates.
3. Train a small recovery-aware imitation candidate.
4. Audit the checkpoint with signal, pre-collapse, and recovery-context gates.
5. Only run guarded online smoke if offline gates improve without regressions.
```

Useful inputs:

```text
data\trajectories\strategy_fresh_metadata_batch_strategy_v1
data\trajectories\strategy_fresh_metadata_batch_strategy_v2
data\trajectories\strategy_threat_positive_teacher_batch_strategy_v1
data\trajectories\strategy_anti_air_recovery_teacher_affordability_batch_strategy_v1
data\trajectories\strategy_pre_collapse_recovery_teacher_batch_strategy_v1
data\trajectories\strategy_checkpoint_cap2_critic_t095_first_exec_online_smoke_runtime_aligned_strategy_v2
```

Key audit scripts:

```text
scripts\audit_strategy_checkpoint_signals.py
scripts\audit_strategy_checkpoint_pre_collapse_recovery.py
scripts\audit_strategy_recovery_context.py
scripts\analyze_strategy_pre_collapse_recovery.py
```

## PPO Status

PPO is not implemented and should not be the next training step.

PPO engineering can begin only after `Recovery-Aware Strategy Candidate v1` is
a credible behavior baseline and rollout logs can support reward attribution.
Actual PPO training should wait until the supervised candidate no longer
systematically misses low-level recovery windows or learns `STAY_COURSE`
collapse as a safe behavior.

## Archives

Full historical records:

```text
doc\archive\2026-07-02_STATE_FULL.md
doc\archive\2026-07-02_DEVELOPMENT_PLAN_FULL.md
```
