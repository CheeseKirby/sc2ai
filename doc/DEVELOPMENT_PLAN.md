# Development Plan

This is the short current plan. Full historical experiment logs are archived in
`doc/archive/`.

## Current Route

The next milestone is:

```text
Recovery-Aware Strategy Candidate v1
```

Do not jump to PPO yet. The immediate goal is to produce a small supervised
strategy checkpoint that can recognize pre-collapse recovery windows, choose the
right recovery action, execute useful non-STAY actions, and keep an explainable
reason trail.

Default runtime remains unchanged throughout this stage:

```text
--strategy-policy rule
--strategy-tactic-mode off
```

## Operating Rules

Keep development evidence-first, but avoid another long gate-only cycle.

Each cycle should produce at least one of:

```text
new targeted trajectories
candidate training run
checkpoint audit
small guarded SC2 evaluation
```

Do not spend the next cycle on:

```text
blind recovery loss weight increases
blind context oversampling increases
threshold-only critic tuning
new diagnostics that do not directly decide collect/train/audit/smoke
```

All trained artifacts are candidates first. They do not replace the default
runtime automatically.

## Phase 1: Add Recovery Evidence

Add or collect more matched context-positive recovery rows. The important
contexts are:

```text
TECH_ROBO:
  late high-vespene / no Robo / no immediate base threat

BUILD_STATIC_DEFENSE:
  late low-static-defense / affordable static defense / no immediate base threat

PRODUCE_ARMY:
  production and army-pressure contexts where units are the correct response
```

Prefer targeted teacher collection or a small reporting tool that identifies
missing/confused rows by map, build, action, time window, and context.

If collecting SC2 data, run the guard first:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.safe_launch import ensure_guard_running, find_existing_guard; ensure_guard_running(0.02); print(find_existing_guard())"
```

Then use `scripts\evaluate.py`, not a direct visible launch.

## Phase 2: Train A Small Candidate

After new data passes readiness, train a small recovery-aware imitation
candidate from the combined useful surfaces:

```text
fresh strategy metadata batches
threat-positive teacher batch
anti-air recovery teacher batch
pre-collapse recovery teacher batch
new matched recovery-context data
```

Recommended starting posture:

```text
--signal-filter trainable-recovery-safe
--max-drop-ambiguous-per-positive 2.0
context-aware recovery filtering
action-specific recovery weights only as controlled ablations
```

Do not treat train or validation accuracy alone as success. The candidate must
be judged by recovery slices and bad-label gates.

## Phase 3: Audit Before Smoke

Before any guarded online smoke, audit the candidate with:

```text
scripts\audit_strategy_checkpoint_signals.py
scripts\audit_strategy_checkpoint_pre_collapse_recovery.py
scripts\audit_strategy_recovery_context.py
```

Required offline direction:

```text
combined pre-collapse recovery gate does not regress
context-matched recovery misses = 0
context-matched cross-action confusion = 0
veto_negative_match = 0
drop_non_executable_match = 0
action_space_exhausted_match = 0
unsafe_fallback_rows does not expand
```

If an `action_space_exhausted` row cannot be eliminated at the target row, it
must be explained by an earlier recovery decision inside the lookback window:

```text
the checkpoint selected TECH_ROBO, BUILD_STATIC_DEFENSE, or PRODUCE_ARMY while
that recovery action was still executable.
```

## Phase 4: Guarded Online Smoke

Only after Phase 3 improves without regressions, run a tiny guarded smoke.

Priority surface:

```text
maps:
  ThunderbirdLE
  AcropolisLE

opponent/build:
  Hard Terran Power
```

Smoke is evidence, not promotion. Inspect:

```text
whether Robo/static/army recovery happens before collapse
whether non-STAY actions execute successfully
whether executor noop loops return
whether STAY_COURSE dominates under pressure
whether results improve beyond the previous Tie/collapse pattern
```

## PPO Readiness

PPO engineering may begin when:

```text
Recovery-Aware Strategy Candidate v1 exists
offline recovery/context gates are clean or clearly improved
small guarded online smoke shows useful non-STAY actions executing
rollout logs contain state/action/effect/reward detail for attribution
rule/off remains an immediate rollback path
```

Actual PPO training should wait until:

```text
the supervised candidate is not dominated by STAY_COURSE collapse
TECH_ROBO / BUILD_STATIC_DEFENSE / PRODUCE_ARMY confusion is cleared
bad-label matches are eliminated or rare and explained
the PPO harness has reproducible checkpoints and hard stop conditions
```

PPO promotion, when it eventually exists, will need separate offline regression
gates, small online comparison, rollback, and explanation checks.

## Definition Of Done For This Stage

This stage is done when there is a hold-or-promote decision for
`Recovery-Aware Strategy Candidate v1` backed by:

```text
new matched recovery-context evidence
readiness results
training run artifact
signal audit
pre-collapse recovery audit
recovery-context audit
small guarded online smoke, only if offline gates justify it
```

If the candidate is still hold-only, the next step should be an error cluster
report that says whether the blocker is missing data, label ambiguity, feature
blindness, action-mask/runtime mismatch, or reward/PPO-harness readiness.

## Archives

Full historical records:

```text
doc\archive\2026-07-02_STATE_FULL.md
doc\archive\2026-07-02_DEVELOPMENT_PLAN_FULL.md
```
