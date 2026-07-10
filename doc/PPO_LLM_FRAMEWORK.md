# PPO and LLM Framework

This document describes scaffolding only. The default runtime remains:

```text
--army-policy rule
--strategy-policy rule
--strategy-tactic-mode off
```

## PPO

The PPO scaffold targets the existing 40-field `strategy_v2` observation and
the eight-value `StrategyAction` space.

Components:

```text
rl/ppo_types.py
  state_before / action / execution_result / state_after transition contract
  StrategyEnvBackend protocol for a future live or replay backend

rl/ppo_env.py
  Gymnasium SC2StrategyPPOEnv adapter
  rejects mismatched state_before transitions

rl/ppo_rewards.py
  explicit placeholder reward weights

rl/ppo_training.py
  Stable-Baselines3 PPO construction, learn, and save wiring

bot/managers/ppo_strategy_policy.py
  deterministic online inference from an SB3 .zip checkpoint

scripts/train_ppo.py
  dry-run config and backend-factory entry point
```

Inspect the configuration without creating a run or starting training:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo.py --dry-run
```

Actual training requires a backend factory:

```text
package.module:callable -> StrategyEnvBackend
```

No live SC2 backend is included yet. A future backend owns safe SC2 launch,
episode reset, action execution, state capture, termination, and cleanup.

## LLM

The existing Army LLM policy remains available through:

```text
--army-policy llm
```

The new low-frequency Strategy LLM policy is available through:

```text
--strategy-policy llm
```

Both use the existing `--llm-*` configuration and support OpenAI Responses or
OpenAI-compatible Chat Completions with strict JSON-schema output. Strategy LLM
responses are limited to the existing eight `StrategyAction` values. Missing
configuration, request errors, parse errors, or invalid actions fall back to
the no-op rule policy (`STAY_COURSE`).

The LLM chooses intent only. `StrategyExecutor` still validates prerequisites
and performs the actual build, production, upgrade, or defense command.

## Explicit Non-Goals

This scaffold does not claim:

```text
PPO reward correctness
live SC2 environment correctness
trained PPO performance
LLM latency safety
LLM or PPO promotion readiness
```

Before real training, implement a guarded live backend, collect clean
state-before/state-after transitions, define reward attribution, add episode
level holdouts, and establish offline plus guarded online promotion gates.
