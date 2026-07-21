# PPO and LLM Strategy Framework

The framework is runnable for interface and evaluation checks, but no trained
PPO model or live-SC2 performance claim is included. The production defaults
remain:

```text
--army-policy rule
--strategy-policy rule
--strategy-tactic-mode off
```

## Shared Decision Boundary

Both learned planners consume the 40-field `strategy_v2` observation and select
one of eight `StrategyAction` intents. `StrategyExecutor` retains authority over
resource checks and concrete SC2 commands.

```text
observation -> planner -> StrategyAction -> deterministic executor -> feedback
```

This keeps LLM latency and model errors away from frame-level unit control.

## PPO

The PPO path contains:

```text
rl/ppo_types.py
  state_before / action / execution_result / state_after contract
  StrategyEnvBackend protocol for surrogate, replay, or live backends

rl/ppo_env.py
  Gymnasium adapter
  rejects inconsistent state_before transitions
  emits reward attribution in info["reward_components"]

rl/ppo_surrogate_backend.py
  five deterministic macro scenarios for portable pipeline checks
  explicitly not a StarCraft II simulator

rl/ppo_rewards.py
  configurable, explainable reward components

rl/ppo_training.py
  Stable-Baselines3 construction, learn, and checkpoint-save wiring

bot/managers/ppo_strategy_policy.py
  deterministic inference from an SB3 .zip checkpoint

scripts/train_ppo.py
  dry-run, surrogate, and external backend entry points
```

Inspect configuration without training:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo.py `
  --backend surrogate `
  --dry-run
```

The surrogate option exists to validate plumbing when compute or SC2 is not
available. A future real backend uses:

```text
--backend external --backend-factory package.module:callable
```

No live backend is included. A live or replay backend must own safe reset,
action execution, pre/post state capture, termination, and cleanup.

## LLM

Army and Strategy LLM policies are available only when explicitly selected:

```text
--army-policy llm
--strategy-policy llm
```

They use OpenAI Responses or OpenAI-compatible Chat Completions with strict
JSON Schema output. Strategy responses are limited to the existing eight
actions and include concise reasoning plus confidence. Missing configuration,
request errors, parse errors, and invalid actions fall back to `STAY_COURSE`.

The portable benchmark requires a second acknowledgement before making calls:

```text
--policies llm --allow-llm-api
```

This prevents an offline evaluation command from silently creating API cost.

## Strategy Lab

Run local baselines without SC2, API calls, or training:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_strategy_lab.py `
  --policies heuristic random stay-course `
  --episodes-per-scenario 4
```

The report compares:

```text
surrogate win rate and mean reward
execution and blocked-action rates
fallback count
mean / p50 / p95 decision latency
action distribution
per-scenario metrics
```

The JSONL trace includes decision source, action, reasoning, confidence,
execution blocker, policy error, reward components, objective progress, and
latency. PPO checkpoints and live LLM clients can be evaluated through the same
interface when explicitly configured.

## Honest Boundaries

The framework does not claim:

```text
surrogate metrics equal live SC2 performance
reward weights are tuned
PPO has been trained
LLM or PPO is promotion-ready
LLM tail latency is safe for online play
```

Before training, implement a replay/live backend with correct transition
ordering, collect episode-level holdouts, freeze evaluation scenarios, and
define promotion gates. See `doc/HYBRID_STRATEGY_LAB.md` for the
architecture, scenario definitions, metrics, and module overview.
