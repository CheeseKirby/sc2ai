"""Run the portable strategy-policy arena without launching StarCraft II."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.managers.llm_army_policy import LLMPolicyConfig  # noqa: E402
from bot.managers.llm_strategy_policy import (  # noqa: E402
    OpenAICompatibleStrategyDecisionClient,
)
from rl.experiments import (  # noqa: E402
    create_experiment_run,
    mark_experiment_status,
    write_json,
)
from rl.strategy_lab import (  # noqa: E402
    HeuristicStrategyLabPolicy,
    LLMStrategyLabPolicy,
    PPOStrategyLabPolicy,
    RandomStrategyLabPolicy,
    StayCourseStrategyLabPolicy,
    benchmark_strategy_policies,
)


POLICY_CHOICES = ("heuristic", "random", "stay-course", "ppo", "llm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark strategy policies in a deterministic surrogate arena; "
            "this command never launches SC2."
        )
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_CHOICES,
        default=["heuristic", "random", "stay-course"],
    )
    parser.add_argument("--episodes-per-scenario", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="strategy-policy-lab")
    parser.add_argument("--ppo-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-llm-api",
        action="store_true",
        help="Required acknowledgement before the benchmark can make paid API calls.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_policies(args: argparse.Namespace) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for name in args.policies:
        if name == "heuristic":
            policies[name] = HeuristicStrategyLabPolicy()
        elif name == "random":
            policies[name] = RandomStrategyLabPolicy(seed=args.seed)
        elif name == "stay-course":
            policies[name] = StayCourseStrategyLabPolicy()
        elif name == "ppo":
            if args.ppo_checkpoint is None:
                raise SystemExit("--policies ppo requires --ppo-checkpoint")
            policies[name] = PPOStrategyLabPolicy(
                args.ppo_checkpoint,
                device=args.device,
            )
        elif name == "llm":
            if not args.allow_llm_api:
                raise SystemExit(
                    "--policies llm requires --allow-llm-api because every "
                    "decision may create latency and API cost"
                )
            config = LLMPolicyConfig.from_env()
            config.validate()
            policies[name] = LLMStrategyLabPolicy(
                OpenAICompatibleStrategyDecisionClient(config)
            )
    return policies


def main() -> int:
    args = parse_args()
    config = {
        "policies": list(args.policies),
        "episodes_per_scenario": args.episodes_per_scenario,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "ppo_checkpoint": str(args.ppo_checkpoint) if args.ppo_checkpoint else None,
        "device": args.device,
        "llm_api_enabled": bool(args.allow_llm_api),
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "kind": "strategy_policy_lab",
                    "run_created": False,
                    "config": config,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    policies = build_policies(args)
    run = create_experiment_run(
        root=args.run_root,
        name=args.run_name,
        kind="strategy_policy_lab",
        config=config,
        tags=["strategy", "ppo", "llm", "surrogate", "evaluation"],
    )
    try:
        report = benchmark_strategy_policies(
            policies,
            episodes_per_scenario=args.episodes_per_scenario,
            seed=args.seed,
            max_steps=args.max_steps,
            trace_path=run.logs_dir / "strategy_decisions.jsonl",
        )
        write_json(run.artifacts_dir / "strategy_lab_report.json", report)
    except Exception as exc:
        mark_experiment_status(run, "failed", summary={"error": repr(exc)})
        raise

    summary = {
        name: {
            "win_rate": metrics["win_rate"],
            "mean_reward": metrics["mean_reward"],
            "blocked_action_rate": metrics["blocked_action_rate"],
            "fallback_count": metrics["fallback_count"],
        }
        for name, metrics in report["policies"].items()
    }
    mark_experiment_status(run, "completed", summary=summary)
    print(
        json.dumps(
            {
                "run": str(run.root),
                "report": str(run.artifacts_dir / "strategy_lab_report.json"),
                "trace": str(run.logs_dir / "strategy_decisions.jsonl"),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
