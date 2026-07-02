"""
Batch evaluation harness for the current bot.

This script intentionally launches matches through scripts/safe_launch.py so
the SC2/Battle.net window guard is active before every game.

Example:
    .venv/Scripts/python.exe scripts/evaluate.py --games-per-combo 3 \
        --difficulties VeryEasy Easy Medium --opponents Protoss Terran Zerg
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bot.managers.coverage_strategy_policy import STRATEGY_TEACHER_PROFILES  # noqa: E402
from rl.experiments import create_experiment_run, mark_experiment_status, write_json  # noqa: E402
from scripts.summarize_eval import summarize_records  # noqa: E402

PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
SAFE_LAUNCH = PROJECT_ROOT / "scripts" / "safe_launch.py"

RESULT_RE = re.compile(r"=== Game result:\s*(Result\.\w+)\s*===")
FALLBACK_RESULT_RE = re.compile(r"Game ended\. Result = (Result\.\w+)")
AI_BUILDS = ("RandomBuild", "Rush", "Timing", "Power", "Macro", "Air")


@dataclass(frozen=True)
class EvalRecord:
    policy_name: str
    policy_type: str
    policy_checkpoint: str | None
    strategy_policy: str
    strategy_tactic_mode: str
    strategy_teacher_profile: str
    strategy_checkpoint: str | None
    strategy_action_critic_checkpoint: str | None
    strategy_action_critic_threshold: float | None
    strategy_action_critic_fallback_policy: str | None
    map_name: str
    difficulty: str
    opponent_race: str
    opponent_ai_build: str
    game_index: int
    return_code: int
    result: str | None
    duration_seconds: float
    trajectory_path: str | None
    strategy_trajectory_path: str | None


def parse_result(output: str) -> str | None:
    """Extract the SC2 result token from safe_launch/run.py output."""
    match = RESULT_RE.search(output)
    if match:
        return match.group(1)
    match = FALLBACK_RESULT_RE.search(output)
    if match:
        return match.group(1)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate ProtossRuleBot")
    parser.add_argument(
        "--maps",
        nargs="+",
        default=["AcropolisLE"],
        help="Map names to evaluate.",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["VeryEasy", "Easy", "Medium"],
        help="Built-in AI difficulties to evaluate.",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["Protoss", "Terran", "Zerg", "Random"],
        help="Opponent races to evaluate.",
    )
    parser.add_argument(
        "--ai-builds",
        nargs="+",
        choices=AI_BUILDS,
        default=["RandomBuild"],
        help="Built-in AI build styles to evaluate.",
    )
    parser.add_argument(
        "--games-per-combo",
        type=int,
        default=5,
        help="Number of games for each map/difficulty/opponent combination.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to runs/eval_<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Optional directory for per-game trajectory JSONL files.",
    )
    parser.add_argument(
        "--strategy-trajectory-dir",
        type=Path,
        default=None,
        help="Optional directory for per-game strategy trajectory JSONL files.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=(
            "Optional experiment root. If set, creates a standard run directory "
            "and writes eval output under its artifacts directory unless "
            "--output is also provided."
        ),
    )
    parser.add_argument(
        "--run-name",
        default="rule-baseline-eval",
        help="Experiment run name used with --run-root.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Optional repeatable tag stored in experiment metadata.",
    )
    parser.add_argument(
        "--record-decision-interval",
        type=int,
        default=16,
        help="Decision interval used when --trajectory-dir is set.",
    )
    parser.add_argument(
        "--guard-interval",
        type=float,
        default=0.02,
        help="window_guard polling interval.",
    )
    parser.add_argument(
        "--hide-watch-seconds",
        type=float,
        default=120.0,
        help="run.py inline hide-watch duration.",
    )
    parser.add_argument(
        "--hide-watch-interval",
        type=float,
        default=0.02,
        help="run.py inline hide-watch interval.",
    )
    parser.add_argument(
        "--game-time-limit",
        type=float,
        default=None,
        help="Optional in-game time limit forwarded to run.py.",
    )
    parser.add_argument(
        "--army-policy",
        choices=["rule", "coverage-teacher", "llm"],
        default="rule",
        help=(
            "Army policy forwarded to run.py when no checkpoint is used. "
            "coverage-teacher is for data collection only; llm is experimental."
        ),
    )
    parser.add_argument(
        "--strategy-policy",
        choices=["rule", "coverage-teacher", "checkpoint"],
        default="rule",
        help=(
            "Strategy policy forwarded to run.py. Defaults to rule no-op. "
            "coverage-teacher is for strategy data collection only; "
            "checkpoint forwards --strategy-checkpoint."
        ),
    )
    parser.add_argument(
        "--strategy-tactic-mode",
        choices=["off", "rule"],
        default="off",
        help=(
            "Explicit opt-in tactic filter forwarded to run.py. "
            "Default off preserves current strategy behavior."
        ),
    )
    parser.add_argument(
        "--strategy-teacher-profile",
        choices=list(STRATEGY_TEACHER_PROFILES),
        default="standard",
        help=(
            "Optional profile forwarded to run.py for --strategy-policy "
            "coverage-teacher. Default standard preserves current behavior."
        ),
    )
    parser.add_argument(
        "--army-attack-threshold",
        type=int,
        default=None,
        help="Optional attack threshold forwarded to run.py.",
    )
    parser.add_argument(
        "--army-retreat-threshold",
        type=int,
        default=None,
        help="Optional retreat threshold forwarded to run.py.",
    )
    parser.add_argument(
        "--retreat-peak-loss-ratio",
        type=float,
        default=None,
        help="Optional peak-loss retreat ratio forwarded to run.py.",
    )
    parser.add_argument(
        "--retreat-min-peak-army",
        type=int,
        default=None,
        help="Optional minimum attack peak army forwarded to run.py.",
    )
    parser.add_argument(
        "--retreat-min-lost-from-peak",
        type=int,
        default=None,
        help="Optional minimum peak-loss units forwarded to run.py.",
    )
    parser.add_argument(
        "--policy-name",
        default=None,
        help="Policy label stored in eval records. Defaults to rule or checkpoint stem.",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint forwarded to run.py --policy-checkpoint.",
    )
    parser.add_argument(
        "--policy-device",
        default="cpu",
        help="Torch device forwarded to run.py --policy-device.",
    )
    parser.add_argument(
        "--strategy-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional strategy checkpoint forwarded to run.py "
            "--strategy-checkpoint when --strategy-policy checkpoint."
        ),
    )
    parser.add_argument(
        "--strategy-device",
        default="cpu",
        help="Torch device forwarded to run.py --strategy-device.",
    )
    parser.add_argument(
        "--strategy-action-critic-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional action critic checkpoint forwarded to run.py for "
            "masked --strategy-policy checkpoint inference."
        ),
    )
    parser.add_argument(
        "--strategy-action-critic-threshold",
        type=float,
        default=0.5,
        help="Forwarded to run.py --strategy-action-critic-threshold.",
    )
    parser.add_argument(
        "--strategy-action-critic-fallback-policy",
        choices=["lowest-risk", "first-executable"],
        default="lowest-risk",
        help="Forwarded to run.py --strategy-action-critic-fallback-policy.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai-responses", "openai-chat"],
        default=None,
        help="Forwarded to run.py --llm-provider when --army-policy llm.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Forwarded to run.py --llm-model when --army-policy llm.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Forwarded to run.py --llm-base-url when --army-policy llm.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default=None,
        help="Forwarded to run.py --llm-api-key-env when --army-policy llm.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help="Forwarded to run.py --llm-timeout when --army-policy llm.",
    )
    parser.add_argument(
        "--llm-decision-interval",
        type=int,
        default=None,
        help="Forwarded to run.py --llm-decision-interval when --army-policy llm.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="Forwarded to run.py --llm-temperature when --army-policy llm.",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=None,
        help="Forwarded to run.py --llm-max-output-tokens when --army-policy llm.",
    )
    parser.add_argument(
        "--llm-allow-no-api-key",
        action="store_true",
        help="Forwarded to run.py --llm-allow-no-api-key when --army-policy llm.",
    )
    parser.add_argument(
        "--llm-log-dir",
        type=Path,
        default=None,
        help="Optional directory for per-game LLM decision JSONL logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = None
    if args.run_root is not None:
        experiment = create_experiment_run(
            root=args.run_root,
            name=args.run_name,
            kind="evaluation",
            config=make_eval_config(args),
            tags=args.tag,
        )
        output = args.output or (experiment.artifacts_dir / "eval.jsonl")
        trajectory_dir = args.trajectory_dir
    else:
        output = args.output or _default_output_path()
        trajectory_dir = args.trajectory_dir

    output.parent.mkdir(parents=True, exist_ok=True)

    total = (
        len(args.maps)
        * len(args.difficulties)
        * len(args.opponents)
        * len(args.ai_builds)
        * args.games_per_combo
    )
    completed = 0
    failures = 0
    records: list[dict] = []

    with output.open("a", encoding="utf-8") as out:
        for map_name in args.maps:
            for difficulty in args.difficulties:
                for opponent_race in args.opponents:
                    for opponent_ai_build in args.ai_builds:
                        for game_index in range(1, args.games_per_combo + 1):
                            completed += 1
                            print(
                                f"[{completed}/{total}] {map_name} "
                                f"{difficulty} vs {opponent_race} "
                                f"{opponent_ai_build} game {game_index}"
                            )
                            record = run_one_game(
                                map_name=map_name,
                                difficulty=difficulty,
                                opponent_race=opponent_race,
                                opponent_ai_build=opponent_ai_build,
                                game_index=game_index,
                                trajectory_dir=trajectory_dir,
                                strategy_trajectory_dir=args.strategy_trajectory_dir,
                                record_decision_interval=(
                                    args.record_decision_interval
                                ),
                                guard_interval=args.guard_interval,
                                hide_watch_seconds=args.hide_watch_seconds,
                                hide_watch_interval=args.hide_watch_interval,
                                game_time_limit=args.game_time_limit,
                                army_policy=args.army_policy,
                                strategy_policy=args.strategy_policy,
                                strategy_tactic_mode=args.strategy_tactic_mode,
                                strategy_teacher_profile=(
                                    args.strategy_teacher_profile
                                ),
                                army_attack_threshold=args.army_attack_threshold,
                                army_retreat_threshold=args.army_retreat_threshold,
                                retreat_peak_loss_ratio=args.retreat_peak_loss_ratio,
                                retreat_min_peak_army=args.retreat_min_peak_army,
                                retreat_min_lost_from_peak=(
                                    args.retreat_min_lost_from_peak
                                ),
                                policy_name=policy_name(args),
                                policy_checkpoint=args.policy_checkpoint,
                                policy_device=args.policy_device,
                                strategy_checkpoint=args.strategy_checkpoint,
                                strategy_device=args.strategy_device,
                                strategy_action_critic_checkpoint=(
                                    args.strategy_action_critic_checkpoint
                                ),
                                strategy_action_critic_threshold=(
                                    args.strategy_action_critic_threshold
                                ),
                                strategy_action_critic_fallback_policy=(
                                    args.strategy_action_critic_fallback_policy
                                ),
                                llm_provider=args.llm_provider,
                                llm_model=args.llm_model,
                                llm_base_url=args.llm_base_url,
                                llm_api_key_env=args.llm_api_key_env,
                                llm_timeout=args.llm_timeout,
                                llm_decision_interval=args.llm_decision_interval,
                                llm_temperature=args.llm_temperature,
                                llm_max_output_tokens=args.llm_max_output_tokens,
                                llm_allow_no_api_key=args.llm_allow_no_api_key,
                                llm_log_dir=args.llm_log_dir,
                            )
                            record_dict = asdict(record)
                            records.append(record_dict)
                            if record.return_code != 0:
                                failures += 1
                            out.write(
                                json.dumps(record_dict, ensure_ascii=False) + "\n"
                            )
                            out.flush()
                            print(
                                f"  -> {record.result or 'NO_RESULT'} "
                                f"code={record.return_code} "
                                f"duration={record.duration_seconds:.1f}s"
                            )

    print(f"Evaluation written to {output}")
    if experiment is not None:
        summary = write_eval_summary(
            records,
            output=output,
            summary_path=experiment.artifacts_dir / "summary.json",
        )
        mark_experiment_status(
            experiment,
            "failed" if failures else "complete",
            summary=summary,
        )
    return 1 if failures else 0


def make_eval_config(args: argparse.Namespace) -> dict:
    """Return the stable config payload stored in experiment metadata."""
    config = {
        "maps": list(args.maps),
        "difficulties": list(args.difficulties),
        "opponents": list(args.opponents),
        "ai_builds": list(args.ai_builds),
        "games_per_combo": int(args.games_per_combo),
        "record_decision_interval": int(args.record_decision_interval),
        "guard_interval": float(args.guard_interval),
        "hide_watch_seconds": float(args.hide_watch_seconds),
        "hide_watch_interval": float(args.hide_watch_interval),
        "game_time_limit": (
            float(args.game_time_limit) if args.game_time_limit is not None else None
        ),
        "army_policy": args.army_policy,
        "strategy_policy": args.strategy_policy,
        "strategy_tactic_mode": args.strategy_tactic_mode,
        "strategy_teacher_profile": args.strategy_teacher_profile,
        "army_attack_threshold": args.army_attack_threshold,
        "army_retreat_threshold": args.army_retreat_threshold,
        "retreat_peak_loss_ratio": args.retreat_peak_loss_ratio,
        "retreat_min_peak_army": args.retreat_min_peak_army,
        "retreat_min_lost_from_peak": args.retreat_min_lost_from_peak,
        "policy_name": policy_name(args),
        "policy_checkpoint": (
            str(args.policy_checkpoint) if args.policy_checkpoint is not None else None
        ),
        "policy_device": args.policy_device,
        "strategy_checkpoint": (
            str(args.strategy_checkpoint)
            if args.strategy_checkpoint is not None
            else None
        ),
        "strategy_device": args.strategy_device,
        "strategy_action_critic_checkpoint": (
            str(args.strategy_action_critic_checkpoint)
            if args.strategy_action_critic_checkpoint is not None
            else None
        ),
        "strategy_action_critic_threshold": args.strategy_action_critic_threshold,
        "strategy_action_critic_fallback_policy": (
            args.strategy_action_critic_fallback_policy
        ),
        "trajectory_dir": str(args.trajectory_dir) if args.trajectory_dir else None,
        "strategy_trajectory_dir": (
            str(args.strategy_trajectory_dir)
            if args.strategy_trajectory_dir
            else None
        ),
    }
    if args.army_policy == "llm":
        config.update(
            {
                "llm_provider": args.llm_provider,
                "llm_model": args.llm_model,
                "llm_base_url": args.llm_base_url,
                "llm_api_key_env": args.llm_api_key_env,
                "llm_timeout": args.llm_timeout,
                "llm_decision_interval": args.llm_decision_interval,
                "llm_temperature": args.llm_temperature,
                "llm_max_output_tokens": args.llm_max_output_tokens,
                "llm_allow_no_api_key": args.llm_allow_no_api_key,
                "llm_log_dir": str(args.llm_log_dir) if args.llm_log_dir else None,
            }
        )
    return config


def write_eval_summary(
    records: list[dict],
    *,
    output: Path,
    summary_path: Path,
) -> dict:
    """Write artifacts/summary.json and return metadata summary payload."""
    summaries = [asdict(summary) for summary in summarize_records(records)]
    write_json(summary_path, summaries)
    return {
        "output": str(output),
        "summary_json": str(summary_path),
        "games": len(records),
        "failures": sum(1 for row in records if int(row.get("return_code", 1)) != 0),
        "groups": summaries,
    }


def policy_name(args: argparse.Namespace) -> str:
    """Return a stable policy label for eval records."""
    if args.policy_name:
        return str(args.policy_name)
    if args.policy_checkpoint is not None:
        return Path(args.policy_checkpoint).stem
    if getattr(args, "strategy_checkpoint", None) is not None:
        return Path(args.strategy_checkpoint).stem
    if getattr(args, "army_policy", None) == "llm":
        model = getattr(args, "llm_model", None)
        return f"llm-{model}" if model else "llm"
    return "rule"


def run_one_game(
    *,
    map_name: str,
    difficulty: str,
    opponent_race: str,
    opponent_ai_build: str,
    game_index: int,
    trajectory_dir: Path | None,
    strategy_trajectory_dir: Path | None,
    record_decision_interval: int,
    guard_interval: float,
    hide_watch_seconds: float,
    hide_watch_interval: float,
    game_time_limit: float | None,
    army_policy: str,
    strategy_policy: str,
    strategy_tactic_mode: str,
    strategy_teacher_profile: str,
    army_attack_threshold: int | None,
    army_retreat_threshold: int | None,
    retreat_peak_loss_ratio: float | None,
    retreat_min_peak_army: int | None,
    retreat_min_lost_from_peak: int | None,
    policy_name: str,
    policy_checkpoint: Path | None,
    policy_device: str,
    strategy_checkpoint: Path | None,
    strategy_device: str,
    strategy_action_critic_checkpoint: Path | None,
    strategy_action_critic_threshold: float,
    strategy_action_critic_fallback_policy: str,
    llm_provider: str | None,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_api_key_env: str | None,
    llm_timeout: float | None,
    llm_decision_interval: int | None,
    llm_temperature: float | None,
    llm_max_output_tokens: int | None,
    llm_allow_no_api_key: bool,
    llm_log_dir: Path | None,
) -> EvalRecord:
    trajectory_path = _trajectory_path(
        trajectory_dir,
        map_name,
        difficulty,
        opponent_race,
        opponent_ai_build,
        game_index,
    )
    strategy_trajectory_path = _trajectory_path(
        strategy_trajectory_dir,
        map_name,
        difficulty,
        opponent_race,
        opponent_ai_build,
        game_index,
    )
    llm_log_path = _llm_log_path(
        llm_log_dir,
        map_name,
        difficulty,
        opponent_race,
        opponent_ai_build,
        game_index,
    )
    command = [
        str(PYTHON if PYTHON.exists() else Path(sys.executable)),
        str(SAFE_LAUNCH),
        "--keep-guard",
        "--guard-interval",
        str(guard_interval),
        "--",
        "--map",
        map_name,
        "--difficulty",
        difficulty,
        "--opponent",
        opponent_race,
        "--ai-build",
        opponent_ai_build,
        "--hide-watch-seconds",
        str(hide_watch_seconds),
        "--hide-watch-interval",
        str(hide_watch_interval),
        "--strategy-policy",
        strategy_policy,
    ]
    if strategy_tactic_mode != "off":
        command.extend(["--strategy-tactic-mode", strategy_tactic_mode])
    if strategy_teacher_profile != "standard":
        command.extend(["--strategy-teacher-profile", strategy_teacher_profile])
    if strategy_policy == "checkpoint":
        if strategy_checkpoint is None:
            raise ValueError(
                "strategy_policy='checkpoint' requires strategy_checkpoint"
            )
        command.extend(
            [
                "--strategy-checkpoint",
                str(strategy_checkpoint),
                "--strategy-device",
                strategy_device,
            ]
        )
        if strategy_action_critic_checkpoint is not None:
            command.extend(
                [
                    "--strategy-action-critic-checkpoint",
                    str(strategy_action_critic_checkpoint),
                    "--strategy-action-critic-threshold",
                    str(strategy_action_critic_threshold),
                    "--strategy-action-critic-fallback-policy",
                    strategy_action_critic_fallback_policy,
                ]
            )
    if game_time_limit is not None:
        command.extend(["--game-time-limit", str(game_time_limit)])
    if policy_checkpoint is not None:
        command.extend(
            [
                "--policy-checkpoint",
                str(policy_checkpoint),
                "--policy-device",
                policy_device,
            ]
        )
    else:
        command.extend(["--army-policy", army_policy])
        if army_attack_threshold is not None:
            command.extend(["--army-attack-threshold", str(army_attack_threshold)])
        if army_retreat_threshold is not None:
            command.extend(["--army-retreat-threshold", str(army_retreat_threshold)])
        if retreat_peak_loss_ratio is not None:
            command.extend(["--retreat-peak-loss-ratio", str(retreat_peak_loss_ratio)])
        if retreat_min_peak_army is not None:
            command.extend(["--retreat-min-peak-army", str(retreat_min_peak_army)])
        if retreat_min_lost_from_peak is not None:
            command.extend(
                ["--retreat-min-lost-from-peak", str(retreat_min_lost_from_peak)]
            )
        if army_policy == "llm":
            if llm_provider is not None:
                command.extend(["--llm-provider", llm_provider])
            if llm_model is not None:
                command.extend(["--llm-model", llm_model])
            if llm_base_url is not None:
                command.extend(["--llm-base-url", llm_base_url])
            if llm_api_key_env is not None:
                command.extend(["--llm-api-key-env", llm_api_key_env])
            if llm_timeout is not None:
                command.extend(["--llm-timeout", str(llm_timeout)])
            if llm_decision_interval is not None:
                command.extend(["--llm-decision-interval", str(llm_decision_interval)])
            if llm_temperature is not None:
                command.extend(["--llm-temperature", str(llm_temperature)])
            if llm_max_output_tokens is not None:
                command.extend(["--llm-max-output-tokens", str(llm_max_output_tokens)])
            if llm_allow_no_api_key:
                command.append("--llm-allow-no-api-key")
            if llm_log_path is not None:
                command.extend(["--llm-log-path", str(llm_log_path)])
    if trajectory_path is not None:
        command.extend(
            [
                "--trajectory-path",
                str(trajectory_path),
                "--record-decision-interval",
                str(record_decision_interval),
            ]
        )
    if strategy_trajectory_path is not None:
        command.extend(
            [
                "--strategy-trajectory-path",
                str(strategy_trajectory_path),
            ]
        )

    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    duration = time.monotonic() - started

    return EvalRecord(
        policy_name=policy_name,
        policy_type="checkpoint" if policy_checkpoint is not None else army_policy,
        policy_checkpoint=str(policy_checkpoint) if policy_checkpoint else None,
        strategy_policy=strategy_policy,
        strategy_tactic_mode=strategy_tactic_mode,
        strategy_teacher_profile=strategy_teacher_profile,
        strategy_checkpoint=(
            str(strategy_checkpoint) if strategy_checkpoint is not None else None
        ),
        strategy_action_critic_checkpoint=(
            str(strategy_action_critic_checkpoint)
            if strategy_action_critic_checkpoint is not None
            else None
        ),
        strategy_action_critic_threshold=(
            float(strategy_action_critic_threshold)
            if strategy_action_critic_checkpoint is not None
            else None
        ),
        strategy_action_critic_fallback_policy=(
            strategy_action_critic_fallback_policy
            if strategy_action_critic_checkpoint is not None
            else None
        ),
        map_name=map_name,
        difficulty=difficulty,
        opponent_race=opponent_race,
        opponent_ai_build=opponent_ai_build,
        game_index=game_index,
        return_code=proc.returncode,
        result=parse_result(proc.stdout),
        duration_seconds=duration,
        trajectory_path=str(trajectory_path) if trajectory_path else None,
        strategy_trajectory_path=(
            str(strategy_trajectory_path) if strategy_trajectory_path else None
        ),
    )


def _default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / f"eval_{stamp}.jsonl"


def _trajectory_path(
    trajectory_dir: Path | None,
    map_name: str,
    difficulty: str,
    opponent_race: str,
    opponent_ai_build: str,
    game_index: int,
) -> Path | None:
    if trajectory_dir is None:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{stamp}_{map_name}_{difficulty}_{opponent_race}_"
        f"{opponent_ai_build}_{game_index:03d}.jsonl"
    )
    return trajectory_dir / filename


def _llm_log_path(
    llm_log_dir: Path | None,
    map_name: str,
    difficulty: str,
    opponent_race: str,
    opponent_ai_build: str,
    game_index: int,
) -> Path | None:
    if llm_log_dir is None:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{stamp}_{map_name}_{difficulty}_{opponent_race}_"
        f"{opponent_ai_build}_{game_index:03d}.jsonl"
    )
    return llm_log_dir / filename


if __name__ == "__main__":
    raise SystemExit(main())
