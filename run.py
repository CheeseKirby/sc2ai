"""
Entry point: run ProtossRuleBot against the built-in AI.

Usage:
    .venv/Scripts/python.exe run.py
    .venv/Scripts/python.exe run.py --map EphemeronLE --difficulty Hard
    .venv/Scripts/python.exe run.py --difficulty Expert
    .venv/Scripts/python.exe run.py --difficulty Easy --opponent Zerg

Flags:
    --realtime     run in 1x speed (good for spectating); default is fastest
    --show-window  do NOT auto-hide the SC2 window (for debugging only)
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must import bot.config FIRST to set SC2PATH before sc2.* imports
from bot.config import DEFAULT_MAP, AVAILABLE_MAPS  # noqa: E402
from bot.managers.coverage_army_policy import CoverageArmyPolicy  # noqa: E402
from bot.managers.llm_army_policy import LLMArmyPolicy, LLMPolicyConfig  # noqa: E402
from bot.managers.rl_army_policy import RLArmyPolicy  # noqa: E402
from bot.protoss_rule_bot import ProtossRuleBot  # noqa: E402
from bot.window_hider import hide_target_windows  # noqa: E402
from rl.trajectory_recorder import JsonlTrajectoryRecorder  # noqa: E402

from sc2 import maps  # noqa: E402
from sc2.data import Difficulty, Race  # noqa: E402
from sc2.main import run_game  # noqa: E402
from sc2.player import Bot, Computer  # noqa: E402

logger = logging.getLogger(__name__)


DIFFICULTY_MAP: dict[str, Difficulty] = {
    "VeryEasy": Difficulty.VeryEasy,
    "Easy": Difficulty.Easy,
    "Medium": Difficulty.Medium,
    "MediumHard": Difficulty.MediumHard,
    "Hard": Difficulty.Hard,
    "Harder": Difficulty.Harder,
    "VeryHard": Difficulty.VeryHard,
    "CheatVision": Difficulty.CheatVision,
    "CheatMoney": Difficulty.CheatMoney,
    "CheatInsane": Difficulty.CheatInsane,
    # User-facing aliases for "official expert AI" training/testing.
    # SC2/burnysc2 does not expose a literal Expert enum; VeryHard is the
    # strongest non-cheating built-in AI, while CheatInsane is the harshest
    # official built-in benchmark.
    "Expert": Difficulty.VeryHard,
    "ExpertCheat": Difficulty.CheatInsane,
}

RACE_MAP: dict[str, Race] = {
    "Protoss": Race.Protoss,
    "Terran": Race.Terran,
    "Zerg": Race.Zerg,
    "Random": Race.Random,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ProtossRuleBot vs builtin AI")
    p.add_argument(
        "--map",
        default=DEFAULT_MAP,
        choices=AVAILABLE_MAPS,
        help=f"Map name (default: {DEFAULT_MAP})",
    )
    p.add_argument(
        "--difficulty",
        default="Easy",
        choices=list(DIFFICULTY_MAP.keys()),
        help=(
            "Built-in AI difficulty (default: Easy). "
            "Use Expert for the strongest fair built-in AI, or ExpertCheat "
            "for the harshest official built-in benchmark."
        ),
    )
    p.add_argument(
        "--opponent",
        default="Random",
        choices=list(RACE_MAP.keys()),
        help="Built-in AI race (default: Random)",
    )
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Realtime (1x) speed; default is fastest.",
    )
    p.add_argument(
        "--show-window",
        action="store_true",
        help="Do not auto-hide the SC2 window (debug only).",
    )
    p.add_argument(
        "--hide-watch-seconds",
        type=float,
        default=30.0,
        help=(
            "After launch, keep polling for new SC2/Battle.net windows for this "
            "many seconds and hide them (default: 30)."
        ),
    )
    p.add_argument(
        "--hide-watch-interval",
        type=float,
        default=0.05,
        help=(
            "Polling interval for the inline hide-watcher in seconds. "
            "Smaller = shorter window-visible flash on startup, but more CPU. "
            "Default 0.05s caps the visible flicker at ~50ms."
        ),
    )
    p.add_argument(
        "--army-policy",
        choices=["rule", "coverage-teacher", "llm"],
        default="rule",
        help=(
            "Army policy to use when --policy-checkpoint is omitted. "
            "coverage-teacher is for data collection only; llm is experimental."
        ),
    )
    p.add_argument(
        "--army-attack-threshold",
        type=int,
        default=None,
        help="Optional override for ProtossRuleBot.ARMY_ATTACK_THRESHOLD.",
    )
    p.add_argument(
        "--army-retreat-threshold",
        type=int,
        default=None,
        help="Optional override for ProtossRuleBot.ARMY_RETREAT_THRESHOLD.",
    )
    p.add_argument(
        "--retreat-peak-loss-ratio",
        type=float,
        default=None,
        help="Optional override for coverage-teacher peak-loss retreat ratio.",
    )
    p.add_argument(
        "--retreat-min-peak-army",
        type=int,
        default=None,
        help="Optional minimum attack-phase peak army for peak-loss retreat.",
    )
    p.add_argument(
        "--retreat-min-lost-from-peak",
        type=int,
        default=None,
        help="Optional minimum units lost from attack peak for peak-loss retreat.",
    )
    p.add_argument(
        "--trajectory-path",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for high-level trajectory records. "
            "When omitted, no training data is written."
        ),
    )
    p.add_argument(
        "--record-decision-interval",
        type=int,
        default=8,
        help=(
            "Record one high-level decision every N bot iterations when "
            "--trajectory-path is set (default: 8)."
        ),
    )
    p.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional policy checkpoint for RLArmyPolicy. "
            "When omitted, uses the rule army policy."
        ),
    )
    p.add_argument(
        "--policy-device",
        default="cpu",
        help="Torch device for --policy-checkpoint inference (default: cpu).",
    )
    p.add_argument(
        "--llm-provider",
        choices=["openai-responses", "openai-chat"],
        default=None,
        help=(
            "LLM API provider for --army-policy llm. Defaults to "
            "SC2_LLM_PROVIDER or openai-responses."
        ),
    )
    p.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Model for --army-policy llm. Defaults to SC2_LLM_MODEL, "
            "OPENAI_MODEL, or a small GPT model."
        ),
    )
    p.add_argument(
        "--llm-base-url",
        default=None,
        help=(
            "Base URL for --army-policy llm. Defaults to SC2_LLM_BASE_URL "
            "or https://api.openai.com/v1."
        ),
    )
    p.add_argument(
        "--llm-api-key-env",
        default=None,
        help=(
            "Environment variable containing the LLM API key. Defaults to "
            "SC2_LLM_API_KEY_ENV or OPENAI_API_KEY."
        ),
    )
    p.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help="LLM request timeout in seconds (default: SC2_LLM_TIMEOUT or 2.5).",
    )
    p.add_argument(
        "--llm-decision-interval",
        type=int,
        default=None,
        help=(
            "Bot iterations between LLM calls. Cached decisions are reissued "
            "between calls (default: SC2_LLM_DECISION_INTERVAL or 64)."
        ),
    )
    p.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="LLM sampling temperature (default: SC2_LLM_TEMPERATURE or 0.2).",
    )
    p.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=None,
        help=(
            "Maximum LLM output tokens for the JSON decision "
            "(default: SC2_LLM_MAX_OUTPUT_TOKENS or 180)."
        ),
    )
    p.add_argument(
        "--llm-allow-no-api-key",
        action="store_true",
        help=(
            "Allow calling an OpenAI-compatible local endpoint without an API "
            "key. Useful for local servers only."
        ),
    )
    p.add_argument(
        "--llm-log-path",
        type=Path,
        default=None,
        help="Optional JSONL path for LLM decisions and explanations.",
    )
    p.add_argument(
        "--game-time-limit",
        type=float,
        default=None,
        help=(
            "Optional in-game time limit in seconds. Useful for smoke tests "
            "with immature learned policies."
        ),
    )
    return p.parse_args()


def _hide_watcher_loop(
    duration_seconds: float, interval: float, stop_event: threading.Event
) -> None:
    """Aggressively hide any target window that appears within ``duration_seconds``.

    Runs in a background thread while ``run_game`` is bringing SC2 up.
    """
    deadline = time.monotonic() + duration_seconds
    hidden_total = 0
    while not stop_event.is_set() and time.monotonic() < deadline:
        try:
            n = hide_target_windows()
        except Exception:
            logger.exception("hide_target_windows failed inside watcher")
            n = 0
        hidden_total += n
        if n > 0:
            logger.info("hide-watcher: hid %d window(s)", n)
        time.sleep(interval)
    logger.info(
        "hide-watcher exiting (hid %d window(s) over %.1fs)",
        hidden_total,
        duration_seconds,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()

    logger.info("Map:        %s", args.map)
    logger.info("Bot:        ProtossRuleBot (Protoss)")
    logger.info("Opponent:   built-in AI (%s, %s)", args.opponent, args.difficulty)
    logger.info("Mode:       %s", "realtime" if args.realtime else "fastest")
    logger.info("Hide:       %s", "OFF (--show-window)" if args.show_window else "ON")
    if args.game_time_limit is not None:
        logger.info("TimeLimit:  %.1f game seconds", args.game_time_limit)
    if args.trajectory_path is not None:
        logger.info("Trajectory: %s", args.trajectory_path)
    if args.policy_checkpoint is not None:
        logger.info(
            "ArmyPolicy: RLArmyPolicy checkpoint=%s device=%s",
            args.policy_checkpoint,
            args.policy_device,
        )
    else:
        logger.info("ArmyPolicy: %s", args.army_policy)
    if args.army_policy == "llm" and args.policy_checkpoint is None:
        logger.info(
            "LLM:        provider=%s model=%s base_url=%s interval=%s",
            args.llm_provider or "env/default",
            args.llm_model or "env/default",
            args.llm_base_url or "env/default",
            args.llm_decision_interval or "env/default",
        )

    stop_event = threading.Event()
    watcher: threading.Thread | None = None
    if not args.show_window:
        watcher = threading.Thread(
            target=_hide_watcher_loop,
            args=(args.hide_watch_seconds, args.hide_watch_interval, stop_event),
            name="hide-watcher",
            daemon=True,
        )
        watcher.start()

    trajectory_recorder = (
        JsonlTrajectoryRecorder(args.trajectory_path)
        if args.trajectory_path is not None
        else None
    )
    bot_ai = ProtossRuleBot(
        trajectory_recorder=trajectory_recorder,
        episode_metadata={
            "map_name": args.map,
            "difficulty": args.difficulty,
            "opponent_race": args.opponent,
        },
        record_decision_interval=args.record_decision_interval,
    )
    if args.army_attack_threshold is not None:
        bot_ai.ARMY_ATTACK_THRESHOLD = args.army_attack_threshold
    if args.army_retreat_threshold is not None:
        bot_ai.ARMY_RETREAT_THRESHOLD = args.army_retreat_threshold
    if args.retreat_peak_loss_ratio is not None:
        bot_ai.RETREAT_PEAK_LOSS_RATIO = args.retreat_peak_loss_ratio
    if args.retreat_min_peak_army is not None:
        bot_ai.RETREAT_MIN_PEAK_ARMY = args.retreat_min_peak_army
    if args.retreat_min_lost_from_peak is not None:
        bot_ai.RETREAT_MIN_LOST_FROM_PEAK = args.retreat_min_lost_from_peak
    if args.policy_checkpoint is not None:
        bot_ai.army_policy = RLArmyPolicy(
            args.policy_checkpoint,
            device=args.policy_device,
        )
    elif args.army_policy == "coverage-teacher":
        bot_ai.army_policy = CoverageArmyPolicy()
    elif args.army_policy == "llm":
        bot_ai.army_policy = LLMArmyPolicy(
            LLMPolicyConfig.from_env(
                provider=args.llm_provider,
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key_env=args.llm_api_key_env,
                timeout_seconds=args.llm_timeout,
                decision_interval=args.llm_decision_interval,
                temperature=args.llm_temperature,
                max_output_tokens=args.llm_max_output_tokens,
                require_api_key=False if args.llm_allow_no_api_key else None,
                log_path=args.llm_log_path,
            )
        )

    try:
        result = run_game(
            map_settings=maps.get(args.map),
            players=[
                Bot(Race.Protoss, bot_ai),
                Computer(RACE_MAP[args.opponent], DIFFICULTY_MAP[args.difficulty]),
            ],
            realtime=args.realtime,
            game_time_limit=args.game_time_limit,
        )
    finally:
        stop_event.set()
        if watcher is not None:
            watcher.join(timeout=1.0)
        if trajectory_recorder is not None:
            trajectory_recorder.close()
        # One last sweep, in case SC2 left an exit dialog open
        if not args.show_window:
            try:
                hide_target_windows()
            except Exception:
                logger.exception("final hide sweep failed")

    logger.info("=== Game result: %s ===", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
