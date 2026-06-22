"""
Stress test for window_guard: rapidly unhide SC2 / Battle.net windows in a loop
and verify the guard re-hides them every time.

This simulates the worst case during bot operation: many windows trying to
become visible as the game runs. The bot itself produces fewer popup events
than this test, so passing here is a strong signal.

Usage:
    .venv/Scripts/python.exe scripts/stress_guard.py --duration 30
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.window_hider import show_target_windows  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repeatedly unhide SC2/Battle.net windows to stress-test window_guard."
    )
    p.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="How long to run the test (seconds, default 30).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between unhide attempts (default 1.0).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info(
        "Stress test: unhide every %.2fs for %.1fs total. "
        "Watch logs/window_guard.log; guard should hide everything we surface.",
        args.interval,
        args.duration,
    )

    deadline = time.monotonic() + args.duration
    rounds = 0
    total_surfaced = 0
    while time.monotonic() < deadline:
        n = show_target_windows()
        rounds += 1
        total_surfaced += n
        logger.info("round %d: surfaced %d window(s)", rounds, n)
        time.sleep(args.interval)

    logger.info(
        "Done. %d rounds, %d total windows surfaced (guard should have hidden all).",
        rounds,
        total_surfaced,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
