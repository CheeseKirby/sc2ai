"""
Window guard: a daemon that continuously hides any SC2 / Battle.net window
that appears on screen.

Why this exists:
  Even with the bot launching SC2 in --no-window mode, several events can still
  pop a real visible window onto the user's desktop:
    - SC2 crash report dialog
    - Battle.net auto-update modal
    - Blizzard "agreement updated" prompt
    - Battle.net relaunching itself after a forced kill
  The user is at work and cannot afford ANY of these to appear. This watchdog
  polls every 0.5s and force-hides anything matching TARGET_PROCESSES.

Usage:
    .venv/Scripts/python.exe scripts/window_guard.py
    .venv/Scripts/python.exe scripts/window_guard.py --interval 0.25 --log logs/guard.log

Stop:
    Ctrl+C in the terminal, or kill the process by PID.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.window_hider import hide_target_windows  # noqa: E402

logger = logging.getLogger("window_guard")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continuously hide SC2 / Battle.net pop-up windows.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help=(
            "Polling interval in seconds (default: 0.05). "
            "At 0.05s the maximum visible window flicker is ~50ms (sub-frame for "
            "most monitors). CPU cost remains near zero because EnumWindows is "
            "cheap when nothing matches. Bump to 0.5+ to be lazier."
        ),
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional log file. If set, info-level events are appended here too.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-hide stdout chatter; only log to file.",
    )
    return p.parse_args()


def setup_logging(log_path: Path | None, quiet: bool) -> None:
    handlers: list[logging.Handler] = []
    if not quiet:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )


_stopped = False


def _request_stop(signum: int, _frame: FrameType | None) -> None:
    global _stopped
    logger.info("Received signal %d, stopping after current scan.", signum)
    _stopped = True


def run_loop(interval: float) -> int:
    """Block forever, hiding target windows every ``interval`` seconds.

    Returns the total number of windows hidden until stopped.
    """
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    total_hidden = 0
    scan_count = 0
    logger.info("window_guard started (interval=%.2fs)", interval)

    while not _stopped:
        try:
            n = hide_target_windows()
        except Exception:
            # Never let one bad scan kill the guard
            logger.exception("hide_target_windows failed; continuing")
            n = 0
        if n > 0:
            total_hidden += n
            logger.info("scan %d: hid %d window(s)", scan_count, n)
        scan_count += 1
        # Periodic heartbeat so the user knows it is alive
        if scan_count % 600 == 0:  # every ~5 min at 0.5s
            logger.info(
                "heartbeat: %d scans, %d total hides", scan_count, total_hidden
            )
        time.sleep(interval)

    logger.info(
        "window_guard stopped. %d scans, %d total hides.", scan_count, total_hidden
    )
    return total_hidden


def main() -> int:
    args = parse_args()
    setup_logging(args.log, args.quiet)
    run_loop(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
