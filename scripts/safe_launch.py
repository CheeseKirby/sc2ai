"""
Safe launcher: ensure window_guard is running, then start the bot.

Why this exists:
  - Guarantees the guard is up BEFORE the bot launches SC2 (closes the race
    where SC2's window flashes onto the screen for a frame before the bot's
    inline hide-watcher catches it).
  - The guard is an independent process: even if the bot crashes, the guard
    keeps suppressing any zombie window that survives.
  - On Ctrl+C, only the bot is stopped. The guard keeps running so SC2 is
    still suppressed during cleanup.

Usage:
    .venv/Scripts/python.exe scripts/safe_launch.py -- --difficulty Easy
    .venv/Scripts/python.exe scripts/safe_launch.py --keep-guard -- --realtime

Everything after `--` is forwarded to run.py.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
# pythonw.exe is the GUI subsystem build of CPython: it does NOT allocate a
# console window. Use it for daemons (like window_guard) so they truly run
# silently with no popup terminal.
PYTHONW = PROJECT_ROOT / ".venv" / "Scripts" / "pythonw.exe"
GUARD_SCRIPT = PROJECT_ROOT / "scripts" / "window_guard.py"
RUN_SCRIPT = PROJECT_ROOT / "run.py"
LOGS_DIR = PROJECT_ROOT / "logs"
GUARD_PID_FILE = LOGS_DIR / "window_guard.pid"
GUARD_LOG_FILE = LOGS_DIR / "window_guard.log"

logger = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=(
            "Launch the bot with window_guard already running. "
            "Pass `--` then any run.py flags."
        ),
        allow_abbrev=False,
    )
    p.add_argument(
        "--keep-guard",
        action="store_true",
        help=(
            "Do NOT stop window_guard after the bot exits. Useful when you want "
            "the guard to keep suppressing windows between matches."
        ),
    )
    p.add_argument(
        "--guard-interval",
        type=float,
        default=0.05,
        help=(
            "window_guard polling interval (default: 0.05s). At 0.05s the "
            "maximum visible window flash is ~50ms; CPU stays near zero."
        ),
    )
    return p.parse_known_args()


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def is_pid_alive(pid: int) -> bool:
    """Cross-platform-ish: on Windows, signal 0 is not portable; use tasklist."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    return str(pid) in out


def find_existing_guard() -> int | None:
    """Scan running processes for an active window_guard.py.

    Uses psutil for reliability. To avoid false positives from other shells
    that mention "window_guard.py" in their command text, we require:
      - argv[0] is a Python interpreter (python.exe or pythonw.exe)
      - argv[1] ends with window_guard.py (it is the script path, not a substring
        appearing somewhere else in the command line)
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return None

    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmdline = proc.info.get("cmdline") or []
        if len(cmdline) < 2:
            continue
        exe = cmdline[0].lower()
        if not (exe.endswith("python.exe") or exe.endswith("pythonw.exe")):
            continue
        if cmdline[1].lower().endswith("window_guard.py"):
            return int(proc.info["pid"])
    return None


def ensure_guard_running(interval: float) -> subprocess.Popen | None:
    """Start window_guard if not already running. Return the Popen if we
    started one (so the caller can decide whether to stop it later)."""
    # Trust process scan over the PID file: PID files can be wrong (e.g. when
    # the guard was started from bash and the recorded PID is the bash one,
    # not the Windows process PID).
    existing = find_existing_guard()
    if existing is not None:
        logger.info("window_guard already running (PID %d)", existing)
        GUARD_PID_FILE.write_text(str(existing))
        return None

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting window_guard...")
    # CREATE_NO_WINDOW + pythonw.exe = truly silent daemon with no console popup.
    # DETACHED_PROCESS / CREATE_NEW_PROCESS_GROUP keep the guard alive when this
    # launcher exits.
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_NO_WINDOW = 0x08000000
    proc = subprocess.Popen(
        [
            str(PYTHONW),
            str(GUARD_SCRIPT),
            "--interval",
            str(interval),
            "--log",
            str(GUARD_LOG_FILE),
            "--quiet",
        ],
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        cwd=str(PROJECT_ROOT),
    )
    GUARD_PID_FILE.write_text(str(proc.pid))
    logger.info("window_guard started (PID %d, log=%s)", proc.pid, GUARD_LOG_FILE)
    # Give it a moment to install its signal handler and do one scan
    time.sleep(1.0)
    return proc


def stop_guard() -> None:
    # Prefer the active process scan over the PID file - more reliable.
    pid = find_existing_guard()
    if pid is None:
        pid = read_pid(GUARD_PID_FILE)
    if pid is None:
        return
    if not is_pid_alive(pid):
        GUARD_PID_FILE.unlink(missing_ok=True)
        return
    logger.info("Stopping window_guard (PID %d)...", pid)
    subprocess.run(
        ["taskkill", "/F", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    GUARD_PID_FILE.unlink(missing_ok=True)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args, forwarded = parse_args()
    # Strip leading "--" if user typed it explicitly
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    ensure_guard_running(args.guard_interval)

    logger.info("Launching bot: run.py %s", " ".join(forwarded))
    bot_proc = subprocess.Popen(
        [str(PYTHON), str(RUN_SCRIPT), *forwarded],
        cwd=str(PROJECT_ROOT),
    )

    bot_exit = 0
    try:
        bot_exit = bot_proc.wait()
    except KeyboardInterrupt:
        logger.info("Ctrl+C received: terminating bot (guard kept alive)...")
        bot_proc.send_signal(signal.SIGTERM)
        try:
            bot_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            bot_proc.kill()
        bot_exit = 130

    logger.info("Bot exited with code %d", bot_exit)

    if not args.keep_guard:
        stop_guard()
    else:
        logger.info("--keep-guard set: leaving window_guard running.")

    return bot_exit


if __name__ == "__main__":
    raise SystemExit(main())
