"""One-shot: hide SC2 / Battle.net windows. They keep running off-screen."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.window_hider import hide_target_windows  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    n = hide_target_windows()
    logging.info("Hidden %d window(s)", n)
    if n == 0:
        logging.info("(no visible target windows found - already hidden, or none running)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
