"""One-shot: restore previously hidden SC2 / Battle.net windows."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.window_hider import show_target_windows  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    n = show_target_windows()
    logging.info("Restored %d window(s)", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
