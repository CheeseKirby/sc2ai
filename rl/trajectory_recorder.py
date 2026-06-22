"""JSONL trajectory recording for rule-bot data collection."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class TrajectoryStep:
    """One high-level decision record."""

    episode_id: str
    step: int
    map_name: str
    difficulty: str
    opponent_race: str
    observation: dict[str, float]
    action: int
    action_name: str
    reward: float = 0.0
    done: bool = False
    result: str | None = None


class JsonlTrajectoryRecorder:
    """Append trajectory steps to a JSON Lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("a", encoding="utf-8")

    def record(self, step: TrajectoryStep) -> None:
        self._file.write(json.dumps(asdict(step), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> JsonlTrajectoryRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

