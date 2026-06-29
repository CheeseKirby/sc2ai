"""JSONL trajectory recording for rule-bot data collection."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, TextIO


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
    opponent_ai_build: str = "RandomBuild"


@dataclass(frozen=True)
class StrategyTrajectoryStep:
    """One low-frequency macro strategy decision record."""

    episode_id: str
    step: int
    map_name: str
    difficulty: str
    opponent_race: str
    strategy_observation: dict[str, float]
    strategy_action: int
    strategy_action_name: str
    strategy_observation_details: dict[str, float] | None = None
    strategy_policy_source: str | None = None
    strategy_policy_reason: str | None = None
    army_observation: dict[str, float] | None = None
    army_action: int | None = None
    army_action_name: str | None = None
    reward: float = 0.0
    done: bool = False
    result: str | None = None
    opponent_ai_build: str = "RandomBuild"
    tactic_id: str | None = None
    tactic_phase: str | None = None
    tactic_source: str | None = None
    tactic_started_game_time: float | None = None
    tactic_switch_reason: str | None = None
    tactic_previous_id: str | None = None
    strategy_action_before_tactic_filter: int | None = None
    strategy_action_before_tactic_filter_name: str | None = None
    strategy_action_after_tactic_filter: int | None = None
    strategy_action_after_tactic_filter_name: str | None = None
    strategy_execution_attempted: bool | None = None
    strategy_execution_effect: str | None = None
    strategy_execution_blocker: str | None = None
    strategy_execution_unit_type: str | None = None
    strategy_execution_target: str | None = None


class JsonlTrajectoryRecorder:
    """Append trajectory steps to a JSON Lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("a", encoding="utf-8")

    def record(self, step: Any) -> None:
        payload = asdict(step) if is_dataclass(step) else dict(step)
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> JsonlTrajectoryRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
