"""Rule selector for tactic-pool experiments."""
from __future__ import annotations

from rl.tactics import TacticId, TacticPhase, TacticState


class RuleTacticSelector:
    """Choose a coarse tactic from build-labeled strategy observations.

    This selector is intentionally not wired into the default runtime path. It is
    a conservative building block for future explicit tactic-aware policies.
    """

    def __init__(self, *, min_tactic_duration: float = 90.0) -> None:
        self.min_tactic_duration = float(min_tactic_duration)

    def select(
        self,
        observation: dict[str, float],
        *,
        opponent_ai_build: str = "RandomBuild",
        previous_state: TacticState | None = None,
    ) -> TacticState:
        """Return the selected tactic state for the current observation."""
        game_time = _value(observation, "game_time")
        tactic, reason = self._select_candidate(observation, opponent_ai_build)
        phase = _phase_for(tactic, game_time)

        if previous_state is None:
            return TacticState(
                current_tactic=tactic,
                phase=phase,
                started_game_time=game_time,
                last_switch_game_time=game_time,
                last_switch_reason=reason,
            )
        if previous_state.current_tactic is tactic:
            return TacticState(
                current_tactic=tactic,
                phase=phase,
                started_game_time=previous_state.started_game_time,
                last_switch_game_time=previous_state.last_switch_game_time,
                last_switch_reason=reason,
                previous_tactic=previous_state.previous_tactic,
            )

        elapsed = game_time - previous_state.last_switch_game_time
        if elapsed < self.min_tactic_duration and tactic not in _EMERGENCY_TACTICS:
            return TacticState(
                current_tactic=previous_state.current_tactic,
                phase=_phase_for(previous_state.current_tactic, game_time),
                started_game_time=previous_state.started_game_time,
                last_switch_game_time=previous_state.last_switch_game_time,
                last_switch_reason="cooldown",
                previous_tactic=previous_state.previous_tactic,
            )

        return TacticState(
            current_tactic=tactic,
            phase=phase,
            started_game_time=game_time,
            last_switch_game_time=game_time,
            last_switch_reason=reason,
            previous_tactic=previous_state.current_tactic,
        )

    def _select_candidate(
        self,
        observation: dict[str, float],
        opponent_ai_build: str,
    ) -> tuple[TacticId, str]:
        if _recovery_needed(observation):
            return TacticId.RECOVERY, "recovery"
        if opponent_ai_build == "Air" or _value(observation, "enemy_air_units_known") > 0.0:
            return TacticId.ANTI_AIR_RESPONSE, "air_response"
        if (
            opponent_ai_build == "Rush"
            or (
                _value(observation, "game_time") <= 240.0
                and _value(observation, "base_under_threat") > 0.0
            )
        ):
            return TacticId.ANTI_RUSH_DEFENSE, "early_threat_or_rush_build"
        if (
            _value(observation, "enemy_cloaked_units_seen") > 0.0
            or _value(observation, "enemy_armored_units_known") > 0.0
        ):
            return TacticId.ROBO_TIMING, "robo_signal"
        if opponent_ai_build == "Power" and _value(observation, "game_time") >= 240.0:
            return TacticId.TECH_POWER, "power_build_midgame"
        if opponent_ai_build == "Timing" and _value(observation, "game_time") >= 240.0:
            return TacticId.ROBO_TIMING, "timing_build_midgame"
        if (
            opponent_ai_build == "Macro"
            and _value(observation, "base_under_threat") <= 0.0
            and _value(observation, "army_count") >= 8.0
        ):
            return TacticId.GATEWAY_PRESSURE, "macro_pressure"
        return TacticId.SAFE_MACRO, "safe_macro"


_EMERGENCY_TACTICS = {
    TacticId.RECOVERY,
    TacticId.ANTI_RUSH_DEFENSE,
    TacticId.ANTI_AIR_RESPONSE,
}


def _recovery_needed(observation: dict[str, float]) -> bool:
    return (
        _value(observation, "own_bases") <= 0.0
        or _value(observation, "workers") <= 4.0
        or (
            _value(observation, "army_count") <= 2.0
            and _value(observation, "game_time") >= 240.0
        )
    )


def _phase_for(tactic: TacticId, game_time: float) -> TacticPhase:
    if tactic is TacticId.RECOVERY:
        return TacticPhase.RECOVERY
    if game_time < 240.0:
        return TacticPhase.OPENING
    if tactic in {TacticId.ROBO_TIMING, TacticId.TECH_POWER}:
        return TacticPhase.POWER_SPIKE
    if tactic in {TacticId.GATEWAY_PRESSURE, TacticId.ANTI_AIR_RESPONSE}:
        return TacticPhase.ATTACK_WINDOW
    return TacticPhase.STABILIZE


def _value(observation: dict[str, float], field: str) -> float:
    return float(observation.get(field, 0.0))
